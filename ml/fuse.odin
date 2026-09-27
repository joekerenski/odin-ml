package ml

// ============================================================================
// Fused elementwise kernel. A group of same-shape ewise UOps becomes a tiny
// register program (insns over slots) run in ONE loop over the output:
//
//   slots[0..n_in)      loads of external inputs (broadcast by load mode)
//   slots[n_in + j]     result of insn j (one per group node, topo order)
//   stores              slots written back to buffers someone still needs
//
// SIMD (f32x4) when every load is vectorizable; scalar loop otherwise.
// ============================================================================

import "base:intrinsics"
import "core:math"
import "core:simd"

MAX_FUSED_INSNS :: 16
MAX_FUSED_INPUTS :: 8
MAX_FUSED_SLOTS :: MAX_FUSED_INPUTS + MAX_FUSED_INSNS

// How input element i of the output maps into an input buffer.
Load_Mode :: enum {
	Direct,  // data[i]
	Scalar,  // data[0]
	Row,     // data[i % n]            [B,N] ← [N]
	Col,     // data[i / inner]        [B,N] ← [B,1]
	Block,   // data[(i / inner) % n]  [N,C,H,W] ← [1,C,1,1]
	Generic, // unravel + broadcast index
}

Fused_In :: struct {
	mode:  Load_Mode,
	data:  []f32,
	n:     int,
	inner: int,
	shape: []i32,
}

Fused_Insn :: struct {
	op:   Op,
	a, b: int, // slot indices
}

Fused_Store :: struct {
	slot: int,
	data: []f32,
}

// Right-align `shape` under `out` and pick the cheapest load.
classify_load :: proc(shape, out: []i32) -> (mode: Load_Mode, n, inner: int) {
	n = int(numel(shape))
	if n == int(numel(out)) do return .Direct, n, 0
	if n == 1 do return .Scalar, n, 0
	pad := len(out) - len(shape)
	lo, hi := -1, -1
	for d in 0 ..< len(shape) {
		if shape[d] != 1 {
			if lo < 0 do lo = pad + d
			hi = pad + d + 1
		}
	}
	for d in lo ..< hi {
		if shape[d - pad] != out[d] do return .Generic, n, 0
	}
	inner = int(numel(out[hi:]))
	if inner == 1 do return .Row, n, 1
	if numel(out[:lo]) == 1 do return .Col, n, inner
	return .Block, n, inner
}

flat_of_shape :: proc(idx: []i32, odim: int, shape: []i32) -> int {
	s := 0
	st := 1
	for d := len(shape) - 1; d >= 0; d -= 1 {
		if shape[d] != 1 do s += int(idx[odim - len(shape) + d]) * st
		st *= int(shape[d])
	}
	return s
}

// Run `group` as one kernel, writing `stores`. Returns false (without doing
// anything) if the group exceeds the kernel's register budget.
run_fused :: proc(group: []^UOp, stores: []^UOp) -> bool {
	if len(group) > MAX_FUSED_INSNS do return false
	out_shape := group[len(group) - 1].shape

	slot_of: map[^UOp]int
	defer delete(slot_of)
	inputs: [MAX_FUSED_INPUTS]Fused_In
	n_in := 0
	for u in group do slot_of[u] = -1
	for u in group {
		for x in u.src {
			if x in slot_of do continue
			if n_in == MAX_FUSED_INPUTS do return false
			mode, n, inner := classify_load(x.shape, out_shape)
			inputs[n_in] = Fused_In{mode, x.data, n, inner, x.shape}
			slot_of[x] = n_in
			n_in += 1
		}
	}

	insns: [MAX_FUSED_INSNS]Fused_Insn
	for u, j in group {
		b := op_is_binary(u.op) ? slot_of[u.src[1]] : 0
		insns[j] = Fused_Insn{u.op, slot_of[u.src[0]], b}
		slot_of[u] = n_in + j
	}

	outs: [MAX_FUSED_INSNS]Fused_Store
	for u, k in stores {
		alloc_out(u)
		outs[k] = Fused_Store{slot_of[u], u.data}
	}

	run_fused_kernel(inputs[:n_in], insns[:len(group)], outs[:len(stores)], out_shape)
	return true
}

run_fused_kernel :: proc(inputs: []Fused_In, insns: []Fused_Insn, stores: []Fused_Store, out_shape: []i32) {
	n := int(numel(out_shape))
	simd_ok := true
	for insn in insns do if insn.op == .Exp do simd_ok = false
	row_n, col_inner := 0, 0
	for inp in inputs {
		switch inp.mode {
		case .Direct, .Scalar:
		case .Row:
			if row_n != 0 && row_n != inp.n do simd_ok = false
			row_n = inp.n
		case .Col:
			if col_inner != 0 && col_inner != inp.inner do simd_ok = false
			col_inner = inp.inner
		case .Block, .Generic:
			simd_ok = false
		}
	}
	if row_n != 0 && col_inner != 0 do simd_ok = false

	if simd_ok && len(insns) == 1 && row_n == 0 && col_inner == 0 && len(stores) == 1 {
		run_single_simd(inputs, insns[0], stores[0].data, n)
		return
	}

	switch {
	case !simd_ok:
		run_fused_scalar(inputs, insns, stores, out_shape, n)
	case row_n != 0:
		run_fused_rows_simd(inputs, insns, stores, n, row_n)
	case col_inner != 0:
		run_fused_rows_simd(inputs, insns, stores, n, col_inner)
	case:
		run_fused_flat_simd(inputs, insns, stores, n)
	}
}

fused_eval :: #force_inline proc(op: Op, a, b: f32) -> f32 {
	#partial switch op {
	case .Add: return a + b
	case .Sub: return a - b
	case .Mul: return a * b
	case .Div: return a / b
	case .Max: return a > b ? a : b
	case .CmpLt: return a < b ? 1 : 0
	case .Neg: return -a
	case .Exp: return math.exp(a)
	case .Expand: return a
	}
	panic("fused_eval: not ewise")
}

fused_eval_simd :: #force_inline proc(op: Op, a, b: simd.f32x4) -> simd.f32x4 {
	#partial switch op {
	case .Add: return simd.add(a, b)
	case .Sub: return simd.sub(a, b)
	case .Mul: return simd.mul(a, b)
	case .Div: return simd.div(a, b)
	case .Max: return simd.max(a, b)
	case .CmpLt: return simd.select(simd.lanes_lt(a, b), simd.f32x4(1), simd.f32x4(0))
	case .Neg: return simd.neg(a)
	case .Expand: return a
	}
	panic("fused_eval_simd: not simd ewise")
}

fused_load :: #force_inline proc(inp: Fused_In, i: int) -> f32 {
	switch inp.mode {
	case .Direct: return inp.data[i]
	case .Scalar: return inp.data[0]
	case .Row: return inp.data[i % inp.n]
	case .Col: return inp.data[i / inp.inner]
	case .Block: return inp.data[(i / inp.inner) % inp.n]
	case .Generic: // needs the multi-index; handled by the caller
	}
	return 0
}

run_fused_scalar :: proc(inputs: []Fused_In, insns: []Fused_Insn, stores: []Fused_Store, out_shape: []i32, n: int) {
	generic := false
	for inp in inputs do if inp.mode == .Generic do generic = true
	n_in := len(inputs)
	slots: [MAX_FUSED_SLOTS]f32
	idx: [MAX_DIMS]i32
	for i in 0 ..< n {
		if generic do unravel_index(i32(i), out_shape, idx[:])
		for inp, k in inputs {
			if inp.mode == .Generic {
				slots[k] = inp.data[flat_of_shape(idx[:], len(out_shape), inp.shape)]
			} else {
				slots[k] = fused_load(inp, i)
			}
		}
		for insn, j in insns do slots[n_in + j] = fused_eval(insn.op, slots[insn.a], slots[insn.b])
		for s in stores do s.data[i] = slots[s.slot]
	}
}

run_fused_flat_simd :: proc(inputs: []Fused_In, insns: []Fused_Insn, stores: []Fused_Store, n: int) {
	n_in := len(inputs)
	slots: [MAX_FUSED_SLOTS]simd.f32x4
	i := 0
	for ; i + 4 <= n; i += 4 {
		for inp, k in inputs {
			if inp.mode == .Scalar {
				slots[k] = inp.data[0]
			} else {
				slots[k] = intrinsics.unaligned_load((^simd.f32x4)(&inp.data[i]))
			}
		}
		for insn, j in insns do slots[n_in + j] = fused_eval_simd(insn.op, slots[insn.a], slots[insn.b])
		for s in stores do intrinsics.unaligned_store((^simd.f32x4)(&s.data[i]), slots[s.slot])
	}
	fused_tail(inputs, insns, stores, i, n)
}

// One op over Direct/Scalar inputs: the op is a compile-time constant, so the
// loop body is a single SIMD instruction (no per-vector dispatch).
run_single_simd :: proc(inputs: []Fused_In, insn: Fused_Insn, out: []f32, n: int) {
	a := inputs[insn.a]
	b := inputs[insn.b]
	switch insn.op {
	case .Add: single_loop(a, b, out, n, .Add)
	case .Sub: single_loop(a, b, out, n, .Sub)
	case .Mul: single_loop(a, b, out, n, .Mul)
	case .Div: single_loop(a, b, out, n, .Div)
	case .Max: single_loop(a, b, out, n, .Max)
	case .CmpLt: single_loop(a, b, out, n, .CmpLt)
	case .Neg: single_loop(a, b, out, n, .Neg)
	case .Expand: single_loop(a, b, out, n, .Expand)
	case .Input, .Const, .Exp, .Sum, .Reshape, .Permute, .MatMul, .Conv2d, .Conv2dBwdInput,
	     .Conv2dBwdWeight, .MaxPool2d, .MaxPool2dBwd, .CrossEntropy, .CrossEntropyBwd:
		panic("run_single_simd: not a simd ewise op")
	}
}

single_loop :: #force_inline proc(a, b: Fused_In, out: []f32, n: int, $op: Op) {
	load :: #force_inline proc(x: Fused_In, i: int) -> simd.f32x4 {
		if x.mode == .Scalar do return x.data[0]
		return intrinsics.unaligned_load((^simd.f32x4)(&x.data[i]))
	}
	i := 0
	if a.mode == .Direct && b.mode == .Direct {
		for ; i + 4 <= n; i += 4 {
			va := intrinsics.unaligned_load((^simd.f32x4)(&a.data[i]))
			vb := intrinsics.unaligned_load((^simd.f32x4)(&b.data[i]))
			intrinsics.unaligned_store((^simd.f32x4)(&out[i]), fused_eval_simd(op, va, vb))
		}
	} else {
		for ; i + 4 <= n; i += 4 {
			intrinsics.unaligned_store((^simd.f32x4)(&out[i]), fused_eval_simd(op, load(a, i), load(b, i)))
		}
	}
	for ; i < n; i += 1 do out[i] = fused_eval(op, fused_load(a, i), fused_load(b, i))
}

// Rows of length `inner`. Row inputs load data[j]; Col inputs splat data[row].
run_fused_rows_simd :: proc(inputs: []Fused_In, insns: []Fused_Insn, stores: []Fused_Store, n, inner: int) {
	n_in := len(inputs)
	slots: [MAX_FUSED_SLOTS]simd.f32x4
	for row in 0 ..< n / inner {
		base := row * inner
		j := 0
		for ; j + 4 <= inner; j += 4 {
			for inp, k in inputs {
				#partial switch inp.mode {
				case .Direct: slots[k] = intrinsics.unaligned_load((^simd.f32x4)(&inp.data[base + j]))
				case .Row: slots[k] = intrinsics.unaligned_load((^simd.f32x4)(&inp.data[j]))
				case .Col: slots[k] = inp.data[row]
				case .Scalar: slots[k] = inp.data[0]
				}
			}
			for insn, t in insns do slots[n_in + t] = fused_eval_simd(insn.op, slots[insn.a], slots[insn.b])
			for s in stores do intrinsics.unaligned_store((^simd.f32x4)(&s.data[base + j]), slots[s.slot])
		}
		fused_tail(inputs, insns, stores, base + j, base + inner)
	}
}

fused_tail :: proc(inputs: []Fused_In, insns: []Fused_Insn, stores: []Fused_Store, from, to: int) {
	n_in := len(inputs)
	ss: [MAX_FUSED_SLOTS]f32
	for i in from ..< to {
		for inp, k in inputs do ss[k] = fused_load(inp, i)
		for insn, j in insns do ss[n_in + j] = fused_eval(insn.op, ss[insn.a], ss[insn.b])
		for s in stores do s.data[i] = ss[s.slot]
	}
}
