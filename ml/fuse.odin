package ml

// ============================================================================
// Fused elementwise kernel. A group of same-shape ewise UOps becomes a tiny
// program (insns over slots) run over the output in CHUNKs:
//
//   slots[0..n_in)      each input's values for this chunk (broadcast by load
//                       mode; Direct inputs are just pointers, no copy)
//   slots[n_in + j]     result of insn j (one per group node, topo order)
//   stores              results someone still needs go straight to their buffer
//
// Per chunk, each insn is one tight SIMD loop with the op fixed at compile
// time, so dispatch costs once per CHUNK elements, not once per vector.
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

	slot_of := make(map[^UOp]int, scratch())
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

	job := Fused_Job{inputs[:n_in], insns[:len(group)], outs[:len(stores)], out_shape}
	backend.fused(&job)
	return true
}

CHUNK :: 256
PAR_GRAIN :: 32 * 1024 // elements per thread part, below this: one thread

Fused_Job :: struct {
	inputs:    []Fused_In,
	insns:     []Fused_Insn,
	stores:    []Fused_Store,
	out_shape: []i32,
}

// CPU backend: split the output across cores.
run_fused_kernel :: proc(job: ^Fused_Job) {
	parallel_for(int(numel(job.out_shape)), PAR_GRAIN, run_fused_range, job)
}

// Output elements [lo, hi), CHUNK at a time.
run_fused_range :: proc(data: rawptr, lo, hi: int) {
	using job := (^Fused_Job)(data)
	n_in := len(inputs)
	buf: [MAX_FUSED_SLOTS][CHUNK]f32 // chunk scratch (L1-resident)
	ptr: [MAX_FUSED_SLOTS][^]f32 // where each slot's chunk values live
	store_of: [MAX_FUSED_SLOTS]int // slot → store index + 1 (0: not stored)
	for st, k in stores do store_of[st.slot] = k + 1

	for inp, k in inputs {
		if inp.mode == .Scalar {
			for &v in buf[k] do v = inp.data[0]
			ptr[k] = &buf[k][0]
		}
	}
	idx: [MAX_DIMS]i32
	for base := lo; base < hi; base += CHUNK {
		m := min(CHUNK, hi - base)
		for inp, k in inputs {
			b := &buf[k]
			switch inp.mode {
			case .Scalar:
			case .Direct: ptr[k] = &inp.data[base]
			case .Row:
				j := base % inp.n
				for i in 0 ..< m {
					b[i] = inp.data[j]
					j += 1
					if j == inp.n do j = 0
				}
				ptr[k] = &b[0]
			case .Col:
				for i in 0 ..< m do b[i] = inp.data[(base + i) / inp.inner]
				ptr[k] = &b[0]
			case .Block:
				for i in 0 ..< m do b[i] = inp.data[((base + i) / inp.inner) % inp.n]
				ptr[k] = &b[0]
			case .Generic:
				for i in 0 ..< m {
					unravel_index(i32(base + i), out_shape, idx[:])
					b[i] = inp.data[flat_of_shape(idx[:], len(out_shape), inp.shape)]
				}
				ptr[k] = &b[0]
			}
		}
		for insn, j in insns {
			slot := n_in + j
			dst: [^]f32 = &buf[slot][0]
			if st := store_of[slot]; st > 0 do dst = &stores[st - 1].data[base]
			run_chunk(insn.op, dst, ptr[insn.a], ptr[insn.b], m)
			ptr[slot] = dst
		}
	}
}

// dst[i] = op(a[i], b[i]) for one chunk. b is ignored by unary ops.
run_chunk :: proc(op: Op, dst, a, b: [^]f32, m: int) {
	#partial switch op {
	case .Add: chunk_loop(dst, a, b, m, .Add)
	case .Sub: chunk_loop(dst, a, b, m, .Sub)
	case .Mul: chunk_loop(dst, a, b, m, .Mul)
	case .Div: chunk_loop(dst, a, b, m, .Div)
	case .Max: chunk_loop(dst, a, b, m, .Max)
	case .CmpLt: chunk_loop(dst, a, b, m, .CmpLt)
	case .Neg: chunk_loop(dst, a, b, m, .Neg)
	case .Sqrt: chunk_loop(dst, a, b, m, .Sqrt)
	case .Expand: chunk_loop(dst, a, b, m, .Expand)
	case .Exp: for i in 0 ..< m do dst[i] = math.exp(a[i])
	case .Log: for i in 0 ..< m do dst[i] = math.ln(a[i])
	case: panic("run_chunk: not an elementwise op")
	}
}

chunk_loop :: #force_inline proc(dst, a, b: [^]f32, m: int, $op: Op) {
	i := 0
	for ; i + 4 <= m; i += 4 {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		vb := intrinsics.unaligned_load((^simd.f32x4)(&b[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&dst[i]), fused_eval_simd(op, va, vb))
	}
	for ; i < m; i += 1 do dst[i] = fused_eval(op, a[i], b[i])
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
	case .Sqrt: return math.sqrt(a)
	case .Expand: return a
	}
	panic("fused_eval: not a simd ewise op")
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
	case .Sqrt: return simd.sqrt(a)
	case .Expand: return a
	}
	panic("fused_eval_simd: not a simd ewise op")
}
