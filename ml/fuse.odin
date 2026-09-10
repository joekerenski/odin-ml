package ml

// ============================================================================
// Schedule: UOp DAG → kernels. Consecutive same-shape ewise ops with a
// single consumer fuse into one loop. Everything else is one kernel
// (MatMul/Sum/Conv/CE) or a view (Reshape/Transpose).
// ============================================================================

import "base:intrinsics"
import "core:fmt"
import "core:math"
import "core:simd"
import "core:time"

MAX_FUSED_INSNS :: 16
MAX_FUSED_INPUTS :: 8
MAX_FUSED_SLOTS :: 24
MAX_FUSED_STORES :: 8

Load_Mode :: enum {
	Direct,  // data[i]
	Scalar,  // data[0] or Const
	Row,     // data[i % inner]  — [B,N] + [N]
	Col,     // data[i / inner]  — [B,N] + [B,1]
	Generic,
}

Fused_In :: struct {
	kind:      enum {
		Buffer,
		Const,
	},
	mode:      Load_Mode,
	data:      []f32,
	const_val: f32,
	inner:     i32,
	shape:     [MAX_DIMS]i32,
	ndim:      int,
}

Fused_Insn :: struct {
	op:   UOps,
	a, b: i32, // slot indices
}

Fused_Store :: struct {
	slot: i32,
	data: []f32,
}

uf_find :: proc(uf: ^map[^UOp]^UOp, x: ^UOp) -> ^UOp {
	p, ok := uf[x]
	if !ok do return x
	if p != x {
		p = uf_find(uf, p)
		uf[x] = p
	}
	return p
}

uf_union :: proc(uf: ^map[^UOp]^UOp, a, b: ^UOp) {
	ra, rb := uf_find(uf, a), uf_find(uf, b)
	if ra != rb do uf[ra] = rb
}

cpu_ewise :: proc(u: ^UOp) -> bool {
	if !uop_is_ewise(u.op) do return false
	if u.tensor != nil && u.tensor.device == .Metal do return false
	return true
}

op_needs_parent_data :: proc(op: Op) -> bool {
	#partial switch op {
	case .Mul, .Div, .MatMul, .Conv2d:
		return true
	}
	return false
}

op_needs_own_data :: proc(op: Op) -> bool {
	return op == .ReLU || op == .Sigmoid
}

should_store :: proc(u: ^UOp, group: []^UOp, sink: ^Tensor, consumers: map[^UOp]int) -> bool {
	t := u.tensor
	if t == nil do return false
	if t == sink do return true
	in_c := 0
	for v in group {
		seen := false
		for s in v.src {
			if s == u && !seen {
				in_c += 1
				seen = true
			}
		}
	}
	if in_c < consumers[u] do return true
	if t.ctx != nil && op_needs_own_data(t.ctx.op) && t.requires_grad do return true
	if t.requires_grad {
		for v in group {
			if v.tensor == nil || v.tensor.ctx == nil do continue
			if !op_needs_parent_data(v.tensor.ctx.op) do continue
			for s in v.src {
				if s == u do return true
			}
		}
	}
	return false
}

execute_uop_graph :: proc(root: ^UOp) {
	topo: [dynamic]^UOp
	visited: map[^UOp]bool
	defer delete(topo)
	defer delete(visited)
	uop_topo(root, &topo, &visited)

	consumers: map[^UOp]int
	defer delete(consumers)
	for u in topo {
		for s, i in u.src {
			dup := false
			for j in 0 ..< i {
				if u.src[j] == s {
					dup = true
					break
				}
			}
			if dup do continue
			consumers[s] += 1
		}
	}

	uf: map[^UOp]^UOp
	defer delete(uf)
	for u in topo {
		if cpu_ewise(u) do uf[u] = u
	}
	for u in topo {
		if u not_in uf do continue
		for s in u.src {
			if s not_in uf do continue
			if consumers[s] != 1 do continue
			if !shapes_equal(s.shape[:], u.shape[:]) do continue
			uf_union(&uf, s, u)
		}
	}

	last: map[^UOp]^UOp
	defer delete(last)
	for u in topo {
		if u not_in uf do continue
		last[uf_find(&uf, u)] = u
	}

	executed: map[^UOp]bool
	defer delete(executed)

	for u in topo {
		if u.op == .Input || u.op == .Const do continue
		if u in executed do continue

		if uop_is_view(u.op) {
			assert(u.tensor != nil, "view uop needs tensor")
			realize_one(u.tensor)
			u.data = u.tensor.data
			executed[u] = true
			continue
		}

		if u in uf {
			if last[uf_find(&uf, u)] != u do continue
			group: [dynamic]^UOp
			root_g := uf_find(&uf, u)
			for v in topo {
				if v in uf && uf_find(&uf, v) == root_g {
					append(&group, v)
				}
			}
			run_ewise_group(group[:], root.tensor, consumers)
			for v in group {
				executed[v] = true
				if v.tensor != nil {
					v.tensor.done = true
					v.data = v.tensor.data
				}
			}
			delete(group)
			continue
		}

		assert(u.tensor != nil, "heavy uop needs tensor")
		realize_one(u.tensor)
		u.data = u.tensor.data
		executed[u] = true
	}
}

run_ewise_group :: proc(group: []^UOp, sink: ^Tensor, consumers: map[^UOp]int) {
	if len(group) == 0 do return
	if len(group) == 1 && group[0].tensor != nil {
		realize_one(group[0].tensor)
		return
	}

	in_group: map[^UOp]bool
	defer delete(in_group)
	for u in group do in_group[u] = true

	externals: [dynamic]^UOp
	seen: map[^UOp]bool
	defer delete(externals)
	defer delete(seen)
	for u in group {
		for s in u.src {
			if s in in_group do continue
			if s in seen do continue
			seen[s] = true
			append(&externals, s)
		}
	}

	if len(externals) > MAX_FUSED_INPUTS || len(group) > MAX_FUSED_INSNS {
		for u in group {
			if u.tensor != nil do realize_one(u.tensor)
		}
		return
	}

	out_u := group[len(group) - 1]
	out_shape := out_u.shape[:]

	slot_of: map[^UOp]i32
	defer delete(slot_of)
	inputs: [MAX_FUSED_INPUTS]Fused_In
	n_in := 0
	for s in externals {
		fi: Fused_In
		if s.op == .Const {
			fi.kind = .Const
			fi.mode = .Scalar
			fi.const_val = s.arg.(f32)
		} else {
			data := s.data
			contig := true
			if s.tensor != nil {
				data = contig_data(s.tensor)
				contig = true
			}
			fi.kind = .Buffer
			fi.data = data
			fi.ndim = len(s.shape)
			for d in 0 ..< fi.ndim do fi.shape[d] = s.shape[d]
			fi.mode, fi.inner = classify_load(s.shape[:], out_shape, contig)
			if fi.mode == .Scalar && len(data) > 0 do fi.const_val = data[0]
		}
		inputs[n_in] = fi
		slot_of[s] = i32(n_in)
		n_in += 1
	}

	insns: [MAX_FUSED_INSNS]Fused_Insn
	n_insns := 0
	for u, i in group {
		a := slot_of[u.src[0]]
		b: i32 = 0
		if uop_is_binary(u.op) {
			b = slot_of[u.src[1]]
		}
		insns[n_insns] = Fused_Insn{op = u.op, a = a, b = b}
		slot_of[u] = i32(n_in + i)
		n_insns += 1
	}

	stores: [MAX_FUSED_STORES]Fused_Store
	n_stores := 0
	for u, i in group {
		if !should_store(u, group, sink, consumers) {
			if u.tensor != nil do u.tensor.done = true
			continue
		}
		t := u.tensor
		n := numel(t.shape[:])
		if t.data == nil {
			t.data = make([]f32, n)
			counters.bytes_alloc += i64(n) * size_of(f32)
		}
		assert(n_stores < MAX_FUSED_STORES)
		stores[n_stores] = Fused_Store{slot = i32(n_in + i), data = t.data}
		n_stores += 1
	}
	assert(n_stores > 0, "fused group has no tensor stores")

	t0: time.Tick
	if debug_level >= 2 do t0 = time.tick_now()

	run_fused_kernel(inputs[:n_in], insns[:n_insns], stores[:n_stores], out_shape, n_in)

	counters.kernels += 1
	if n_insns > 1 do counters.fused_ops += n_insns - 1

	if debug_level >= 2 {
		dt_ns := i64(time.tick_since(t0))
		counters.time_ns += dt_ns
		fmt.printf("  fwd     fused[")
		for u, i in group {
			if i > 0 do fmt.print(",")
			fmt.print(uop_name(u.op))
		}
		fmt.printfln("] shape=%v  %7.3f ms", out_u.shape, f64(dt_ns) / 1e6)
	}
}

classify_load :: proc(shape, out_shape: []i32, contig: bool) -> (Load_Mode, i32) {
	if is_scalar_shape(shape) do return .Scalar, 0
	if shapes_equal(shape, out_shape) && contig do return .Direct, 0
	if contig {
		if ok, inn := inner_match_size(out_shape, shape); ok && numel(shape) == inn && inn > 1 {
			return .Row, inn
		}
		if ok, _, inn := col_broadcast_match(out_shape, shape); ok && inn > 1 {
			return .Col, inn
		}
	}
	return .Generic, 0
}

fused_eval :: #force_inline proc(op: UOps, a, b: f32) -> f32 {
	switch op {
	case .Add: return a + b
	case .Sub: return a - b
	case .Mul: return a * b
	case .Div: return a / b
	case .Max: return a > b ? a : b
	case .Neg: return -a
	case .Exp: return math.exp(a)
	case .Const, .Input, .Sum, .Reshape, .Transpose, .MatMul, .Conv2d, .MaxPool2d, .CrossEntropy:
		panic("fused_eval: not ewise")
	}
	return 0
}

fused_eval_simd :: #force_inline proc(op: UOps, a, b: simd.f32x4) -> simd.f32x4 {
	#partial switch op {
	case .Add: return simd.add(a, b)
	case .Sub: return simd.sub(a, b)
	case .Mul: return simd.mul(a, b)
	case .Div: return simd.div(a, b)
	case .Max: return simd.max(a, b)
	case .Neg: return simd.neg(a)
	}
	panic("fused_eval_simd: not simd ewise")
}

fused_load :: #force_inline proc(inp: Fused_In, i: int) -> f32 {
	if inp.kind == .Const do return inp.const_val
	switch inp.mode {
	case .Direct: return inp.data[i]
	case .Scalar: return inp.data[0]
	case .Row: return inp.data[i % int(inp.inner)]
	case .Col: return inp.data[i / int(inp.inner)]
	case .Generic: return 0
	}
	return 0
}

run_fused_kernel :: proc(
	inputs: []Fused_In,
	insns: []Fused_Insn,
	stores: []Fused_Store,
	out_shape: []i32,
	n_in: int,
) {
	n := int(numel(out_shape))
	has_exp := false
	for insn in insns {
		if insn.op == .Exp do has_exp = true
	}
	any_row, any_col, any_generic := false, false, false
	for inp in inputs {
		#partial switch inp.mode {
		case .Row: any_row = true
		case .Col: any_col = true
		case .Generic: any_generic = true
		}
	}

	if has_exp || any_generic || (any_row && any_col) {
		run_fused_scalar(inputs, insns, stores, out_shape, n_in, n, any_generic)
		return
	}
	if any_row {
		inner := 0
		for inp in inputs {
			if inp.mode == .Row {
				inner = int(inp.inner)
				break
			}
		}
		if inner > 0 && n % inner == 0 {
			run_fused_row_simd(inputs, insns, stores, n_in, n, inner)
			return
		}
	}
	if any_col {
		inner := 0
		for inp in inputs {
			if inp.mode == .Col {
				inner = int(inp.inner)
				break
			}
		}
		if inner > 0 && n % inner == 0 {
			run_fused_col_simd(inputs, insns, stores, n_in, n, inner)
			return
		}
	}
	run_fused_flat_simd(inputs, insns, stores, n_in, n)
}

run_fused_scalar :: proc(
	inputs: []Fused_In,
	insns: []Fused_Insn,
	stores: []Fused_Store,
	out_shape: []i32,
	n_in, n: int,
	generic: bool,
) {
	slots: [MAX_FUSED_SLOTS]f32
	idx: [MAX_DIMS]i32
	odim := len(out_shape)
	for i in 0 ..< n {
		if generic {
			unravel_index(i32(i), out_shape, idx[:])
			for k in 0 ..< len(inputs) {
				inp := inputs[k]
				if inp.kind == .Const {
					slots[k] = inp.const_val
				} else if inp.mode == .Generic {
					slots[k] = inp.data[flat_of_shape(idx[:], odim, inp.shape[:inp.ndim])]
				} else {
					slots[k] = fused_load(inp, i)
				}
			}
		} else {
			for k in 0 ..< len(inputs) do slots[k] = fused_load(inputs[k], i)
		}
		for insn, j in insns {
			slots[n_in + j] = fused_eval(insn.op, slots[insn.a], slots[insn.b])
		}
		for s in stores do s.data[i] = slots[s.slot]
	}
}

run_fused_flat_simd :: proc(
	inputs: []Fused_In,
	insns: []Fused_Insn,
	stores: []Fused_Store,
	n_in, n: int,
) {
	slots: [MAX_FUSED_SLOTS]simd.f32x4
	i := 0
	for ; i + 4 <= n; i += 4 {
		for k in 0 ..< len(inputs) {
			inp := inputs[k]
			if inp.kind == .Const || inp.mode == .Scalar {
				slots[k] = inp.kind == .Const ? inp.const_val : inp.data[0]
			} else {
				slots[k] = intrinsics.unaligned_load((^simd.f32x4)(&inp.data[i]))
			}
		}
		for insn, j in insns {
			slots[n_in + j] = fused_eval_simd(insn.op, slots[insn.a], slots[insn.b])
		}
		for s in stores {
			intrinsics.unaligned_store((^simd.f32x4)(&s.data[i]), slots[s.slot])
		}
	}
	ss: [MAX_FUSED_SLOTS]f32
	for ; i < n; i += 1 {
		for k in 0 ..< len(inputs) do ss[k] = fused_load(inputs[k], i)
		for insn, j in insns {
			ss[n_in + j] = fused_eval(insn.op, ss[insn.a], ss[insn.b])
		}
		for s in stores do s.data[i] = ss[s.slot]
	}
}

run_fused_row_simd :: proc(
	inputs: []Fused_In,
	insns: []Fused_Insn,
	stores: []Fused_Store,
	n_in, n, inner: int,
) {
	slots: [MAX_FUSED_SLOTS]simd.f32x4
	outer := n / inner
	for row in 0 ..< outer {
		base := row * inner
		j := 0
		for ; j + 4 <= inner; j += 4 {
			for k in 0 ..< len(inputs) {
				inp := inputs[k]
				switch inp.kind {
				case .Const:
					slots[k] = inp.const_val
				case .Buffer:
					switch inp.mode {
					case .Direct:
						slots[k] = intrinsics.unaligned_load((^simd.f32x4)(&inp.data[base + j]))
					case .Row:
						slots[k] = intrinsics.unaligned_load((^simd.f32x4)(&inp.data[j]))
					case .Scalar:
						slots[k] = inp.data[0]
					case .Col, .Generic:
						slots[k] = fused_load(inp, base + j) // splat scalar fallback
					}
				}
			}
			for insn, t in insns {
				slots[n_in + t] = fused_eval_simd(insn.op, slots[insn.a], slots[insn.b])
			}
			for s in stores {
				intrinsics.unaligned_store((^simd.f32x4)(&s.data[base + j]), slots[s.slot])
			}
		}
		ss: [MAX_FUSED_SLOTS]f32
		for ; j < inner; j += 1 {
			ii := base + j
			for k in 0 ..< len(inputs) do ss[k] = fused_load(inputs[k], ii)
			for insn, t in insns {
				ss[n_in + t] = fused_eval(insn.op, ss[insn.a], ss[insn.b])
			}
			for s in stores do s.data[ii] = ss[s.slot]
		}
	}
}

run_fused_col_simd :: proc(
	inputs: []Fused_In,
	insns: []Fused_Insn,
	stores: []Fused_Store,
	n_in, n, inner: int,
) {
	slots: [MAX_FUSED_SLOTS]simd.f32x4
	outer := n / inner
	for row in 0 ..< outer {
		base := row * inner
		j := 0
		for ; j + 4 <= inner; j += 4 {
			for k in 0 ..< len(inputs) {
				inp := inputs[k]
				switch inp.kind {
				case .Const:
					slots[k] = inp.const_val
				case .Buffer:
					switch inp.mode {
					case .Direct:
						slots[k] = intrinsics.unaligned_load((^simd.f32x4)(&inp.data[base + j]))
					case .Col:
						slots[k] = inp.data[row]
					case .Scalar:
						slots[k] = inp.data[0]
					case .Row, .Generic:
						slots[k] = fused_load(inp, base + j)
					}
				}
			}
			for insn, t in insns {
				slots[n_in + t] = fused_eval_simd(insn.op, slots[insn.a], slots[insn.b])
			}
			for s in stores {
				intrinsics.unaligned_store((^simd.f32x4)(&s.data[base + j]), slots[s.slot])
			}
		}
		ss: [MAX_FUSED_SLOTS]f32
		for ; j < inner; j += 1 {
			ii := base + j
			for k in 0 ..< len(inputs) do ss[k] = fused_load(inputs[k], ii)
			for insn, t in insns {
				ss[n_in + t] = fused_eval(insn.op, ss[insn.a], ss[insn.b])
			}
			for s in stores do s.data[ii] = ss[s.slot]
		}
	}
}
