package ml

// ============================================================================
// CPU backend for the kernel IR (kernel_ir.odin). Known shapes go to the
// tuned kernels; anything else runs a plain (correct, unhurried) evaluator:
//
//   elementwise          → the chunked SIMD interpreter (fuse.odin)
//   single strided load  → strided_copy (permutes, copies)
//   Load → Reduce, dense → reduce_block (kernel_f32.odin)
//   other reductions     → scalar evaluator (prologue / reduce / epilogue)
// ============================================================================

import "base:runtime"
import "core:math"

cpu_kernel :: proc(k: ^Kernel) {
	if kernel_has_reduce(k) {
		cpu_reduce(k)
		return
	}
	if k.n_nodes == 1 && k.n_stores == 1 && k.loads[0].mode != .Direct {
		v := k.loads[0]
		strided_copy(k.bufs[k.stores[0].buf], k.bufs[v.buf][v.offset:], k.dims[:k.nd], v.strides[:k.nd])
		return
	}
	run_fused_kernel_ir(k)
}

// Elementwise: loads and constants become the interpreter's input slots, ALU
// nodes its instructions.
run_fused_kernel_ir :: proc(k: ^Kernel) {
	inputs: [MAX_FUSED_SLOTS]Fused_In
	insns: [2 * MAX_FUSED_INSNS]Fused_Insn
	outs: [MAX_FUSED_INSNS]Fused_Store
	slot: [MAX_KERNEL_NODES]int
	n_in, n_insn := 0, 0
	for n, i in k.nodes[:k.n_nodes] {
		#partial switch n.kind {
		case .Load:
			v := k.loads[n.load]
			mode: Load_Mode
			switch v.mode {
			case .Direct: mode = .Direct
			case .Scalar: mode = .Scalar
			case .Row: mode = .Row
			case .Col: mode = .Col
			case .Block: mode = .Block
			case .Generic: mode = .Generic
			}
			inputs[n_in] = Fused_In{mode = mode, data = k.bufs[v.buf][v.offset:], n = v.n, inner = v.inner, strides = v.strides}
			slot[i] = n_in
			n_in += 1
		case .Const:
			inputs[n_in] = Fused_In{mode = .Const, n = 1, value = n.value}
			slot[i] = n_in
			n_in += 1
		}
	}
	// nodes are in topological order: operands always have their slot already
	for n, i in k.nodes[:k.n_nodes] {
		if n.kind != .ALU do continue
		b := op_is_binary(n.op) ? slot[n.b] : 0
		insns[n_insn] = Fused_Insn{n.op, slot[n.a], b}
		slot[i] = n_in + n_insn
		n_insn += 1
	}
	stored: [MAX_FUSED_SLOTS]bool
	for s, j in k.stores[:k.n_stores] {
		sl := slot[s.node]
		// the interpreter stores each slot once, and only instruction results: a load
		// or constant, or a second store of a value, goes through an identity op
		if sl < n_in || stored[sl] {
			insns[n_insn] = Fused_Insn{.Expand, sl, 0}
			sl = n_in + n_insn
			n_insn += 1
		}
		stored[sl] = true
		outs[j] = Fused_Store{sl, k.bufs[s.buf]}
	}
	job := Fused_Job{inputs[:n_in], insns[:n_insn], outs[:k.n_stores], k.dims[:k.nd]}
	run_fused_kernel(&job)
}

@(private = "file")
cpu_reduce :: proc(k: ^Kernel) {
	red := kernel_reduce_node(k)
	// the shape every reduction of the scheduler has today: [outer, r, inner], one dense load
	if k.nd == 3 && k.red_lo == 1 && k.red_hi == 2 && k.n_nodes == 2 && red == 1 && k.n_stores == 1 && k.stores[0].node == 1 {
		v := k.loads[0]
		O, R, I := k.dims[0], k.dims[1], k.dims[2]
		if dense3(v, O, R, I) {
			out, src := k.bufs[k.stores[0].buf], k.bufs[v.buf][v.offset:]
			if k.nodes[red].op == .Sum do reduce_block(out, src, O, R, I, .Sum)
			else do reduce_block(out, src, O, R, I, .ReduceMax)
			return
		}
	}
	if k.nd == 3 && k.red_lo == 1 && k.red_hi == 2 {
		cpu_reduce_fused(k, red)
		return
	}
	cpu_reduce_generic(k, red)
}

// A fused reduction on the tuned paths: the prologue as an elementwise pass over
// [O, R, I] (into a dense temp, plus the stores it makes), reduce_block, then
// the epilogue as an elementwise pass over [O, 1, I]. Same passes as unfused.
@(private = "file")
cpu_reduce_fused :: proc(k: ^Kernel, red: int) {
	O, R, I := k.dims[0], k.dims[1], k.dims[2]
	op := k.nodes[red].op
	a := k.nodes[red].a

	// reduce input: a dense load as is, the operand's own store if it has one,
	// else a scratch temp written by the prologue pass
	src: []f32
	if v := k.loads[k.nodes[a].load]; k.nodes[a].kind == .Load && dense3(v, O, R, I) {
		src = k.bufs[v.buf][v.offset:]
	}
	pro_stores := false
	for st in k.stores[:k.n_stores] {
		if st.node >= red do continue
		pro_stores = true
		if st.node == a && src == nil do src = k.bufs[st.buf]
	}
	if src == nil || pro_stores {
		p := k^
		p.red_lo, p.red_hi = 0, 0
		p.n_nodes = red
		p.n_stores = 0
		for st in k.stores[:k.n_stores] do if st.node < red {
			p.stores[p.n_stores] = st
			p.n_stores += 1
		}
		if src == nil {
			src = cpu_temp(&cpu_temps[0], O * R * I)
			p.bufs[p.n_bufs] = src
			p.stores[p.n_stores] = K_Store{node = a, buf = p.n_bufs}
			p.n_bufs += 1
			p.n_stores += 1
		}
		kernel_finish(&p)
		run_fused_kernel_ir(&p)
	}

	// the reduction: straight into its store if it has one
	out: []f32
	for st in k.stores[:k.n_stores] do if st.node == red do out = k.bufs[st.buf]
	if out == nil do out = cpu_temp(&cpu_temps[1], O * I)
	if op == .Sum do reduce_block(out, src, O, R, I, .Sum)
	else do reduce_block(out, src, O, R, I, .ReduceMax)

	// epilogue: nodes after the reduce, the reduce read back as a load
	epi := false
	for st in k.stores[:k.n_stores] do if st.node > red do epi = true
	if !epi do return
	e := k^
	e.dims[1] = 1
	e.red_lo, e.red_hi = 0, 0
	e.bufs[e.n_bufs] = out
	e.loads[e.n_loads] = K_View{buf = e.n_bufs, strides = {I, 0, 1, 0, 0, 0, 0, 0}}
	e.nodes[red] = K_Node{kind = .Load, load = e.n_loads}
	e.n_bufs += 1
	e.n_loads += 1
	// drop the prologue: epilogue nodes only reference the reduce, epilogue loads and constants
	remap: [MAX_KERNEL_NODES]int
	n := 0
	for i in red ..< k.n_nodes {
		x := e.nodes[i]
		if x.kind == .ALU {
			x.a = remap[x.a]
			if op_is_binary(x.op) do x.b = remap[x.b]
		}
		e.nodes[n] = x
		remap[i] = n
		n += 1
	}
	e.n_nodes = n
	e.n_stores = 0
	for st in k.stores[:k.n_stores] do if st.node > red {
		e.stores[e.n_stores] = K_Store{node = remap[st.node], buf = st.buf}
		e.n_stores += 1
	}
	kernel_finish(&e)
	run_fused_kernel_ir(&e)
}

// One output point at a time: prologue per reduced point, then the epilogue.
@(private = "file")
cpu_reduce_generic :: proc(k: ^Kernel, red: int) {
	nd := k.nd
	vals: [MAX_KERNEL_NODES]f32
	idx: [MAX_DIMS]int
	n_out := kernel_out_points(k)
	n_red := 1
	for d in k.red_lo ..< k.red_hi do n_red *= k.dims[d]
	eval :: proc(k: ^Kernel, vals: ^[MAX_KERNEL_NODES]f32, idx: []int, from, to: int) {
		for i in from ..< to {
			n := k.nodes[i]
			switch n.kind {
			case .Load:
				v := k.loads[n.load]
				off := v.offset
				for d in 0 ..< k.nd do off += idx[d] * v.strides[d]
				vals[i] = k.bufs[v.buf][off]
			case .Const:
				vals[i] = n.value
			case .ALU:
				a, b := vals[n.a], vals[n.b]
				#partial switch n.op {
				case .Exp: vals[i] = math.exp(a)
				case .Log: vals[i] = math.ln(a)
				case: vals[i] = fused_eval(n.op, a, b)
				}
			case .Reduce:
			}
		}
	}
	for o in 0 ..< n_out {
		// output point o → coordinates of the non-reduced dims
		r := o
		for d := nd - 1; d >= 0; d -= 1 {
			if d >= k.red_lo && d < k.red_hi do continue
			idx[d] = r % k.dims[d]
			r /= k.dims[d]
		}
		acc := k.nodes[red].op == .Sum ? f32(0) : math.inf_f32(-1)
		for q in 0 ..< n_red {
			rr := q
			for d := k.red_hi - 1; d >= k.red_lo; d -= 1 {
				idx[d] = rr % k.dims[d]
				rr /= k.dims[d]
			}
			eval(k, &vals, idx[:nd], 0, red)
			x := vals[k.nodes[red].a]
			acc = k.nodes[red].op == .Sum ? acc + x : max(acc, x)
		}
		vals[red] = acc
		for d in k.red_lo ..< k.red_hi do idx[d] = 0
		eval(k, &vals, idx[:nd], red + 1, k.n_nodes)
		for s in k.stores[:k.n_stores] do k.bufs[s.buf][o] = vals[s.node]
	}
}

// A view that is the dense [O, R, I] buffer (strides of size-1 dims don't matter).
@(private = "file")
dense3 :: proc(v: K_View, O, R, I: int) -> bool {
	return (O == 1 || v.strides[0] == R * I) && (R == 1 || v.strides[1] == I) && (I == 1 || v.strides[2] == 1)
}

// Grow-only, non-zeroed scratch for fused reductions (CPU kernels run one at a
// time on the calling thread; parallel_for only splits inside a kernel).
@(private = "file")
cpu_temps: [2][]f32

@(private = "file")
cpu_temp :: proc(t: ^[]f32, n: int) -> []f32 {
	if len(t^) < n {
		if t^ != nil do delete(t^, scratch())
		bytes, err := runtime.mem_alloc_non_zeroed(n * size_of(f32), 64, scratch())
		assert(err == nil, "cpu_temp: out of memory")
		t^ = ([^]f32)(raw_data(bytes))[:n]
	}
	return t[:n]
}
