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
	insns: [MAX_FUSED_INSNS]Fused_Insn
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
	for s, j in k.stores[:k.n_stores] do outs[j] = Fused_Store{slot[s.node], k.bufs[s.buf]}
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
		if v.strides[0] == R * I && v.strides[1] == I && (v.strides[2] == 1 || I == 1) {
			out, src := k.bufs[k.stores[0].buf], k.bufs[v.buf][v.offset:]
			if k.nodes[red].op == .Sum do reduce_block(out, src, O, R, I, .Sum)
			else do reduce_block(out, src, O, R, I, .ReduceMax)
			return
		}
	}
	cpu_reduce_generic(k, red)
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
