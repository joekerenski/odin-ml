package ml

// ============================================================================
// Autograd — the backward pass.
//
// backward(t):
//   1. seed t.grad with ones (treat t as the loss),
//   2. DFS from t in post-order to get a topological list of all tensors that
//      feed into t,
//   3. walk that list in reverse (outputs before inputs), and for every non-leaf
//      node call backward_op, which reads node.grad and pushes contributions
//      into each parent's .grad.
//
// Each op's backward is the chain rule: dL/d(parent) = dL/d(out) * d(out)/d(parent).
// For broadcasting, a parent that was broadcast up in the forward must have its
// gradient summed back down to its real shape (sum_to_shape).
//
// All scratch allocations go through context.allocator — in the training loop
// this is the same arena as the forward pass, so everything is reclaimed with
// one free_all. No manual freeing needed.
// ============================================================================

import "core:fmt"
import "core:math"
import "core:time"

// ---- topo sort (post-order DFS) -------------------------------------------

topo_sort :: proc(t: ^Tensor, topo: ^[dynamic]^Tensor, visited: ^map[^Tensor]bool) {
	if t in visited^ do return
	visited^[t] = true
	if t.ctx != nil {
		for p in t.ctx.parents do topo_sort(p, topo, visited)
	}
	append(topo, t)
}

// ---- reduce a gradient down to a smaller shape (broadcast reverse) --------

// grad has grad_shape; sum it down to target_shape. Any dim where target is 1
// (but grad is >1) gets summed out, plus any leading dims in grad that target
// doesn't have get summed out.
sum_to_shape :: proc(grad: []f32, grad_shape, target_shape: []i32) -> []f32 {
	out := make([]f32, numel(target_shape))
	for i in 0..<len(out) do out[i] = 0

	// Fast path: NCHW grad → [1,C,1,1] or [C] channel bias reduce
	if len(grad_shape) == 4 {
		N, C, H, W := grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]
		HW := H * W
		if len(target_shape) == 4 &&
			target_shape[0] == 1 && target_shape[1] == C &&
			target_shape[2] == 1 && target_shape[3] == 1 {
			for n in 0..<N {
				for c in 0..<C {
					base := (n * C + c) * HW
					s: f32 = 0
					for i in 0..<HW do s += grad[base + i]
					out[c] += s
				}
			}
			return out
		}
		if len(target_shape) == 1 && target_shape[0] == C {
			for n in 0..<N {
				for c in 0..<C {
					base := (n * C + c) * HW
					s: f32 = 0
					for i in 0..<HW do s += grad[base + i]
					out[c] += s
				}
			}
			return out
		}
	}

	gg := len(grad_shape)
	tg := len(target_shape)
	idx_buf: [MAX_DIMS]i32
	for flat in 0..<len(grad) {
		unravel_index(i32(flat), grad_shape, idx_buf[:])
		t_flat: i32 = 0
		for d := 0; d < tg; d += 1 {
			gd := gg - tg + d
			v := idx_buf[gd]
			if target_shape[d] == 1 && grad_shape[gd] > 1 do v = 0
			t_flat += v * stride_of(target_shape, d)
		}
		out[t_flat] += grad[flat]
	}
	return out
}

// ---- accumulate a gradient contribution into a parent ----------------------

accum_grad :: proc(parent: ^Tensor, contribution: []f32, contrib_shape: []i32) {
	if !parent.requires_grad do return
	if parent.grad == nil {
		parent.grad = new_tensor(parent.shape[:], false)
	}
	// Same shape: just += (no alloc / reduce)
	if shapes_equal(contrib_shape, parent.shape[:]) {
		for i in 0..<len(parent.grad.data) do parent.grad.data[i] += contribution[i]
		return
	}
	reduced := sum_to_shape(contribution, contrib_shape, parent.shape[:])
	for i in 0..<len(parent.grad.data) do parent.grad.data[i] += reduced[i]
}

// ---- raw helpers (no graph nodes, just buffers) ---------------------------

matmul_raw :: proc(a: []f32, M, K: i32, b: []f32, K2, N: i32) -> []f32 {
	assert(K == K2, "matmul_raw: inner dim mismatch")
	out := make([]f32, M * N)
	matmul_f32(out, a, b, M, K, N)
	return out
}

transpose_raw :: proc(data: []f32, shape: []i32, axis0, axis1: int, out_shape: []i32) -> []f32 {
	out := make([]f32, numel(shape))
	idx_buf: [MAX_DIMS]i32
	for flat in 0..<len(data) {
		unravel_index(i32(flat), shape, idx_buf[:])
		idx_buf[axis0], idx_buf[axis1] = idx_buf[axis1], idx_buf[axis0]
		o_flat: i32 = 0
		for d in 0..<len(out_shape) do o_flat += idx_buf[d] * stride_of(out_shape, d)
		out[o_flat] = data[flat]
	}
	return out
}

// ---- entry point -----------------------------------------------------------

backward :: proc(t: ^Tensor) {
	assert(t.requires_grad, "backward: tensor does not require grad")
	realize(t) // forward must complete before grads
	t.grad = ones(t.shape[:], false)

	topo: [dynamic]^Tensor = make([dynamic]^Tensor, 0)
	visited: map[^Tensor]bool
	defer delete(topo)
	defer delete(visited)
	topo_sort(t, &topo, &visited)

	if debug_level >= 3 {
		fmt.println("  [backward] reverse topo")
	}

	bwd_start: time.Tick
	if debug_level >= 1 do bwd_start = time.tick_now()

	bk := 0
	for i := len(topo) - 1; i >= 0; i -= 1 {
		node := topo[i]
		// Skip leaves and nodes that never needed grad (e.g. reshape of input).
		if node.ctx == nil || node.grad == nil do continue
		t0: time.Tick
		if debug_level >= 2 do t0 = time.tick_now()
		backward_op(node)
		if debug_level >= 2 {
			dt_ns := i64(time.tick_since(t0))
			counters.time_ns += dt_ns
			fmt.printfln(
				"  bwd %3d %-14s shape=%v  %7.3f ms",
				bk, op_name(node.ctx.op), node.shape, f64(dt_ns) / 1e6,
			)
		}
		counters.bwd_ops += 1
		bk += 1
	}

	if debug_level == 1 {
		ms := f64(time.tick_since(bwd_start)) / 1e6
		fmt.printfln("  backward: %d ops  %.3f ms", bk, ms)
	}
}

// ---- per-op backward dispatch ---------------------------------------------

backward_op :: proc(out: ^Tensor) {
	ctx := out.ctx
	p := ctx.parents
	g := out.grad.data

	switch ctx.op {
	case .Add:
		accum_grad(p[0], g, out.shape[:])
		accum_grad(p[1], g, out.shape[:])

	case .Sub:
		accum_grad(p[0], g, out.shape[:])
		neg_g := make([]f32, len(g))
		for i in 0..<len(g) do neg_g[i] = -g[i]
		accum_grad(p[1], neg_g, out.shape[:])

	case .Mul:
		a, b := p[0], p[1]
		ad, bd := contig_data(a), contig_data(b)
		ga := make([]f32, len(g))
		gb := make([]f32, len(g))
		mul_broadcast_into(ga, out.shape[:], g, out.shape[:], bd, b.shape[:])
		mul_broadcast_into(gb, out.shape[:], g, out.shape[:], ad, a.shape[:])
		accum_grad(a, ga, out.shape[:])
		accum_grad(b, gb, out.shape[:])

	case .Div:
		// d/da (a/b) = 1/b ; d/db (a/b) = -a/b²  (with broadcast)
		a, b := p[0], p[1]
		ad, bd := contig_data(a), contig_data(b)
		ga := make([]f32, len(g))
		gb := make([]f32, len(g))
		div_broadcast_into(ga, out.shape[:], g, out.shape[:], bd, b.shape[:])
		ab := make([]f32, len(g))
		mul_broadcast_into(ab, out.shape[:], ad, a.shape[:], g, out.shape[:])
		b2 := make([]f32, len(g))
		mul_broadcast_into(b2, out.shape[:], bd, b.shape[:], bd, b.shape[:])
		div_broadcast_into(gb, out.shape[:], ab, out.shape[:], b2, out.shape[:])
		for i in 0..<len(gb) do gb[i] = -gb[i]
		accum_grad(a, ga, out.shape[:])
		accum_grad(b, gb, out.shape[:])

	case .Neg:
		neg_g := make([]f32, len(g))
		for i in 0..<len(g) do neg_g[i] = -g[i]
		accum_grad(p[0], neg_g, out.shape[:])

	case .MatMul:
		a, b := p[0], p[1]
		ad, bd := contig_data(a), contig_data(b)
		M, K, N := out.shape[0], a.shape[1], out.shape[1]
		bT := transpose_raw(bd, b.shape[:], 0, 1, {b.shape[1], b.shape[0]})
		da := matmul_raw(g, M, N, bT, N, K)
		aT := transpose_raw(ad, a.shape[:], 0, 1, {a.shape[1], a.shape[0]})
		db := matmul_raw(aT, K, M, g, M, N)
		accum_grad(a, da, a.shape[:])
		accum_grad(b, db, b.shape[:])

	case .Sum:
		a := p[0]
		an := int(numel(a.shape[:]))
		if ctx.axis == -1 {
			ga := make([]f32, an)
			for i in 0..<an do ga[i] = g[0]
			accum_grad(a, ga, a.shape[:])
		} else {
			k := int(ctx.axis)
			ga := make([]f32, an)
			idx_buf: [MAX_DIMS]i32
			for flat in 0..<an {
				unravel_index(i32(flat), a.shape[:], idx_buf[:])
				g_idx: i32 = 0
				for d in 0..<len(out.shape) {
					v := idx_buf[d]
					if d == k do v = 0
					g_idx += v * stride_of(out.shape[:], d)
				}
				ga[flat] = g[g_idx]
			}
			accum_grad(a, ga, a.shape[:])
		}

	case .Reshape:
		// Grad is dense in out shape; parent just needs same bytes, parent shape.
		ga := make([]f32, len(g))
		copy(ga, g)
		accum_grad(p[0], ga, p[0].shape[:])

	case .Transpose:
		// If out is a view, out.grad is still dense in out.shape order.
		// Transpose grad back to parent shape (dense).
		axis0 := int(ctx.axis)
		axis1 := int(ctx.axis1)
		ga := transpose_raw(g, out.shape[:], axis0, axis1, p[0].shape[:])
		accum_grad(p[0], ga, p[0].shape[:])

	case .ReLU:
		a := p[0]
		ad := contig_data(a)
		ga := make([]f32, len(g))
		for i in 0..<len(g) do ga[i] = ad[i] > 0 ? g[i] : 0.0
		accum_grad(a, ga, a.shape[:])

	case .Sigmoid:
		// out may be contig (sigmoid always writes dense out)
		od := contig_data(out)
		ga := make([]f32, len(g))
		for i in 0..<len(g) do ga[i] = g[i] * od[i] * (1.0 - od[i])
		accum_grad(p[0], ga, out.shape[:])

	case .CrossEntropy:
		a := p[0]
		softmax := ctx.cache
		B := a.shape[0]
		C := a.shape[1]
		ga := make([]f32, int(B * C))
		scale := g[0] / f32(B)
		for i in 0..<len(ga) do ga[i] = softmax.data[i] * scale
		for b in 0..<B {
			ga[b * C + i32(ctx.labels[b])] -= scale
		}
		accum_grad(a, ga, a.shape[:])

	case .Conv2d:
		x, w := p[0], p[1]
		xd, wd := contig_data(x), contig_data(w)
		N, Ci, H, Ww := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		Co := w.shape[0]
		if x.requires_grad {
			dx := make([]f32, int(numel(x.shape[:])))
			conv2d_backward_input(
				dx, g, wd,
				N, Ci, H, Ww, Co, ctx.kH, ctx.kW, ctx.sH, ctx.sW, ctx.pH, ctx.pW,
			)
			accum_grad(x, dx, x.shape[:])
		}
		if w.requires_grad {
			dw := make([]f32, int(numel(w.shape[:])))
			conv2d_backward_weight(
				dw, g, xd,
				N, Ci, H, Ww, Co, ctx.kH, ctx.kW, ctx.sH, ctx.sW, ctx.pH, ctx.pW,
			)
			accum_grad(w, dw, w.shape[:])
		}

	case .MaxPool2d:
		x := p[0]
		if x.requires_grad {
			dx := make([]f32, int(numel(x.shape[:])))
			maxpool2d_backward(dx, g, ctx.indices)
			accum_grad(x, dx, x.shape[:])
		}
	}
}