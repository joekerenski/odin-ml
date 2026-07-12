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

import "core:math"

// ---- reduce a gradient down to a smaller shape (broadcast reverse) --------

// grad has grad_shape; sum it down to target_shape. Any dim where target is 1
// (but grad is >1) gets summed out, plus any leading dims in grad that target
// doesn't have get summed out.
sum_to_shape :: proc(grad: []f32, grad_shape, target_shape: []i32) -> []f32 {
	out := make([]f32, numel(target_shape))
	for i in 0..<len(out) do out[i] = 0
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

// ---- topo sort (post-order DFS) -------------------------------------------

topo_sort :: proc(t: ^Tensor, topo, visited: ^[dynamic]^Tensor) {
	for v in visited^ do if v == t do return
	append(visited, t)
	if t.ctx != nil {
		for p in t.ctx.parents do topo_sort(p, topo, visited)
	}
	append(topo, t)
}

// ---- entry point -----------------------------------------------------------

backward :: proc(t: ^Tensor) {
	assert(t.requires_grad, "backward: tensor does not require grad")
	t.grad = ones(t.shape[:], false)

	topo: [dynamic]^Tensor = make([dynamic]^Tensor, 0)
	visited: [dynamic]^Tensor = make([dynamic]^Tensor, 0)
	defer delete(topo)
	defer delete(visited)
	topo_sort(t, &topo, &visited)

	for i := len(topo) - 1; i >= 0; i -= 1 {
		node := topo[i]
		if node.ctx != nil do backward_op(node)
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
		ga := make([]f32, len(g))
		gb := make([]f32, len(g))
		mul_broadcast_into(ga, out.shape[:], g, out.shape[:], b.data, b.shape[:])
		mul_broadcast_into(gb, out.shape[:], g, out.shape[:], a.data, a.shape[:])
		accum_grad(a, ga, out.shape[:])
		accum_grad(b, gb, out.shape[:])

	case .Div:
		a, b := p[0], p[1]
		ga := make([]f32, len(g))
		gb := make([]f32, len(g))
		for i in 0..<len(g) {
			bv := b.data[i]
			ga[i] = g[i] / bv
			gb[i] = -g[i] * a.data[i] / (bv * bv)
		}
		accum_grad(a, ga, out.shape[:])
		accum_grad(b, gb, out.shape[:])

	case .Neg:
		neg_g := make([]f32, len(g))
		for i in 0..<len(g) do neg_g[i] = -g[i]
		accum_grad(p[0], neg_g, out.shape[:])

	case .MatMul:
		a, b := p[0], p[1]
		M, K, N := out.shape[0], a.shape[1], out.shape[1]
		bT := transpose_raw(b.data, b.shape[:], 0, 1, {b.shape[1], b.shape[0]})
		da := matmul_raw(g, M, N, bT, N, K)
		aT := transpose_raw(a.data, a.shape[:], 0, 1, {a.shape[1], a.shape[0]})
		db := matmul_raw(aT, K, M, g, M, N)
		accum_grad(a, da, a.shape[:])
		accum_grad(b, db, b.shape[:])

	case .Sum:
		a := p[0]
		if ctx.axis == -1 {
			ga := make([]f32, len(a.data))
			for i in 0..<len(ga) do ga[i] = g[0]
			accum_grad(a, ga, a.shape[:])
		} else {
			k := int(ctx.axis)
			ga := make([]f32, len(a.data))
			idx_buf: [MAX_DIMS]i32
			for flat in 0..<len(a.data) {
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
		ga := make([]f32, len(g))
		for i in 0..<len(g) do ga[i] = g[i]
		accum_grad(p[0], ga, p[0].shape[:])

	case .Transpose:
		axis0 := int(ctx.axis)
		axis1 := int(ctx.axis1)
		ga := transpose_raw(g, out.shape[:], axis0, axis1, p[0].shape[:])
		accum_grad(p[0], ga, p[0].shape[:])

	case .ReLU:
		a := p[0]
		ga := make([]f32, len(g))
		for i in 0..<len(g) do ga[i] = a.data[i] > 0 ? g[i] : 0.0
		accum_grad(a, ga, a.shape[:])

	case .Sigmoid:
		ga := make([]f32, len(g))
		for i in 0..<len(g) do ga[i] = g[i] * out.data[i] * (1.0 - out.data[i])
		accum_grad(p[0], ga, out.shape[:])

	case .CrossEntropy:
		// Forward cached softmax (ctx.cache) and stored one-hot labels as p[1].
		// dL/d(logits) = (softmax - one_hot) / B  *  upstream_grad
		// upstream_grad for a scalar loss = 1.0 (seeded by backward()).
		a := p[0]        // logits [B, C]
		one_hot := p[1]  // one-hot labels [B, C] (requires_grad = false)
		softmax := ctx.cache
		B := a.shape[0]
		ga := make([]f32, len(a.data))
		scale := g[0] / f32(B)
		for i in 0..<len(ga) do ga[i] = (softmax.data[i] - one_hot.data[i]) * scale
		accum_grad(a, ga, a.shape[:])
	}
}