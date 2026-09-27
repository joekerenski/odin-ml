package ml

// ============================================================================
// Ops — graph builders. No compute: each op infers its output shape and
// returns a new UOp node. realize() runs them (realize.odin).
//
// Primitives map 1:1 to an Op. Everything else is a composition of ops.
// ============================================================================

// ---- broadcasting ---------------------------------------------------------

// NumPy rule: right-aligned, each dim equal or one of them is 1.
shapes_broadcast :: proc(a, b: []i32) -> bool {
	n := max(len(a), len(b))
	for i in 0 ..< n {
		av := i < len(a) ? a[len(a) - 1 - i] : 1
		bv := i < len(b) ? b[len(b) - 1 - i] : 1
		if av != bv && av != 1 && bv != 1 do return false
	}
	return true
}

// Broadcast result shape into `out` (len >= MAX_DIMS). Returns ndim.
broadcast_result :: proc(out: []i32, a, b: []i32) -> int {
	n := max(len(a), len(b))
	for i in 0 ..< n {
		av := i < len(a) ? a[len(a) - 1 - i] : 1
		bv := i < len(b) ? b[len(b) - 1 - i] : 1
		out[n - 1 - i] = max(av, bv)
	}
	return n
}

is_scalar_shape :: proc(shape: []i32) -> bool {
	for s in shape do if s != 1 do return false
	return true
}

// ---- elementwise ----------------------------------------------------------

binary :: proc(op: Op, a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape, b.shape), "binary op: shapes not broadcastable")
	buf: [MAX_DIMS]i32
	n := broadcast_result(buf[:], a.shape, b.shape)
	return new_node(op, buf[:n], nil, a, b)
}

add :: proc(a, b: ^Tensor) -> ^Tensor { return binary(.Add, a, b) }
sub :: proc(a, b: ^Tensor) -> ^Tensor { return binary(.Sub, a, b) }
mul :: proc(a, b: ^Tensor) -> ^Tensor { return binary(.Mul, a, b) }
div :: proc(a, b: ^Tensor) -> ^Tensor { return binary(.Div, a, b) }
maximum :: proc(a, b: ^Tensor) -> ^Tensor { return binary(.Max, a, b) }
cmplt :: proc(a, b: ^Tensor) -> ^Tensor { return binary(.CmpLt, a, b) }

neg :: proc(a: ^Tensor) -> ^Tensor { return new_node(.Neg, a.shape, nil, a) }
exp :: proc(a: ^Tensor) -> ^Tensor { return new_node(.Exp, a.shape, nil, a) }
log :: proc(a: ^Tensor) -> ^Tensor { return new_node(.Log, a.shape, nil, a) }
sqrt :: proc(a: ^Tensor) -> ^Tensor { return new_node(.Sqrt, a.shape, nil, a) }

square :: proc(a: ^Tensor) -> ^Tensor { return mul(a, a) }

// Same values, but gradients stop here.
detach :: proc(a: ^Tensor) -> ^Tensor {
	u := new_node(.Reshape, a.shape, nil, a)
	u.requires_grad = false
	return u
}

relu :: proc(a: ^Tensor) -> ^Tensor {
	return maximum(a, scalar(0))
}

sigmoid :: proc(a: ^Tensor) -> ^Tensor {
	return div(scalar(1), add(scalar(1), exp(neg(a))))
}

// ---- reduce ---------------------------------------------------------------
// Axes count from the end when negative (-1 = last), as in numpy/tinygrad.
// Reductions keep reduced dims as size 1. The one-argument forms
// (sum(x), mean(x), max_all(x)) reduce everything to shape {1}.

norm_axis :: proc(axis: i32, ndim: int) -> i32 {
	a := axis < 0 ? axis + i32(ndim) : axis
	assert(a >= 0 && int(a) < ndim, "axis out of range")
	return a
}

reduce :: proc(op: Op, a: ^Tensor, axes: []i32) -> ^Tensor {
	shape, norm: [MAX_DIMS]i32
	copy(shape[:], a.shape)
	for ax, i in axes {
		norm[i] = norm_axis(ax, len(a.shape))
		shape[norm[i]] = 1
	}
	return new_node(op, shape[:len(a.shape)], copy_shape(norm[:len(axes)]), a)
}

reduce_all :: proc(op: Op, a: ^Tensor) -> ^Tensor {
	all: [MAX_DIMS]i32
	for i in 0 ..< len(a.shape) do all[i] = i32(i)
	return reshape(reduce(op, a, all[:len(a.shape)]), {1})
}

sum_axes :: proc(a: ^Tensor, axes: []i32) -> ^Tensor { return reduce(.Sum, a, axes) }
max_axes :: proc(a: ^Tensor, axes: []i32) -> ^Tensor { return reduce(.ReduceMax, a, axes) }

sum_all :: proc(a: ^Tensor) -> ^Tensor { return reduce_all(.Sum, a) }
sum_axis :: proc(a: ^Tensor, axis: i32) -> ^Tensor { return reduce(.Sum, a, {axis}) }
sum :: proc {
	sum_all,
	sum_axis,
}

max_all :: proc(a: ^Tensor) -> ^Tensor { return reduce_all(.ReduceMax, a) }
max_axis :: proc(a: ^Tensor, axis: i32) -> ^Tensor { return reduce(.ReduceMax, a, {axis}) }

mean_all :: proc(a: ^Tensor) -> ^Tensor {
	return mul(sum_all(a), scalar(1.0 / f32(numel(a.shape))))
}
mean_axis :: proc(a: ^Tensor, axis: i32) -> ^Tensor {
	return mul(sum_axis(a, axis), scalar(1.0 / f32(a.shape[norm_axis(axis, len(a.shape))])))
}
mean :: proc {
	mean_all,
	mean_axis,
}

// ---- softmax family (stable: shift by a detached max) ---------------------

log_softmax :: proc(x: ^Tensor, axis: i32) -> ^Tensor {
	z := sub(x, detach(max_axis(x, axis)))
	return sub(z, log(sum(exp(z), axis)))
}

softmax :: proc(x: ^Tensor, axis: i32) -> ^Tensor {
	e := exp(sub(x, detach(max_axis(x, axis))))
	return div(e, sum(e, axis))
}

// log Σ exp(x) over axis, keepdim.
logsumexp :: proc(x: ^Tensor, axis: i32) -> ^Tensor {
	m := detach(max_axis(x, axis))
	return add(log(sum(exp(sub(x, m)), axis)), m)
}

// Normalize over the last axis (no affine; see nn.LayerNorm).
layer_norm :: proc(x: ^Tensor, eps: f32 = 1e-5) -> ^Tensor {
	xc := sub(x, mean_axis(x, -1))
	return div(xc, sqrt(add(mean_axis(square(xc), -1), scalar(eps))))
}

// ---- movement -------------------------------------------------------------

reshape :: proc(a: ^Tensor, shape: []i32) -> ^Tensor {
	assert(numel(a.shape) == numel(shape), "reshape: element count mismatch")
	if shapes_equal(a.shape, shape) do return a
	return new_node(.Reshape, shape, nil, a)
}

// out.shape[i] = a.shape[order[i]]
permute :: proc(a: ^Tensor, order: []i32) -> ^Tensor {
	assert(len(order) == len(a.shape), "permute: order rank mismatch")
	buf: [MAX_DIMS]i32
	for o, i in order do buf[i] = a.shape[o]
	return new_node(.Permute, buf[:len(order)], copy_shape(order), a)
}

transpose :: proc(a: ^Tensor, axis0, axis1: i32) -> ^Tensor {
	n := i32(len(a.shape))
	ax0, ax1 := norm_axis(axis0, int(n)), norm_axis(axis1, int(n))
	order: [MAX_DIMS]i32
	for i in 0 ..< n do order[i] = i
	order[ax0], order[ax1] = ax1, ax0
	return permute(a, order[:n])
}

T :: proc(a: ^Tensor) -> ^Tensor {
	assert(len(a.shape) == 2, "T: only 2D supported")
	return transpose(a, 0, 1)
}

// Swap the last two axes (matrix transpose of every batch element).
mT :: proc(a: ^Tensor) -> ^Tensor {
	assert(len(a.shape) >= 2, "mT: need at least 2 dims")
	return transpose(a, -2, -1)
}

// Broadcast a to `shape` (NumPy rule).
expand :: proc(a: ^Tensor, shape: []i32) -> ^Tensor {
	if shapes_equal(a.shape, shape) do return a
	buf: [MAX_DIMS]i32
	n := broadcast_result(buf[:], a.shape, shape)
	assert(shapes_equal(buf[:n], shape), "expand: not broadcastable to shape")
	return new_node(.Expand, shape, nil, a)
}

// [N, ...] → [N, product(...)]
flatten :: proc(x: ^Tensor) -> ^Tensor {
	assert(len(x.shape) >= 2, "flatten: need at least 2 dims")
	return reshape(x, {x.shape[0], numel(x.shape[1:])})
}

// ---- primitives -----------------------------------------------------------

// [..., M, K] @ [..., K, N] → [..., M, N]. Batch dims broadcast.
// A 2D right operand (weights) folds the batch into M: one big GEMM.
matmul :: proc(a, b: ^Tensor) -> ^Tensor {
	na, nb := len(a.shape), len(b.shape)
	assert(na >= 2 && nb >= 2, "matmul: needs at least 2D inputs")
	M, K, N := a.shape[na - 2], a.shape[na - 1], b.shape[nb - 1]
	assert(K == b.shape[nb - 2], "matmul: inner dims must match")

	if nb == 2 && na > 2 {
		buf: [MAX_DIMS]i32
		copy(buf[:], a.shape)
		buf[na - 1] = N
		flat := matmul(reshape(a, {numel(a.shape[:na - 1]), K}), b)
		return reshape(flat, buf[:na])
	}

	batch: [MAX_DIMS]i32
	nbatch := broadcast_result(batch[:], a.shape[:na - 2], b.shape[:nb - 2])
	shape_a, shape_b, shape_o: [MAX_DIMS]i32
	copy(shape_a[:], batch[:nbatch]); shape_a[nbatch], shape_a[nbatch + 1] = M, K
	copy(shape_b[:], batch[:nbatch]); shape_b[nbatch], shape_b[nbatch + 1] = K, N
	copy(shape_o[:], batch[:nbatch]); shape_o[nbatch], shape_o[nbatch + 1] = M, N
	aa := expand(a, shape_a[:nbatch + 2])
	bb := expand(b, shape_b[:nbatch + 2])
	return new_node(.MatMul, shape_o[:nbatch + 2], nil, aa, bb)
}

out_spatial :: proc(in_size, k, stride, pad: i32) -> i32 {
	return (in_size + 2 * pad - k) / stride + 1
}

// x:[N,Ci,H,W]  w:[Co,Ci,kH,kW]  →  [N,Co,Ho,Wo]
conv2d :: proc(x, w: ^Tensor, stride: i32 = 1, padding: i32 = 0) -> ^Tensor {
	assert(len(x.shape) == 4, "conv2d: x must be NCHW")
	assert(len(w.shape) == 4, "conv2d: w must be [Co,Ci,kH,kW]")
	assert(x.shape[1] == w.shape[1], "conv2d: Ci mismatch")
	win := Window{w.shape[2], w.shape[3], stride, stride, padding, padding}
	Ho := out_spatial(x.shape[2], win.kH, stride, padding)
	Wo := out_spatial(x.shape[3], win.kW, stride, padding)
	assert(Ho > 0 && Wo > 0, "conv2d: empty output spatial size")
	return new_node(.Conv2d, {x.shape[0], w.shape[0], Ho, Wo}, win, x, w)
}

// x:[N,C,H,W] → [N,C,Ho,Wo]. stride 0 means stride = kernel_size.
max_pool2d :: proc(x: ^Tensor, kernel_size: i32 = 2, stride: i32 = 0, padding: i32 = 0) -> ^Tensor {
	assert(len(x.shape) == 4, "max_pool2d: x must be NCHW")
	s := stride == 0 ? kernel_size : stride
	win := Window{kernel_size, kernel_size, s, s, padding, padding}
	Ho := out_spatial(x.shape[2], kernel_size, s, padding)
	Wo := out_spatial(x.shape[3], kernel_size, s, padding)
	assert(Ho > 0 && Wo > 0, "max_pool2d: empty output")
	return new_node(.MaxPool2d, {x.shape[0], x.shape[1], Ho, Wo}, win, x)
}

// [B] class ids → [B, C] one-hot leaf.
one_hot :: proc(labels: []u8, C: i32) -> ^Tensor {
	t := new_tensor({i32(len(labels)), C})
	for l, b in labels {
		assert(i32(l) < C, "one_hot: label out of range")
		t.data[b * int(C) + int(l)] = 1
	}
	return t
}

// Mean softmax cross-entropy over the batch. logits:[B,C], labels: class ids.
cross_entropy :: proc(logits: ^Tensor, labels: []u8) -> ^Tensor {
	assert(len(logits.shape) == 2, "cross_entropy: logits must be [B,C]")
	B, C := logits.shape[0], logits.shape[1]
	assert(len(labels) >= int(B), "cross_entropy: not enough labels for batch")
	picked := mul(log_softmax(logits, 1), one_hot(labels[:B], C))
	return mul(sum(picked), scalar(-1.0 / f32(B)))
}
