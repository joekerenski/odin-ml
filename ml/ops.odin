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

relu :: proc(a: ^Tensor) -> ^Tensor {
	return maximum(a, scalar(0))
}

sigmoid :: proc(a: ^Tensor) -> ^Tensor {
	return div(scalar(1), add(scalar(1), exp(neg(a))))
}

// ---- reduce ---------------------------------------------------------------

// Sum over `axes`, keeping them as size 1.
sum_axes :: proc(a: ^Tensor, axes: []i32) -> ^Tensor {
	buf: [MAX_DIMS]i32
	copy(buf[:], a.shape)
	for ax in axes {
		assert(ax >= 0 && int(ax) < len(a.shape), "sum_axes: axis out of range")
		buf[ax] = 1
	}
	return new_node(.Sum, buf[:len(a.shape)], copy_shape(axes), a)
}

// axis = -1: sum everything → shape {1}. Otherwise keepdim over one axis.
sum :: proc(a: ^Tensor, axis: i32) -> ^Tensor {
	if axis == -1 {
		all: [MAX_DIMS]i32
		for i in 0 ..< len(a.shape) do all[i] = i32(i)
		return reshape(sum_axes(a, all[:len(a.shape)]), {1})
	}
	return sum_axes(a, {axis})
}

mean :: proc(a: ^Tensor) -> ^Tensor {
	return mul(sum(a, -1), scalar(1.0 / f32(numel(a.shape))))
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
	assert(axis0 >= 0 && axis0 < n && axis1 >= 0 && axis1 < n, "transpose: axis out of range")
	order: [MAX_DIMS]i32
	for i in 0 ..< n do order[i] = i
	order[axis0], order[axis1] = axis1, axis0
	return permute(a, order[:n])
}

T :: proc(a: ^Tensor) -> ^Tensor {
	assert(len(a.shape) == 2, "T: only 2D supported")
	return transpose(a, 0, 1)
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

matmul :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(len(a.shape) == 2 && len(b.shape) == 2, "matmul: needs 2D inputs")
	assert(a.shape[1] == b.shape[0], "matmul: inner dims must match")
	return new_node(.MatMul, {a.shape[0], b.shape[1]}, nil, a, b)
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

// Mean softmax cross-entropy over the batch. logits:[B,C], labels: class ids.
cross_entropy :: proc(logits: ^Tensor, labels: []u8) -> ^Tensor {
	assert(len(logits.shape) == 2, "cross_entropy: logits must be [B,C]")
	B, C := logits.shape[0], logits.shape[1]
	assert(len(labels) >= int(B), "cross_entropy: not enough labels for batch")
	labs := make([]u8, B) // own a copy: callers reuse their slices
	copy(labs, labels[:B])
	for l in labs do assert(i32(l) < C, "cross_entropy: label out of range")
	return new_node(.CrossEntropy, {1}, labs, logits)
}
