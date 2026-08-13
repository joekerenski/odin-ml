package ml

// ============================================================================
// Ops — lazy graph builders (no compute, no output buffer).
//
// Each op:
//   1. infers output shape (and device / requires_grad),
//   2. allocates a Tensor shell via new_tensor_lazy (data = nil),
//   3. attaches a Context (LazyOp: kind + parents + meta).
//
// Call realize(t) to topo-sort and run kernels (see realize.odin).
// ============================================================================


// ---- helpers --------------------------------------------------------------

// Check if two shapes are mutually broadcastable (NumPy rule):
// for each dimension (right-aligned), either they're equal or at least one is 1.
shapes_broadcast :: proc(a, b: []i32) -> bool {
	na := len(a)
	nb := len(b)
	n := na > nb ? na : nb
	for i in 0..<n {
		ai := na - 1 - i
		bi := nb - 1 - i
		av: i32 = 1
		bv: i32 = 1
		if ai >= 0 do av = a[ai]
		if bi >= 0 do bv = b[bi]
		if av != bv && av != 1 && bv != 1 do return false
	}
	return true
}

// Compute the broadcasted result shape into a stack buffer.
// Returns the number of dims (ndim). `out` must be at least MAX_DIMS wide.
broadcast_result :: proc(out: []i32, a, b: []i32) -> int {
	na := len(a)
	nb := len(b)
	n := na > nb ? na : nb
	for i in 0..<n {
		ai := na - 1 - i
		bi := nb - 1 - i
		av: i32 = 1
		bv: i32 = 1
		if ai >= 0 do av = a[ai]
		if bi >= 0 do bv = b[bi]
		out[n - 1 - i] = av > bv ? av : bv
	}
	return n
}

// Add `a` (a_shape) into `out` (out_shape) following NumPy broadcasting.
// Accumulates (+=) so callers can combine multiple operands.
broadcast_add_into :: proc(out: []f32, out_shape: []i32, a: []f32, a_shape: []i32) {
	odim := len(out_shape)
	adim := len(a_shape)
	total := numel(out_shape)
	idx_buf: [MAX_DIMS]i32
	for flat in 0..<total {
		unravel_index(i32(flat), out_shape, idx_buf[:])
		a_idx: i32 = 0
		for d := 0; d < adim; d += 1 {
			ad := odim - adim + d
			v := idx_buf[ad]
			if a_shape[d] == 1 do v = 0
			a_idx += v * stride_of(a_shape, d)
		}
		out[flat] += a[a_idx]
	}
}

// Is `shape` effectively a scalar? True for empty shape or all dims == 1.
is_scalar_shape :: proc(shape: []i32) -> bool {
	for s in shape do if s != 1 do return false
	return true
}

// True iff `shape_b` matches the inner part of `shape_a` (right-aligned, exact sizes).
// Returns the inner element count (i32).
inner_match_size :: proc(shape_a, shape_b: []i32) -> (matches: bool, inner_elems: i32) {
	nb := len(shape_b)
	if nb == 0 || nb > len(shape_a) do return false, 0
	inner_elems = 1
	for i in 0..<nb {
		od := len(shape_a) - nb + i
		if shape_b[i] != shape_a[od] do return false, 0
		inner_elems *= shape_b[i]
	}
	return true, inner_elems
}

// True iff `shape_b` matches `shape_a` but with innermost dim = 1.
// I.e. b = [outer..., 1] and a = [outer..., N] for some N > 1.
// Returns the outer size and the inner N size for the SIMD loop.
// This is the "column vector broadcast" pattern: b's inner dim is 1 and gets
// stretched to a's inner dim.
col_broadcast_match :: proc(shape_a, shape_b: []i32) -> (matches: bool, outer: i32, inner_n: i32) {
	na := len(shape_a)
	nb := len(shape_b)
	if na == 0 || nb != na do return false, 0, 0
	if shape_b[nb - 1] != 1 do return false, 0, 0
	if shape_a[na - 1] <= 1 do return false, 0, 0  // inner_n must be > 1 for SIMD to help
	// outer dims must match exactly
	for i in 0..<na - 1 {
		if shape_a[i] != shape_b[i] do return false, 0, 0
	}
	outer = 1
	for i in 0..<na - 1 do outer *= shape_a[i]
	inner_n = shape_a[na - 1]
	return true, outer, inner_n
}

// Fast-path patterns for binary elementwise ops (Add/Mul/…).
// `outer`/`inner` filled for Row/Col; for NCHW_Bias_* the large tensor is
// always the first of (big, bias) when calling bias_add_nchw.
Bin_Pattern :: enum {
	Same,        // identical shapes, contiguous
	Scalar_B,    // b is scalar, a contiguous
	Scalar_A,    // a is scalar, b contiguous
	Row_B,       // b is inner vector repeating over a's outer rows
	Col_B,       // b is [outer...,1] stretched over a's last dim
	Col_A,       // a is [outer...,1] stretched over b's last dim
	NCHW_Bias_B, // a NCHW + b [1,C,1,1]
	NCHW_Bias_A, // b NCHW + a [1,C,1,1]
	Generic,
}

Bin_Class :: struct {
	kind:  Bin_Pattern,
	outer: i32,
	inner: i32,
}

// Classify a,b for a fast binary kernel. Prefers the cheapest match.
classify_binary :: proc(a, b: ^Tensor, out_shape: []i32) -> Bin_Class {
	if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
		return {kind = .Same}
	}
	if is_scalar_shape(b.shape[:]) && is_contiguous(a) {
		return {kind = .Scalar_B}
	}
	if is_scalar_shape(a.shape[:]) && is_contiguous(b) {
		return {kind = .Scalar_A}
	}
	// NCHW channel bias [1,C,1,1]
	if len(a.shape) == 4 && len(b.shape) == 4 &&
		is_contiguous(a) && is_contiguous(b) {
		if b.shape[0] == 1 && b.shape[2] == 1 && b.shape[3] == 1 &&
			a.shape[1] == b.shape[1] && shapes_equal(out_shape, a.shape[:]) {
			return {kind = .NCHW_Bias_B}
		}
		if a.shape[0] == 1 && a.shape[2] == 1 && a.shape[3] == 1 &&
			b.shape[1] == a.shape[1] && shapes_equal(out_shape, b.shape[:]) {
			return {kind = .NCHW_Bias_A}
		}
	}
	if is_contiguous(a) {
		if ok, inner := inner_match_size(a.shape[:], b.shape[:]); ok && numel(b.shape[:]) == inner && inner > 1 {
			return {kind = .Row_B, outer = numel(a.shape[:]) / inner, inner = inner}
		}
		if is_contiguous(b) {
			if ok, outer, inner := col_broadcast_match(a.shape[:], b.shape[:]); ok && inner > 1 {
				return {kind = .Col_B, outer = outer, inner = inner}
			}
			if ok, outer, inner := col_broadcast_match(b.shape[:], a.shape[:]); ok && inner > 1 {
				return {kind = .Col_A, outer = outer, inner = inner}
			}
		}
	}
	return {kind = .Generic}
}

// out = a * b elementwise-broadcast into out_shape.
mul_broadcast_into :: proc(out: []f32, out_shape: []i32, a: []f32, a_shape: []i32, b: []f32, b_shape: []i32) {
	odim := len(out_shape)
	adim := len(a_shape)
	bdim := len(b_shape)
	idx_buf: [MAX_DIMS]i32
	for f in 0..<numel(out_shape) {
		unravel_index(i32(f), out_shape, idx_buf[:])
		a_idx: i32 = 0
		for d := 0; d < adim; d += 1 {
			ad := odim - adim + d
			v := idx_buf[ad]
			if a_shape[d] == 1 do v = 0
			a_idx += v * stride_of(a_shape, d)
		}
		b_idx: i32 = 0
		for d := 0; d < bdim; d += 1 {
			bd := odim - bdim + d
			v := idx_buf[bd]
			if b_shape[d] == 1 do v = 0
			b_idx += v * stride_of(b_shape, d)
		}
		out[f] = a[a_idx] * b[b_idx]
	}
}

// out = a / b elementwise-broadcast into out_shape.
div_broadcast_into :: proc(out: []f32, out_shape: []i32, a: []f32, a_shape: []i32, b: []f32, b_shape: []i32) {
	odim := len(out_shape)
	adim := len(a_shape)
	bdim := len(b_shape)
	idx_buf: [MAX_DIMS]i32
	for f in 0..<numel(out_shape) {
		unravel_index(i32(f), out_shape, idx_buf[:])
		a_idx: i32 = 0
		for d := 0; d < adim; d += 1 {
			ad := odim - adim + d
			v := idx_buf[ad]
			if a_shape[d] == 1 do v = 0
			a_idx += v * stride_of(a_shape, d)
		}
		b_idx: i32 = 0
		for d := 0; d < bdim; d += 1 {
			bd := odim - bdim + d
			v := idx_buf[bd]
			if b_shape[d] == 1 do v = 0
			b_idx += v * stride_of(b_shape, d)
		}
		out[f] = a[a_idx] / b[b_idx]
	}
}

// ---- context attachment ---------------------------------------------------

make_ctx :: proc(out: ^Tensor, op: Op, parents: []^Tensor) {
	out.ctx = new(Context)
	out.ctx.op = op
	out.ctx.parents = make([dynamic]^Tensor, 0)
	for p in parents {
		append(&out.ctx.parents, p)
		if p.requires_grad do out.requires_grad = true
	}
}

// ---- elementwise binary ops (lazy) ----------------------------------------

add :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "add: shapes not broadcastable")
	assert(a.device == b.device, "add: tensors must be on same device")
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor_lazy(out_shape_buf[:ndim], false, a.device)
	make_ctx(out, .Add, {a, b})
	return out
}

sub :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "sub: shapes not broadcastable")
	assert(a.device == b.device, "sub: tensors must be on same device")
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor_lazy(out_shape_buf[:ndim], false, a.device)
	make_ctx(out, .Sub, {a, b})
	return out
}

mul :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "mul: shapes not broadcastable")
	assert(a.device == b.device, "mul: tensors must be on same device")
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor_lazy(out_shape_buf[:ndim], false, a.device)
	make_ctx(out, .Mul, {a, b})
	return out
}

div :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "div: shapes not broadcastable")
	assert(a.device == b.device, "div: tensors must be on same device")
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor_lazy(out_shape_buf[:ndim], false, a.device)
	make_ctx(out, .Div, {a, b})
	return out
}

neg :: proc(a: ^Tensor) -> ^Tensor {
	out := new_tensor_lazy(a.shape[:], false, a.device)
	make_ctx(out, .Neg, {a})
	return out
}

relu :: proc(a: ^Tensor) -> ^Tensor {
	out := new_tensor_lazy(a.shape[:], false, a.device)
	make_ctx(out, .ReLU, {a})
	return out
}

sigmoid :: proc(a: ^Tensor) -> ^Tensor {
	out := new_tensor_lazy(a.shape[:], false, a.device)
	make_ctx(out, .Sigmoid, {a})
	return out
}

sum :: proc(a: ^Tensor, axis: i32) -> ^Tensor {
	if axis == -1 {
		out := new_tensor_lazy({1}, false, a.device)
		out.ctx = new(Context)
		out.ctx.op = .Sum
		out.ctx.axis = -1
		out.ctx.parents = make([dynamic]^Tensor, 0)
		append(&out.ctx.parents, a)
		if a.requires_grad do out.requires_grad = true
		return out
	}
	assert(axis >= 0 && axis < i32(len(a.shape)), "sum: axis out of range")
	out_shape_buf: [MAX_DIMS]i32
	for i in 0..<len(a.shape) do out_shape_buf[i] = a.shape[i]
	out_shape_buf[axis] = 1
	out := new_tensor_lazy(out_shape_buf[:len(a.shape)], false, a.device)
	out.ctx = new(Context)
	out.ctx.op = .Sum
	out.ctx.axis = axis
	out.ctx.parents = make([dynamic]^Tensor, 0)
	append(&out.ctx.parents, a)
	if a.requires_grad do out.requires_grad = true
	return out
}

// mean = sum * (1/n). scale is a leaf with data; sum/mul are lazy.
mean :: proc(a: ^Tensor) -> ^Tensor {
	s := sum(a, -1)
	n := f32(numel(a.shape[:]))
	scale := new_tensor({1}, false, a.device)
	scale.data[0] = 1.0 / n
	return mul(s, scale)
}

reshape :: proc(a: ^Tensor, new_shape: []i32) -> ^Tensor {
	assert(numel(a.shape[:]) == numel(new_shape), "reshape: element count mismatch")
	out := new_tensor_lazy(new_shape, false, a.device)
	out.ctx = new(Context)
	out.ctx.op = .Reshape
	out.ctx.parents = make([dynamic]^Tensor, 0)
	append(&out.ctx.parents, a)
	if a.requires_grad do out.requires_grad = true
	return out
}

transpose :: proc(a: ^Tensor, axis0, axis1: i32) -> ^Tensor {
	assert(axis0 >= 0 && axis0 < i32(len(a.shape)), "transpose: axis0 out of range")
	assert(axis1 >= 0 && axis1 < i32(len(a.shape)), "transpose: axis1 out of range")
	out_shape_buf: [MAX_DIMS]i32
	for i in 0..<len(a.shape) do out_shape_buf[i] = a.shape[i]
	out_shape_buf[axis0], out_shape_buf[axis1] = a.shape[axis1], a.shape[axis0]
	out := new_tensor_lazy(out_shape_buf[:len(a.shape)], false, a.device)
	out.ctx = new(Context)
	out.ctx.op = .Transpose
	out.ctx.axis = axis0
	out.ctx.axis1 = axis1
	out.ctx.parents = make([dynamic]^Tensor, 0)
	append(&out.ctx.parents, a)
	if a.requires_grad do out.requires_grad = true
	return out
}

T :: proc(a: ^Tensor) -> ^Tensor {
	assert(len(a.shape) == 2, "T: only 2D supported")
	return transpose(a, 0, 1)
}

matmul :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(len(a.shape) == 2 && len(b.shape) == 2, "matmul: needs 2D inputs")
	assert(a.shape[1] == b.shape[0], "matmul: inner dims must match")
	assert(a.device == b.device, "matmul: devices must match")
	M, N := a.shape[0], b.shape[1]
	out := new_tensor_lazy({M, N}, false, a.device)
	make_ctx(out, .MatMul, {a, b})
	return out
}

// ---- spatial (NCHW) -------------------------------------------------------

// x:[N,Ci,H,W]  w:[Co,Ci,kH,kW]  →  [N,Co,Ho,Wo]
// stride/pad are scalars (same H/W) for the common case.
conv2d :: proc(x, w: ^Tensor, stride: i32 = 1, padding: i32 = 0) -> ^Tensor {
	assert(len(x.shape) == 4, "conv2d: x must be NCHW")
	assert(len(w.shape) == 4, "conv2d: w must be [Co,Ci,kH,kW]")
	assert(x.shape[1] == w.shape[1], "conv2d: Ci mismatch")
	assert(x.device == w.device, "conv2d: device mismatch")
	N, H, W := x.shape[0], x.shape[2], x.shape[3]
	Co, kH, kW := w.shape[0], w.shape[2], w.shape[3]
	Ho := out_spatial(H, kH, stride, padding)
	Wo := out_spatial(W, kW, stride, padding)
	assert(Ho > 0 && Wo > 0, "conv2d: empty output spatial size")
	out := new_tensor_lazy({N, Co, Ho, Wo}, false, x.device)
	out.ctx = new(Context)
	out.ctx.op = .Conv2d
	out.ctx.parents = make([dynamic]^Tensor, 0)
	append(&out.ctx.parents, x)
	append(&out.ctx.parents, w)
	out.ctx.kH, out.ctx.kW = kH, kW
	out.ctx.sH, out.ctx.sW = stride, stride
	out.ctx.pH, out.ctx.pW = padding, padding
	if x.requires_grad || w.requires_grad do out.requires_grad = true
	return out
}

// x:[N,C,H,W] → [N,C,Ho,Wo]
max_pool2d :: proc(x: ^Tensor, kernel_size: i32 = 2, stride: i32 = 0, padding: i32 = 0) -> ^Tensor {
	assert(len(x.shape) == 4, "max_pool2d: x must be NCHW")
	s := stride
	if s == 0 do s = kernel_size
	N, C, H, W := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	Ho := out_spatial(H, kernel_size, s, padding)
	Wo := out_spatial(W, kernel_size, s, padding)
	assert(Ho > 0 && Wo > 0, "max_pool2d: empty output")
	out := new_tensor_lazy({N, C, Ho, Wo}, false, x.device)
	out.ctx = new(Context)
	out.ctx.op = .MaxPool2d
	out.ctx.parents = make([dynamic]^Tensor, 0)
	append(&out.ctx.parents, x)
	out.ctx.kH, out.ctx.kW = kernel_size, kernel_size
	out.ctx.sH, out.ctx.sW = s, s
	out.ctx.pH, out.ctx.pW = padding, padding
	if x.requires_grad do out.requires_grad = true
	return out
}

// Flatten all dims after batch: [N, ...] → [N, product(...)]
flatten :: proc(x: ^Tensor) -> ^Tensor {
	assert(len(x.shape) >= 2, "flatten: need at least 2 dims")
	n := x.shape[0]
	rest: i32 = 1
	for i in 1..<len(x.shape) do rest *= x.shape[i]
	return reshape(x, {n, rest})
}

// Softmax filled at realize into ctx.cache; labels live on ctx (not a fake parent).
cross_entropy :: proc(logits: ^Tensor, labels: []u8) -> ^Tensor {
	B := logits.shape[0]
	C := logits.shape[1]
	assert(int(len(labels)) >= int(B), "cross_entropy: not enough labels for batch")
	for b in 0..<B {
		assert(i32(labels[b]) < C, "cross_entropy: label out of range")
	}

	// Copy labels so they survive if caller reuses the slice (and live on arena).
	labs := make([]u8, B)
	copy(labs, labels[:B])

	loss := new_tensor_lazy({1}, false, logits.device)
	loss.ctx = new(Context)
	loss.ctx.op = .CrossEntropy
	loss.ctx.parents = make([dynamic]^Tensor, 0)
	append(&loss.ctx.parents, logits)
	loss.ctx.labels = labs
	loss.ctx.cache = new_tensor_lazy({B, C}, false, logits.device)
	if logits.requires_grad do loss.requires_grad = true
	return loss
}
