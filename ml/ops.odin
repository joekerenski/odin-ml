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

// one_hot leaf now; softmax filled at realize into ctx.cache
cross_entropy :: proc(logits: ^Tensor, labels: []u8) -> ^Tensor {
	B := logits.shape[0]
	C := logits.shape[1]
	assert(int(len(labels)) >= int(B), "cross_entropy: not enough labels for batch")

	one_hot := zeros({B, C}, false, logits.device)
	for b in 0..<B {
		label := i32(labels[b])
		assert(label < C, "cross_entropy: label out of range")
		one_hot.data[b*C + label] = 1.0
	}

	loss := new_tensor_lazy({1}, false, logits.device)
	loss.ctx = new(Context)
	loss.ctx.op = .CrossEntropy
	loss.ctx.parents = make([dynamic]^Tensor, 0)
	append(&loss.ctx.parents, logits)
	append(&loss.ctx.parents, one_hot)
	loss.ctx.cache = new_tensor_lazy({B, C}, false, logits.device)
	if logits.requires_grad do loss.requires_grad = true
	return loss
}
