package ml

// ============================================================================
// Ops — forward implementations.
//
// Each op:
//   1. computes a new tensor's data from its parents,
//   2. attaches a Context that records the op kind and the parents,
//   3. sets requires_grad on the output iff any parent needs grad.
//
// Broadcasting: supported for elementwise ops (add/sub/mul/div) in the simple
// right-aligned NumPy sense: a dimension of size 1 broadcasts up to match.
//
// All allocations go through context.allocator. In the training loop this is
// an arena, so the entire forward graph is reclaimed with one free_all call.
// Hot loops use stack buffers ([MAX_DIMS]i32) — zero per-iteration allocation.
// ============================================================================

import "core:math"

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
		r := i32(flat)
		for d := odim - 1; d >= 0; d -= 1 {
			idx_buf[d] = i32(r / stride_of(out_shape, d))
			r = r % stride_of(out_shape, d)
		}
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

// out = a * b elementwise-broadcast into out_shape.
mul_broadcast_into :: proc(out: []f32, out_shape: []i32, a: []f32, a_shape: []i32, b: []f32, b_shape: []i32) {
	odim := len(out_shape)
	adim := len(a_shape)
	bdim := len(b_shape)
	idx_buf: [MAX_DIMS]i32
	for f in 0..<numel(out_shape) {
		r := f
		for d := odim - 1; d >= 0; d -= 1 {
			idx_buf[d] = i32(r / stride_of(out_shape, d))
			r = r % stride_of(out_shape, d)
		}
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
		r := f
		for d := odim - 1; d >= 0; d -= 1 {
			idx_buf[d] = i32(r / stride_of(out_shape, d))
			r = r % stride_of(out_shape, d)
		}
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

// ---- elementwise binary ops (with broadcasting) ---------------------------

add :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "add: shapes not broadcastable")
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor(out_shape_buf[:ndim])
	broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
	broadcast_add_into(out.data, out.shape[:], b.data, b.shape[:])
	make_ctx(out, .Add, {a, b})
	return out
}

sub :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "sub: shapes not broadcastable")
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor(out_shape_buf[:ndim])
	broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
	neg_b := make([]f32, len(b.data))
for i in 0..<len(b.data) do neg_b[i] = -b.data[i]
	broadcast_add_into(out.data, out.shape[:], neg_b, b.shape[:])
	make_ctx(out, .Sub, {a, b})
	return out
}

mul :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "mul: shapes not broadcastable")
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor(out_shape_buf[:ndim])
	mul_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
	make_ctx(out, .Mul, {a, b})
	return out
}

div :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "div: shapes not broadcastable")
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor(out_shape_buf[:ndim])
	div_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
	make_ctx(out, .Div, {a, b})
	return out
}

neg :: proc(a: ^Tensor) -> ^Tensor {
	out := new_tensor(a.shape[:])
	for i in 0..<len(out.data) do out.data[i] = -a.data[i]
	make_ctx(out, .Neg, {a})
	return out
}

// ---- elementwise unary activations ----------------------------------------

relu :: proc(a: ^Tensor) -> ^Tensor {
	out := new_tensor(a.shape[:])
	for i in 0..<len(out.data) {
		out.data[i] = a.data[i] > 0 ? a.data[i] : 0.0
	}
	make_ctx(out, .ReLU, {a})
	return out
}

sigmoid :: proc(a: ^Tensor) -> ^Tensor {
	out := new_tensor(a.shape[:])
	for i in 0..<len(out.data) {
		out.data[i] = 1.0 / (1.0 + math.exp(-a.data[i]))
	}
	make_ctx(out, .Sigmoid, {a})
	return out
}

// ---- reductions -----------------------------------------------------------

sum :: proc(a: ^Tensor, axis: i32) -> ^Tensor {
	if axis == -1 {
		out := new_tensor({1})
		s: f32 = 0
		for v in a.data do s += v
		out.data[0] = s
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
	out := new_tensor(out_shape_buf[:len(a.shape)])
	for i in 0..<len(out.data) do out.data[i] = 0
	idx_buf: [MAX_DIMS]i32
	for flat in 0..<len(a.data) {
		r := i32(flat)
		for d := len(a.shape) - 1; d >= 0; d -= 1 {
			idx_buf[d] = i32(r / stride_of(a.shape[:], d))
			r = r % stride_of(a.shape[:], d)
		}
		out_idx: i32 = 0
		for d in 0..<len(a.shape) do out_idx += (idx_buf[d] == axis ? 0 : idx_buf[d]) * stride_of(out.shape[:], d)
		out.data[out_idx] += a.data[flat]
	}
	out.ctx = new(Context)
	out.ctx.op = .Sum
	out.ctx.axis = axis
	out.ctx.parents = make([dynamic]^Tensor, 0)
	append(&out.ctx.parents, a)
	if a.requires_grad do out.requires_grad = true
	return out
}

mean :: proc(a: ^Tensor) -> ^Tensor {
	s := sum(a, -1)
	n := f32(numel(a.shape[:]))
	scale := new_tensor({1})
	scale.data[0] = 1.0 / n
	return mul(s, scale)
}

// ---- shape ops ------------------------------------------------------------

reshape :: proc(a: ^Tensor, new_shape: []i32) -> ^Tensor {
	assert(numel(a.shape[:]) == numel(new_shape), "reshape: element count mismatch")
	out := new_tensor(new_shape)
	for i in 0..<len(out.data) do out.data[i] = a.data[i]
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
	out := new_tensor(out_shape_buf[:len(a.shape)])
	idx_buf: [MAX_DIMS]i32
	for flat in 0..<len(a.data) {
		r := i32(flat)
		for d := len(a.shape) - 1; d >= 0; d -= 1 {
			idx_buf[d] = i32(r / stride_of(a.shape[:], d))
			r = r % stride_of(a.shape[:], d)
		}
		idx_buf[axis0], idx_buf[axis1] = idx_buf[axis1], idx_buf[axis0]
		o_flat: i32 = 0
		for d in 0..<len(out.shape) do o_flat += idx_buf[d] * stride_of(out.shape[:], d)
		out.data[o_flat] = a.data[flat]
	}
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

// ---- linear algebra -------------------------------------------------------

matmul :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(len(a.shape) == 2 && len(b.shape) == 2, "matmul: needs 2D inputs")
	assert(a.shape[1] == b.shape[0], "matmul: inner dims must match")
	M, K, N := a.shape[0], a.shape[1], b.shape[1]
	out := new_tensor({M, N})
	for i in 0..<M {
		for j in 0..<N {
			s: f32 = 0
			for k in 0..<K do s += a.data[i*K + k] * b.data[k*N + j]
			out.data[i*N + j] = s
		}
	}
	make_ctx(out, .MatMul, {a, b})
	return out
}