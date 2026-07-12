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

// ---- elementwise binary ops (with broadcasting) ---------------------------

// Contiguous same-shape → SIMD (CPU) or Metal compute shader (GPU).
// Otherwise general broadcast path (CPU only for now).
add :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "add: shapes not broadcastable")
	assert(a.device == b.device, "add: tensors must be on same device")

	if a.device == .Metal && shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
		out := new_tensor(a.shape[:], a.requires_grad || b.requires_grad, .Metal)
		metal_add(out.data, a.data, b.data)
		make_ctx(out, .Add, {a, b})
		return out
	}

	if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
		out := new_tensor(a.shape[:], a.requires_grad || b.requires_grad, a.device)
		add_f32_contiguous(out.data, a.data, b.data)
		make_ctx(out, .Add, {a, b})
		return out
	}

	// SIMD broadcast fast paths (CPU only)
	if a.device == .CPU {
		// Scalar broadcast: b is shape all-1
		if is_scalar_shape(b.shape[:]) && is_contiguous(a) {
			out_shape_buf: [MAX_DIMS]i32
			ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
			out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
			add_scalar_contiguous(out.data, a.data, b.data[0])
			make_ctx(out, .Add, {a, b})
			return out
		}
		if is_scalar_shape(a.shape[:]) && is_contiguous(b) {
			out_shape_buf: [MAX_DIMS]i32
			ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
			out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
			add_scalar_contiguous(out.data, b.data, a.data[0])
			make_ctx(out, .Add, {a, b})
			return out
		}
		// Row broadcast: b's shape matches inner dims of out
		if is_contiguous(a) {
			if matches, inner_n := inner_match_size(a.shape[:], b.shape[:]); matches && numel(b.shape[:]) == inner_n && inner_n > 1 {
				outer := numel(a.shape[:]) / i32(inner_n)
				out_shape_buf: [MAX_DIMS]i32
				ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
				out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
				add_row_broadcast(out.data, a.data, b.data, outer, inner_n)
				make_ctx(out, .Add, {a, b})
				return out
			}
		}
		// Col broadcast: b is [outer..., M, 1], a is [outer..., M, N]
		if is_contiguous(a) && is_contiguous(b) {
			if matches, outer, inner_n := col_broadcast_match(a.shape[:], b.shape[:]); matches && inner_n > 1 {
				out_shape_buf: [MAX_DIMS]i32
				ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
				out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
				add_col_broadcast(out.data, a.data, b.data, outer, inner_n)
				make_ctx(out, .Add, {a, b})
				return out
			}
			if matches, outer, inner_n := col_broadcast_match(b.shape[:], a.shape[:]); matches && inner_n > 1 {
				out_shape_buf: [MAX_DIMS]i32
				ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
				out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
				add_col_broadcast(out.data, b.data, a.data, outer, inner_n)
				make_ctx(out, .Add, {a, b})
				return out
			}
		}
	}

	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
	broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
	broadcast_add_into(out.data, out.shape[:], b.data, b.shape[:])
	make_ctx(out, .Add, {a, b})
	return out
}

sub :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "sub: shapes not broadcastable")
	if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
		out := new_tensor(a.shape[:])
		sub_f32_contiguous(out.data, a.data, b.data)
		make_ctx(out, .Sub, {a, b})
		return out
	}

	// Scalar broadcast fast paths
	if is_scalar_shape(b.shape[:]) && is_contiguous(a) {
		out_shape_buf: [MAX_DIMS]i32
		ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
		out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
		add_scalar_contiguous(out.data, a.data, -b.data[0])
		make_ctx(out, .Sub, {a, b})
		return out
	}

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
	if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
		out := new_tensor(a.shape[:])
		mul_f32_contiguous(out.data, a.data, b.data)
		make_ctx(out, .Mul, {a, b})
		return out
	}

	// SIMD broadcast fast paths
	if is_scalar_shape(b.shape[:]) && is_contiguous(a) {
		out_shape_buf: [MAX_DIMS]i32
		ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
		out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
		mul_scalar_contiguous(out.data, a.data, b.data[0])
		make_ctx(out, .Mul, {a, b})
		return out
	}
	if is_scalar_shape(a.shape[:]) && is_contiguous(b) {
		out_shape_buf: [MAX_DIMS]i32
		ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
		out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
		mul_scalar_contiguous(out.data, b.data, a.data[0])
		make_ctx(out, .Mul, {a, b})
		return out
	}
	if is_contiguous(a) {
		if matches, inner_n := inner_match_size(a.shape[:], b.shape[:]); matches && numel(b.shape[:]) == inner_n && inner_n > 1 {
			outer := numel(a.shape[:]) / i32(inner_n)
			out_shape_buf: [MAX_DIMS]i32
			ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
			out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
			mul_row_broadcast(out.data, a.data, b.data, outer, inner_n)
			make_ctx(out, .Mul, {a, b})
			return out
		}
	}
	// Col broadcast
	if is_contiguous(a) && is_contiguous(b) {
		if matches, outer, inner_n := col_broadcast_match(a.shape[:], b.shape[:]); matches && inner_n > 1 {
			out_shape_buf: [MAX_DIMS]i32
			ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
			out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
			mul_col_broadcast(out.data, a.data, b.data, outer, inner_n)
			make_ctx(out, .Mul, {a, b})
			return out
		}
		if matches, outer, inner_n := col_broadcast_match(b.shape[:], a.shape[:]); matches && inner_n > 1 {
			out_shape_buf: [MAX_DIMS]i32
			ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
			out := new_tensor(out_shape_buf[:ndim], a.requires_grad || b.requires_grad, a.device)
			mul_col_broadcast(out.data, b.data, a.data, outer, inner_n)
			make_ctx(out, .Mul, {a, b})
			return out
		}
	}

	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor(out_shape_buf[:ndim])
	mul_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
	make_ctx(out, .Mul, {a, b})
	return out
}

div :: proc(a, b: ^Tensor) -> ^Tensor {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "div: shapes not broadcastable")
	if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
		out := new_tensor(a.shape[:])
		div_f32_contiguous(out.data, a.data, b.data)
		make_ctx(out, .Div, {a, b})
		return out
	}
	out_shape_buf: [MAX_DIMS]i32
	ndim := broadcast_result(out_shape_buf[:], a.shape[:], b.shape[:])
	out := new_tensor(out_shape_buf[:ndim])
	div_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
	make_ctx(out, .Div, {a, b})
	return out
}

neg :: proc(a: ^Tensor) -> ^Tensor {
	out := new_tensor(a.shape[:])
	neg_f32_contiguous(out.data, a.data)
	make_ctx(out, .Neg, {a})
	return out
}

// ---- elementwise unary activations ----------------------------------------
// does it make sense to rely on SIMD for all broadcasting?

relu :: proc(a: ^Tensor) -> ^Tensor {
	out := new_tensor(a.shape[:])
	relu_f32_contiguous(out.data, a.data)
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
		unravel_index(i32(flat), a.shape[:], idx_buf[:])
		out_idx: i32 = 0
		for d in 0..<len(a.shape) {
			v := idx_buf[d]
			if d == int(axis) do v = 0
			out_idx += v * stride_of(out.shape[:], d)
		}
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
		unravel_index(i32(flat), a.shape[:], idx_buf[:])
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
	matmul_f32(out.data, a.data, b.data, M, K, N)
	make_ctx(out, .MatMul, {a, b})
	return out
}
