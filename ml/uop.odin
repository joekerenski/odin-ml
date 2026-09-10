package ml

import "core:math"

// ============================================================================
// UOp — one dialect from Tensor programs to kernels.
//
// High-level Tensor ops lower here. Elementwise ops (Add/Mul/Max/…) fuse.
// MatMul/Conv/CE stay primitives (GEMM via Accelerate until a later pass
// recovers it from mul+sum). ReLU/Sigmoid are compositions:
//   relu(x)    = max(x, 0)
//   sigmoid(x) = 1 / (1 + exp(-x))
// ============================================================================

UOps :: enum {
	Const,
	Input,
	Add,
	Sub,
	Mul,
	Div,
	Max,
	Neg,
	Exp,
	Sum,
	Reshape,
	Transpose,
	MatMul,
	Conv2d,
	MaxPool2d,
	CrossEntropy,
}

UArg :: union {
	f32,
	i32,
	[]i32,
}

UOp :: struct {
	op:     UOps,
	src:    []^UOp,
	shape:  [dynamic]i32,
	arg:    UArg,
	data:   []f32,
	tensor: ^Tensor, // back-pointer; nil for Const and composition internals
}

uop_is_ewise :: proc(op: UOps) -> bool {
	#partial switch op {
	case .Add, .Sub, .Mul, .Div, .Max, .Neg, .Exp:
		return true
	}
	return false
}

uop_is_binary :: proc(op: UOps) -> bool {
	#partial switch op {
	case .Add, .Sub, .Mul, .Div, .Max:
		return true
	}
	return false
}

uop_is_view :: proc(op: UOps) -> bool {
	return op == .Reshape || op == .Transpose
}

uop_name :: proc(op: UOps) -> string {
	switch op {
	case .Const: return "Const"
	case .Input: return "Input"
	case .Add: return "Add"
	case .Sub: return "Sub"
	case .Mul: return "Mul"
	case .Div: return "Div"
	case .Max: return "Max"
	case .Neg: return "Neg"
	case .Exp: return "Exp"
	case .Sum: return "Sum"
	case .Reshape: return "Reshape"
	case .Transpose: return "Transpose"
	case .MatMul: return "MatMul"
	case .Conv2d: return "Conv2d"
	case .MaxPool2d: return "MaxPool2d"
	case .CrossEntropy: return "CrossEntropy"
	}
	return "?"
}

uop_srcs :: proc(ps: ..^UOp) -> []^UOp {
	s := make([]^UOp, len(ps))
	for i in 0 ..< len(ps) do s[i] = ps[i]
	return s
}

uop_input :: proc(data: []f32, shape: []i32) -> ^UOp {
	assert(i32(len(data)) == numel(shape), "uop_input: data length != product(shape)")
	u := new(UOp)
	u.op = .Input
	u.shape = copy_shape(shape)
	u.data = make([]f32, len(data))
	copy(u.data, data)
	return u
}

uop_const :: proc(v: f32) -> ^UOp {
	u := new(UOp)
	u.op = .Const
	u.arg = v
	u.shape = copy_shape({1})
	return u
}

uop_bin :: proc(op: UOps, a, b: ^UOp) -> ^UOp {
	assert(shapes_broadcast(a.shape[:], b.shape[:]), "uop_bin: shapes not broadcastable")
	u := new(UOp)
	u.op = op
	u.src = uop_srcs(a, b)
	buf: [MAX_DIMS]i32
	n := broadcast_result(buf[:], a.shape[:], b.shape[:])
	u.shape = copy_shape(buf[:n])
	return u
}

uop_add :: proc(a, b: ^UOp) -> ^UOp { return uop_bin(.Add, a, b) }
uop_sub :: proc(a, b: ^UOp) -> ^UOp { return uop_bin(.Sub, a, b) }
uop_mul :: proc(a, b: ^UOp) -> ^UOp { return uop_bin(.Mul, a, b) }
uop_div :: proc(a, b: ^UOp) -> ^UOp { return uop_bin(.Div, a, b) }
uop_max :: proc(a, b: ^UOp) -> ^UOp { return uop_bin(.Max, a, b) }

uop_un :: proc(op: UOps, a: ^UOp) -> ^UOp {
	u := new(UOp)
	u.op = op
	u.src = uop_srcs(a)
	u.shape = copy_shape(a.shape[:])
	return u
}

uop_neg :: proc(a: ^UOp) -> ^UOp { return uop_un(.Neg, a) }
uop_exp :: proc(a: ^UOp) -> ^UOp { return uop_un(.Exp, a) }

uop_sum :: proc(a: ^UOp, axis: i32) -> ^UOp {
	u := new(UOp)
	u.op = .Sum
	u.src = uop_srcs(a)
	u.arg = axis
	if axis == -1 {
		u.shape = copy_shape({1})
		return u
	}
	assert(axis >= 0 && axis < i32(len(a.shape)), "uop_sum: axis out of range")
	oshape := make([dynamic]i32, 0, len(a.shape) - 1)
	for d in 0 ..< len(a.shape) {
		if d == int(axis) do continue
		append(&oshape, a.shape[d])
	}
	u.shape = oshape
	return u
}

uop_reshape :: proc(a: ^UOp, shape: []i32) -> ^UOp {
	assert(numel(a.shape[:]) == numel(shape), "uop_reshape: element count mismatch")
	u := new(UOp)
	u.op = .Reshape
	u.src = uop_srcs(a)
	u.shape = copy_shape(shape)
	return u
}

// Primitive GEMM. Composition (sum of mul of reshapes) is correct math but
// needs fusion + a GEMM pattern-match before it can replace this.
uop_matmul :: proc(a, b: ^UOp) -> ^UOp {
	assert(len(a.shape) == 2 && len(b.shape) == 2, "uop_matmul: needs 2D inputs")
	assert(a.shape[1] == b.shape[0], "uop_matmul: inner dims must match")
	u := new(UOp)
	u.op = .MatMul
	u.src = uop_srcs(a, b)
	u.shape = copy_shape({a.shape[0], b.shape[1]})
	return u
}

// Tensor DAG → UOp DAG. Already-realized nodes become Input buffers.
lower_tensor :: proc(t: ^Tensor, memo: ^map[^Tensor]^UOp) -> ^UOp {
	if t == nil do return nil
	if existing, ok := memo[t]; ok do return existing

	if is_realized(t) {
		u := new(UOp)
		u.op = .Input
		u.shape = copy_shape(t.shape[:])
		u.data = t.data
		u.tensor = t
		memo[t] = u
		return u
	}

	p := t.ctx.parents
	u: ^UOp
	switch t.ctx.op {
	case .Add:
		u = uop_add(lower_tensor(p[0], memo), lower_tensor(p[1], memo))
	case .Sub:
		u = uop_sub(lower_tensor(p[0], memo), lower_tensor(p[1], memo))
	case .Mul:
		u = uop_mul(lower_tensor(p[0], memo), lower_tensor(p[1], memo))
	case .Div:
		u = uop_div(lower_tensor(p[0], memo), lower_tensor(p[1], memo))
	case .Neg:
		u = uop_neg(lower_tensor(p[0], memo))
	case .ReLU:
		u = uop_max(lower_tensor(p[0], memo), uop_const(0))
	case .Sigmoid:
		x := lower_tensor(p[0], memo)
		u = uop_div(uop_const(1), uop_add(uop_const(1), uop_exp(uop_neg(x))))
	case .Sum:
		u = new(UOp)
		u.op = .Sum
		u.src = uop_srcs(lower_tensor(p[0], memo))
		u.arg = t.ctx.axis
		u.shape = copy_shape(t.shape[:])
	case .Reshape:
		u = uop_reshape(lower_tensor(p[0], memo), t.shape[:])
	case .Transpose:
		u = new(UOp)
		u.op = .Transpose
		u.src = uop_srcs(lower_tensor(p[0], memo))
		u.shape = copy_shape(t.shape[:])
	case .MatMul:
		u = uop_matmul(lower_tensor(p[0], memo), lower_tensor(p[1], memo))
	case .Conv2d:
		u = new(UOp)
		u.op = .Conv2d
		u.src = uop_srcs(lower_tensor(p[0], memo), lower_tensor(p[1], memo))
		u.shape = copy_shape(t.shape[:])
	case .MaxPool2d:
		u = new(UOp)
		u.op = .MaxPool2d
		u.src = uop_srcs(lower_tensor(p[0], memo))
		u.shape = copy_shape(t.shape[:])
	case .CrossEntropy:
		u = new(UOp)
		u.op = .CrossEntropy
		u.src = uop_srcs(lower_tensor(p[0], memo))
		u.shape = copy_shape(t.shape[:])
	}
	u.tensor = t
	memo[t] = u
	return u
}

uop_topo :: proc(u: ^UOp, topo: ^[dynamic]^UOp, visited: ^map[^UOp]bool) {
	if u == nil || u in visited^ do return
	visited^[u] = true
	for s in u.src do uop_topo(s, topo, visited)
	append(topo, u)
}

// Standalone interpreter (tests / sketches). Tensor realize uses the scheduler.
uop_realize :: proc(u: ^UOp) -> ^UOp {
	if u == nil do return u
	if u.data != nil do return u
	for s in u.src do uop_realize(s)
	switch u.op {
	case .Input:
	case .Const:
		u.data = make([]f32, 1)
		u.data[0] = u.arg.(f32)
	case .Add:
		u.data = make([]f32, numel(u.shape[:]))
		if shapes_equal(u.src[0].shape[:], u.src[1].shape[:]) {
			add_f32_contiguous(u.data, u.src[0].data, u.src[1].data)
		} else {
			for i in 0 ..< len(u.data) do u.data[i] = 0
			broadcast_add_into(u.data, u.shape[:], u.src[0].data, u.src[0].shape[:])
			broadcast_add_into(u.data, u.shape[:], u.src[1].data, u.src[1].shape[:])
		}
	case .Sub:
		u.data = make([]f32, numel(u.shape[:]))
		if shapes_equal(u.src[0].shape[:], u.src[1].shape[:]) {
			sub_f32_contiguous(u.data, u.src[0].data, u.src[1].data)
		} else {
			for i in 0 ..< len(u.data) do u.data[i] = 0
			broadcast_add_into(u.data, u.shape[:], u.src[0].data, u.src[0].shape[:])
			neg_b := make([]f32, len(u.src[1].data))
			for i in 0 ..< len(neg_b) do neg_b[i] = -u.src[1].data[i]
			broadcast_add_into(u.data, u.shape[:], neg_b, u.src[1].shape[:])
		}
	case .Mul:
		u.data = make([]f32, numel(u.shape[:]))
		if shapes_equal(u.src[0].shape[:], u.src[1].shape[:]) {
			mul_f32_contiguous(u.data, u.src[0].data, u.src[1].data)
		} else {
			mul_broadcast_into(
				u.data, u.shape[:],
				u.src[0].data, u.src[0].shape[:],
				u.src[1].data, u.src[1].shape[:],
			)
		}
	case .Div:
		u.data = make([]f32, numel(u.shape[:]))
		div_broadcast_into(
			u.data, u.shape[:],
			u.src[0].data, u.src[0].shape[:],
			u.src[1].data, u.src[1].shape[:],
		)
	case .Max:
		u.data = make([]f32, numel(u.shape[:]))
		n := len(u.data)
		if shapes_equal(u.src[0].shape[:], u.src[1].shape[:]) {
			a, b := u.src[0].data, u.src[1].data
			for i in 0 ..< n do u.data[i] = a[i] > b[i] ? a[i] : b[i]
		} else {
			idx: [MAX_DIMS]i32
			odim := len(u.shape)
			for f in 0 ..< n {
				unravel_index(i32(f), u.shape[:], idx[:])
				ai := flat_of_shape(idx[:], odim, u.src[0].shape[:])
				bi := flat_of_shape(idx[:], odim, u.src[1].shape[:])
				av, bv := u.src[0].data[ai], u.src[1].data[bi]
				u.data[f] = av > bv ? av : bv
			}
		}
	case .Neg:
		u.data = make([]f32, numel(u.shape[:]))
		neg_f32_contiguous(u.data, u.src[0].data)
	case .Exp:
		u.data = make([]f32, numel(u.shape[:]))
		for i in 0 ..< len(u.data) do u.data[i] = math.exp(u.src[0].data[i])
	case .Sum:
		u.data = uop_reduce_sum(u.src[0].data, u.src[0].shape[:], u.arg.(i32), u.shape[:])
	case .Reshape:
		u.data = u.src[0].data
	case .MatMul:
		a, b := u.src[0], u.src[1]
		M, K, N := a.shape[0], a.shape[1], b.shape[1]
		u.data = make([]f32, int(M * N))
		matmul_f32(u.data, a.data, b.data, M, K, N)
	case .Transpose, .Conv2d, .MaxPool2d, .CrossEntropy:
		assert(u.tensor != nil, "uop_realize: heavy op needs a tensor")
		realize_one(u.tensor)
		u.data = u.tensor.data
	}
	return u
}

flat_of_shape :: proc(idx: []i32, odim: int, shape: []i32) -> i32 {
	sdim := len(shape)
	s: i32 = 0
	for d := 0; d < sdim; d += 1 {
		od := odim - sdim + d
		v := idx[od]
		if shape[d] == 1 do v = 0
		s += v * stride_of(shape, d)
	}
	return s
}

uop_reduce_sum :: proc(a: []f32, shape: []i32, axis: i32, oshape: []i32) -> []f32 {
	out := make([]f32, numel(oshape))
	if axis == -1 {
		s: f32 = 0
		for v in a do s += v
		out[0] = s
		return out
	}
	idx: [MAX_DIMS]i32
	for f in 0 ..< len(a) {
		unravel_index(i32(f), shape, idx[:])
		o: i32 = 0
		od := 0
		for d in 0 ..< len(shape) {
			if d == int(axis) do continue
			o += idx[d] * stride_of(oshape, od)
			od += 1
		}
		out[o] += a[f]
	}
	return out
}
