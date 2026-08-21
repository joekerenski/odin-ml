package ml

UOps :: enum {
	Const,
	Input,
	Add,
	Mul,
	Sum,
	Reshape,
}

UArg :: union {
	f32,
	i32,
	[]i32,
}

UOp :: struct {
	op:    UOps,
	src:   []^UOp,
	shape: [dynamic]i32,
	arg:   UArg,
	data:  []f32,
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
uop_mul :: proc(a, b: ^UOp) -> ^UOp { return uop_bin(.Mul, a, b) }

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

uop_matmul :: proc(a, b: ^UOp) -> ^UOp {
	assert(len(a.shape) == 2 && len(b.shape) == 2, "uop_matmul: needs 2D inputs")
	assert(a.shape[1] == b.shape[0], "uop_matmul: inner dims must match")
	M, K, N := a.shape[0], a.shape[1], b.shape[1]
	return uop_sum(uop_mul(uop_reshape(a, {M, K, 1}), uop_reshape(b, {1, K, N})), 1)
}

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
	case .Sum:
		u.data = uop_reduce_sum(u.src[0].data, u.src[0].shape[:], u.arg.(i32), u.shape[:])
	case .Reshape:
		u.data = u.src[0].data
	}
	return u
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
