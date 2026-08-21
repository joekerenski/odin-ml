package main

// ============================================================================
// UOp sketch — "the ideal API" for tensors only, no training, no nn.
//
// The bet: there is ONE primitive dialect. High-level ops are NOT primitives,
// they are *compositions*. So there is no MatMul op, no ReLU op, no Conv2d op
// in the IR — those are sugar built from ~10 primitives.
//
//   relu(x)    = max(x, 0)
//   sigmoid(x) = 1 / (1 + exp(-x))
//   matmul(a,b)= sum(mul(reshape(a,[M,K,1]), reshape(b,[1,K,N])), axis=1)
//
//   Tensor -> UOp DAG (build)  ->  realize (interpret / later codegen->Metal)
//
// This is tinygrad's "single dialect" idea: because everything lowers to the
// same few primitives, fusion and rewrite are uniform. Nothing here carries
// conv/pool/CE-specific metadata — each UOp carries only ITS arg.
// ============================================================================

import "core:fmt"

// ---- primitive dialect ------------------------------------------------------

Op :: enum {
	Const,   // arg = f32 value            (scalar source)
	Input,   // leaf tensor (data set)     (a realized buffer / model weight)
	Add,     // elementwise, broadcasting  (src = [a, b])
	Sub,
	Mul,
	Div,
	Max,
	Neg,     // unary                        (src = [a])
	Exp,
	Sum,     // reduce over `axis` (drops it) (arg = i32 axis)
	Reshape, // numel-preserving view         (arg = []i32 new shape)
}

// arg carries ONLY what this one op needs (f32 / axis / shape).
Arg :: union {
	f32,
	i32,
	[]i32,
}

// Every node is tensor-valued: it has a shape. data is nil until realized.
// (Leaves `Input` carry data; the rest are lazily evaluated on realize.)
// Check in tinygrad to see how UOps are defined there ...
UOp :: struct {
	op:    Op,
	src:   []^UOp,
	shape: []i32,
	arg:   Arg,
	data:  []f32,
}

// ---- builders (this is what a "Tensor" frontend would call) -----------------

tensor :: proc(data: []f32, shape: []i32) -> ^UOp {
	u := new(UOp)
	u.op = .Input
	u.data = data
	u.shape = shape
	return u
}

const :: proc(v: f32) -> ^UOp {
	u := new(UOp)
	u.op = .Const
	u.arg = v
	return u
}

bin :: proc(op: Op, a, b: ^UOp) -> ^UOp {
	u := new(UOp)
	u.op = op
	u.src = []^UOp{a, b}
	u.shape = broadcast_shape(a.shape, b.shape)
	return u
}

add :: proc(a, b: ^UOp) -> ^UOp { return bin(.Add, a, b) }
sub :: proc(a, b: ^UOp) -> ^UOp { return bin(.Sub, a, b) }
mul :: proc(a, b: ^UOp) -> ^UOp { return bin(.Mul, a, b) }
div :: proc(a, b: ^UOp) -> ^UOp { return bin(.Div, a, b) }

un :: proc(op: Op, a: ^UOp) -> ^UOp {
	u := new(UOp)
	u.op = op
	u.src = []^UOp{a}
	u.shape = a.shape
	return u
}

neg :: proc(a: ^UOp) -> ^UOp { return un(.Neg, a) }
exp :: proc(a: ^UOp) -> ^UOp { return un(.Exp, a) }

sum :: proc(a: ^UOp, axis: i32) -> ^UOp {
	u := new(UOp)
	u.op = .Sum
	u.src = []^UOp{a}
	u.arg = axis
	oshape := make([dynamic]i32, 0, len(a.shape) - 1)
	for d in 0 ..< len(a.shape) {
		if d == int(axis) do continue
		append(&oshape, a.shape[d])
	}
	u.shape = oshape[:]
	return u
}

reshape :: proc(a: ^UOp, shape: []i32) -> ^UOp {
	u := new(UOp)
	u.op = .Reshape
	u.src = []^UOp{a}
	u.arg = shape
	u.shape = shape
	return u
}

// ---- sugar (compositions — NOT new primitives) ------------------------------

max2 :: proc(a, b: ^UOp) -> ^UOp { return bin(.Max, a, b) }

relu :: proc(x: ^UOp) -> ^UOp { return max2(x, const(0)) }

sigmoid :: proc(x: ^UOp) -> ^UOp {
	// 1 / (1 + exp(-x))
	return div(const(1), add(const(1), exp(neg(x))))
}

matmul :: proc(a, b: ^UOp) -> ^UOp {
	// C[m,n] = sum_k A[m,k] * B[k,n]   ==   sum(mul(A[m,k,1], B[1,k,n]), axis=1)
	M, K, N := a.shape[0], a.shape[1], b.shape[1]
	return sum(mul(reshape(a, {M, K, 1}), reshape(b, {1, K, N})), 1)
}

// ---- shape helpers ----------------------------------------------------------

numel :: proc(shape: []i32) -> i32 {
	n: i32 = 1
	for s in shape do n *= s
	return n
}

stride_of :: proc(shape: []i32, axis: int) -> i32 {
	s: i32 = 1
	for i := axis + 1; i < len(shape); i += 1 do s *= shape[i]
	return s
}

unravel :: proc(flat: i32, shape: []i32, idx: []i32) {
	r := flat
	for d := 0; d < len(shape); d += 1 {
		s := stride_of(shape, d)
		idx[d] = r / s
		r = r % s
	}
}

broadcast_shape :: proc(a, b: []i32) -> []i32 {
	na, nb := len(a), len(b)
	n := na > nb ? na : nb
	out := make([]i32, n)
	for i := 0; i < n; i += 1 {
		av: i32 = 1
		bv: i32 = 1
		if na - 1 - i >= 0 do av = a[na - 1 - i]
		if nb - 1 - i >= 0 do bv = b[nb - 1 - i]
		out[n - 1 - i] = av > bv ? av : bv
	}
	return out
}

// Map an output multi-index (full output rank) to a source flat index,
// applying right-aligned broadcasting (dim==1 stretches).
flat_of :: proc(idx: []i32, odim: int, shape: []i32) -> i32 {
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

// ---- realize: walk the UOp DAG and evaluate (interpreter) -------------------
//
// In the real thing this is where the scheduler + codegen live:
//   UOp DAG -> fuse -> per-kernel loop program -> Metal/C source.
// Here we just interpret, to prove the DAG is correct.

realize :: proc(u: ^UOp) -> []f32 {
	if u.data != nil do return u.data

	switch u.op {
	case .Input:
		// data already set by `tensor`
	case .Const:
		u.data = make([]f32, 1)
		u.data[0] = u.arg.(f32)
	case .Add, .Sub, .Mul, .Div, .Max:
		a := realize(u.src[0])
		b := realize(u.src[1])
		u.data = ewise_bin(u.op, a, u.src[0].shape, b, u.src[1].shape, u.shape)
	case .Neg, .Exp:
		a := realize(u.src[0])
		u.data = ewise_un(u.op, a, u.shape)
	case .Sum:
		a := realize(u.src[0])
		u.data = reduce_sum(a, u.src[0].shape, u.arg.(i32), u.shape)
	case .Reshape:
		u.data = realize(u.src[0]) // view: same bytes, new shape
	}
	return u.data
}

ewise_bin :: proc(op: Op, a: []f32, ashape: []i32, b: []f32, bshape: []i32, oshape: []i32) -> []f32 {
	out := make([]f32, numel(oshape))
	odim := len(oshape)
	idx: [8]i32
	for f := 0; f < len(out); f += 1 {
		unravel(i32(f), oshape, idx[:])
		ai := flat_of(idx[:], odim, ashape)
		bi := flat_of(idx[:], odim, bshape)
		av, bv := a[ai], b[bi]
		#partial switch op {
		case .Add: out[f] = av + bv
		case .Sub: out[f] = av - bv
		case .Mul: out[f] = av * bv
		case .Div: out[f] = av / bv
		case .Max: out[f] = av > bv ? av : bv
		}
	}
	return out
}

ewise_un :: proc(op: Op, a: []f32, oshape: []i32) -> []f32 {
	out := make([]f32, numel(oshape))
	for f := 0; f < len(out); f += 1 {
		#partial switch op {
		case .Neg: out[f] = -a[f]
		case .Exp: out[f] = 1.0
			// NOTE: placeholder, use math.exp for real sigmoid
		}
	}
	return out
}

reduce_sum :: proc(a: []f32, shape: []i32, axis: i32, oshape: []i32) -> []f32 {
	out := make([]f32, numel(oshape))
	idx: [8]i32
	for f := 0; f < len(a); f += 1 {
		unravel(i32(f), shape, idx[:])
		o: i32 = 0
		od := 0
		for d := 0; d < len(shape); d += 1 {
			if d == int(axis) do continue
			o += idx[d] * stride_of(oshape, od)
			od += 1
		}
		out[o] += a[f]
	}
	return out
}

// ---- debug ------------------------------------------------------------------

op_name :: proc(op: Op) -> string {
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
	}
	return "?"
}

dump :: proc(u: ^UOp, indent: int = 0) {
	for _ in 0 ..< indent do fmt.print("  ")
	fmt.printfln("%-7s shape=%v", op_name(u.op), u.shape)
	for s in u.src do dump(s, indent + 1)
}

print_tensor :: proc(u: ^UOp) {
	realize(u)
	fmt.printfln("shape=%v  data=%v", u.shape, u.data)
}

// ---- the "hallucinated" test, as you'd actually write it --------------------

main :: proc() {
	a := tensor({1, 2, 3, 4}, {2, 2})
	b := tensor({5, 6, 7, 8}, {2, 2})

	fmt.println("== add(a, b) ==  (builds UOps, no compute)")
	c := add(a, b)
	dump(c)
	fmt.print("  -> ")
	print_tensor(c) // realize inside

	fmt.println()
	fmt.println("== matmul(a, b) ==  (is just sum(mul(reshape,reshape)))")
	m := matmul(a, b)
	dump(m)
	fmt.print("  -> ")
	print_tensor(m) // expect [[19,22],[43,50]]

	fmt.println()
	fmt.println("== relu(m) ==  (relu = max(x,0), a composition too)")
	r := relu(m)
	dump(r)
	fmt.print("  -> ")
	print_tensor(r)
}

// ============================================================================
// NEXT STEP (fill the gap): the *lowered* form.
//
// The UOp DAG above is still tensor-valued (each node has a shape). To get to
// a Metal kernel you lower it into a per-kernel *loop program* over flat
// memory — a second, smaller dialect:
//
//   LRange, LLoad, LStore, LAdd, LMul, LConst, LBuffer  (maybe LSink/LPhi for reduce)
//
//   relu(a+b) lowers to ONE loop (fusion = both elementwise bodies in one LRange):
//
//     LRange i in [0, n):
//       LLoad  x = A[i]
//       LLoad  y = B[i]
//       LAdd   s = x + y
//       LMax   r = max(s, 0)
//       LStore C[i] = r
//     LEndRange
//
// The whole point of the tensor-valued UOp layer above is that fusion is a
// *rewrite on UOps* before any loop appears — so the scheduler sees
//   Max(Add(Load, Load), Const 0)
// and emits a single LRange instead of two kernels.
// ============================================================================
