package ml

// ============================================================================
// UOp — the one graph. A Tensor IS a UOp node.
//
//   op     what this node computes
//   src    input nodes
//   arg    op-specific payload (Const value, axes, permutation, conv geometry)
//   shape  known at build time
//   data   dense row-major f32; nil until realize() (or elided by fusion)
//
// Frontend ops (ops.odin) build nodes, autograd (autograd.odin) builds more
// nodes, realize (realize.odin) schedules and runs them. Nothing else.
//
// Compositions, not primitives:
//   relu(x)          = max(x, 0)
//   sigmoid(x)       = 1 / (1 + exp(-x))
//   mean(x)          = sum(x) * (1/n)
//   log_softmax(x)   = x - max - log(sum(exp(x - max)))
//   cross_entropy    = -mean(sum(onehot * log_softmax(logits)))
//   layer_norm(x)    = (x - mean) / sqrt(var + eps)
// MatMul / Conv / Pool stay primitives with hand-written kernels.
// ============================================================================

import "base:runtime"
import "core:fmt"

MAX_DIMS :: 8

Op :: enum {
	// sources (always realized)
	Input, // buffer: weights, data
	Const, // scalar, arg = f32, shape {1}

	// elementwise — fusable, broadcast by shape
	Add,
	Sub,
	Mul,
	Div,
	Max,
	CmpLt, // a < b ? 1 : 0
	Neg,
	Exp,
	Log,
	Sqrt,
	Expand, // broadcast src to this shape (identity per element)

	// reduce — arg = []i32 axes; reduced dims kept as 1
	Sum,
	ReduceMax,

	// movement
	Reshape, // same buffer, new shape
	Permute, // arg = []i32 order

	// primitives with hand-written kernels
	MatMul,          // [..., M, K] @ [..., K, N], equal batch dims
	Conv2d,          // src (x, w)
	Conv2dBwdInput,  // src (g, w)  → x shape
	Conv2dBwdWeight, // src (g, x)  → w shape
	MaxPool2d,       // src (x)
	MaxPool2dBwd,    // src (g, x)  → x shape
}

// NCHW window geometry for Conv2d / MaxPool2d and their backward ops.
Window :: struct {
	kH, kW, sH, sW, pH, pW: i32,
}

Arg :: union {
	f32,
	[]i32,
	Window,
}

UOp :: struct {
	op:            Op,
	src:           []^UOp,
	arg:           Arg,
	shape:         []i32,
	data:          []f32,
	requires_grad: bool, // leaves: set by the user; ops: any src requires grad
	grad:          ^UOp, // leaves only, filled by backward()
	internal:      bool, // built by backward(): nobody outside holds it, so its
	                     // buffer can be reused once its readers ran
	// scheduler scratch, valid while epoch is the current realize's: topo
	// position (≥ 0), leaf number -(k + 1), or LEAF_UNNUMBERED
	epoch:         u32,
	pos:           i32,
}

// Set while backward() builds the grad graph: new nodes are internal.
@(private)
building_grad: bool

Tensor :: UOp

op_is_ewise :: proc(op: Op) -> bool {
	#partial switch op {
	case .Add, .Sub, .Mul, .Div, .Max, .CmpLt, .Neg, .Exp, .Log, .Sqrt, .Expand:
		return true
	}
	return false
}

op_is_binary :: proc(op: Op) -> bool {
	#partial switch op {
	case .Add, .Sub, .Mul, .Div, .Max, .CmpLt:
		return true
	}
	return false
}

op_name :: proc(op: Op) -> string {
	return fmt.tprint(op)
}

// Build a node. Copies shape and srcs (with context.allocator); requires_grad
// propagates from srcs.
new_node :: proc(op: Op, shape: []i32, arg: Arg, srcs: ..^UOp) -> ^UOp {
	assert(len(shape) <= MAX_DIMS, "new_node: too many dims")
	u := new(UOp)
	u.op = op
	u.shape = copy_shape(shape)
	u.arg = arg
	u.internal = building_grad
	u.src = make([]^UOp, len(srcs))
	for s, i in srcs {
		u.src[i] = s
		if s.requires_grad do u.requires_grad = true
	}
	return u
}

// Internal bookkeeping (scheduler maps, kernel temporaries) lives on the heap
// and is freed right away — never on the caller's allocator, which only holds
// graph nodes and tensor data (e.g. a per-step arena).
scratch :: proc() -> runtime.Allocator {
	return runtime.heap_allocator()
}

// Post-order DFS over the full graph (through realized nodes).
toposort :: proc(u: ^UOp, topo: ^[dynamic]^UOp, visited: ^map[^UOp]bool) {
	if u in visited^ do return
	visited^[u] = true
	for s in u.src do toposort(s, topo, visited)
	append(topo, u)
}
