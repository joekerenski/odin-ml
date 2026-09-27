package ml

// ============================================================================
// Tensor — creators and helpers. `Tensor :: UOp` (see uop.odin).
//
// Leaves are .Input nodes that own a dense buffer (weights, inputs).
// Ops (ops.odin) return unrealized nodes; realize() fills .data.
//
// Flow: build graph with ops → backward(loss) → optimizer step.
// ============================================================================

import "core:fmt"
import "core:math/rand"

// Seed the global PRNG (creators, random_permutation, etc.).
seed :: proc(s: u64) {
	rand.reset(s)
}

// ---- shape helpers --------------------------------------------------------

numel :: proc(shape: []i32) -> i32 {
	n: i32 = 1
	for s in shape do n *= s
	return n
}

// Row-major stride of a single axis (no allocation).
stride_of :: proc(shape: []i32, axis: int) -> i32 {
	s: i32 = 1
	for i := axis + 1; i < len(shape); i += 1 do s *= shape[i]
	return s
}

// Decompose a dense row-major flat index into a multi-index (len(idx) >= ndim).
unravel_index :: proc(flat: i32, shape: []i32, idx: []i32) {
	r := flat
	for d := len(shape) - 1; d >= 0; d -= 1 {
		idx[d] = r % shape[d]
		r /= shape[d]
	}
}

shapes_equal :: proc(a, b: []i32) -> bool {
	if len(a) != len(b) do return false
	for i in 0 ..< len(a) {
		if a[i] != b[i] do return false
	}
	return true
}

copy_shape :: proc(shape: []i32) -> []i32 {
	out := make([]i32, len(shape))
	copy(out, shape)
	return out
}

// ---- leaves ---------------------------------------------------------------

// Fresh zeroed leaf with the given shape.
new_tensor :: proc(shape: []i32, requires_grad := false) -> ^Tensor {
	t := new_node(.Input, shape, nil)
	t.data = make([]f32, numel(shape))
	t.requires_grad = requires_grad
	return t
}

// Leaf viewing an existing buffer (no copy). `data` must outlive the tensor.
from_data :: proc(data: []f32, shape: []i32, requires_grad := false) -> ^Tensor {
	assert(i32(len(data)) == numel(shape), "from_data: data length != product(shape)")
	t := new_node(.Input, shape, nil)
	t.data = data
	t.requires_grad = requires_grad
	return t
}

// Leaf owning a copy of `data`.
from_data_copy :: proc(data: []f32, shape: []i32, requires_grad := false) -> ^Tensor {
	assert(i32(len(data)) == numel(shape), "from_data_copy: data length != product(shape)")
	t := new_tensor(shape, requires_grad)
	copy(t.data, data)
	return t
}

// Scalar constant node (shape {1}); broadcasts in ewise ops.
scalar :: proc(v: f32) -> ^Tensor {
	t := new_node(.Const, {1}, v)
	t.data = make([]f32, 1)
	t.data[0] = v
	return t
}

// Realize t and copy its values into a new leaf (no graph, no grad).
clone :: proc(t: ^Tensor, requires_grad := false) -> ^Tensor {
	realize(t)
	return from_data_copy(t.data, t.shape, requires_grad)
}

zeros :: proc(shape: []i32, requires_grad := false) -> ^Tensor {
	return new_tensor(shape, requires_grad)
}

ones :: proc(shape: []i32, requires_grad := false) -> ^Tensor {
	t := new_tensor(shape, requires_grad)
	for i in 0 ..< len(t.data) do t.data[i] = 1.0
	return t
}

uniform :: proc(shape: []i32, low, high: f32, requires_grad := false) -> ^Tensor {
	t := new_tensor(shape, requires_grad)
	scale := high - low
	for i in 0 ..< len(t.data) do t.data[i] = low + scale * rand.float32()
	return t
}

randn :: proc(shape: []i32, mean, std: f32, requires_grad := false) -> ^Tensor {
	t := new_tensor(shape, requires_grad)
	for i in 0 ..< len(t.data) do t.data[i] = mean + std * rand.float32_normal(0, 1)
	return t
}

// Elementwise close: |a-b| <= atol + rtol*|b|. Realizes both; same numel required.
allclose :: proc(a, b: ^Tensor, rtol: f32 = 1e-5, atol: f32 = 1e-6) -> bool {
	realize(a)
	realize(b)
	if numel(a.shape) != numel(b.shape) do return false
	for i in 0 ..< len(a.data) {
		diff := abs(a.data[i] - b.data[i])
		if diff > atol + rtol * abs(b.data[i]) do return false
	}
	return true
}

// ---- grad management ------------------------------------------------------

// Nil out .grad on parameters. Call after an arena reset — the grad memory
// was just reclaimed, so the pointers are dangling.
clear_grads :: proc(params: ..^Tensor) {
	for p in params do p.grad = nil
}

// Zero existing .grad buffers in place (instead of the arena pattern).
zero_grad :: proc(tensors: ..^Tensor) {
	for t in tensors {
		if t.grad != nil && t.grad.data != nil {
			for i in 0 ..< len(t.grad.data) do t.grad.data[i] = 0
		}
	}
}

// ---- pretty printing ------------------------------------------------------

println :: proc(t: ^Tensor) {
	realize(t)
	fmt.printfln("Tensor(%v, shape=%v, requires_grad=%v)", t.op, t.shape, t.requires_grad)
	fmt.printfln("  data: %v", t.data)
	if t.grad != nil && t.grad.data != nil {
		fmt.printfln("  grad: %v", t.grad.data)
	}
}
