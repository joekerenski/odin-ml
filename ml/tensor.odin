package ml

// ============================================================================
// Tensor — the central type, modeled on tinygrad.
//
// A Tensor is either a leaf (a weight or an input) or the result of an op.
// It owns:
//   data          flat f32 buffer, row-major
//   shape         dimension sizes, e.g. [2, 3]
//   strides       row-major strides, e.g. [3, 1] for shape [2, 3]
//   requires_grad whether this tensor needs a gradient
//   grad          accumulated gradient (nil until backward runs)
//   ctx          the op that produced this tensor; nil for leaves
//
// The compute graph is implicit: every non-leaf tensor points at a Context,
// which references its parent tensors and the op kind. backward() walks that
// graph in reverse and fills in .grad everywhere requires_grad is set.
//
// ALLOCATION: all tensors are created through `context.allocator`. The intended
// usage is to set `context.allocator` to a `Dynamic_Arena` for each training
// step, build the forward+backward graph inside it, read out the loss and
// apply gradients, then call `dynamic_arena_free_all` to reclaim everything at
// once. Parameters (weights) are allocated with the persistent allocator
// (before the arena is set) so they survive across steps. After `free_all`,
// call `clear_grads(params...)` to nil out dangling `.grad` pointers.
// ============================================================================

import "core:fmt"
import "core:math/rand"

MAX_DIMS :: 8

Tensor :: struct {
	data:          []f32,
	shape:         [dynamic]i32,
	strides:       [dynamic]i32,
	requires_grad: bool,
	grad:          ^Tensor,
	ctx:           ^Context,
	device:        Device,
}

Op :: enum {
	Add, Sub, Mul, Div, Neg,
	MatMul,
	Sum, Reshape, Transpose,
	ReLU, Sigmoid,
}

Context :: struct {
	op:      Op,
	parents: [dynamic]^Tensor,
	axis:    i32,          // Sum: which axis (-1 = all); Transpose: axis0
	axis1:   i32,          // Transpose: axis1
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

// Decompose a dense row-major flat index into multi-index (idx must be >= ndim).
// Correct order: walk dims from the left (major → minor) using strides.
unravel_index :: proc(flat: i32, shape: []i32, idx: []i32) {
	r := flat
	for d := 0; d < len(shape); d += 1 {
		s := stride_of(shape, d)
		idx[d] = r / s
		r = r % s
	}
}

compute_strides :: proc(shape: []i32) -> [dynamic]i32 {
	st: [dynamic]i32 = make([dynamic]i32, len(shape))
	if len(shape) == 0 do return st
	st[len(shape) - 1] = 1
	for i := len(shape) - 2; i >= 0; i -= 1 {
		st[i] = st[i + 1] * shape[i + 1]
	}
	return st
}

shapes_equal :: proc(a, b: []i32) -> bool {
	if len(a) != len(b) do return false
	for i in 0..<len(a) {
		if a[i] != b[i] do return false
	}
	return true
}

copy_shape :: proc(shape: []i32) -> [dynamic]i32 {
	out: [dynamic]i32 = make([dynamic]i32, len(shape))
	for i in 0..<len(shape) do out[i] = shape[i]
	return out
}

// ---- allocation -----------------------------------------------------------

// Allocate a fresh tensor with the given shape. Uses context.allocator.
// Device defaults to `default_device` (CPU unless changed).
new_tensor :: proc(shape: []i32, requires_grad := false, device := default_device) -> ^Tensor {
	assert(len(shape) <= MAX_DIMS, "new_tensor: too many dims")
	t: ^Tensor = new(Tensor)
	t.data = make([]f32, numel(shape))
	t.shape = copy_shape(shape)
	t.strides = compute_strides(t.shape[:])
	t.requires_grad = requires_grad
	t.device = device
	return t
}

// View an existing f32 buffer as a tensor (does not copy the buffer).
// `data` must outlive the tensor. For stack literals / owned data prefer
// `from_data_copy`.
from_data :: proc(data: []f32, shape: []i32, requires_grad := false, device := default_device) -> ^Tensor {
	assert(i32(len(data)) == numel(shape), "from_data: data length != product(shape)")
	assert(len(shape) <= MAX_DIMS, "from_data: too many dims")
	t: ^Tensor = new(Tensor)
	t.data = data
	t.shape = copy_shape(shape)
	t.strides = compute_strides(t.shape[:])
	t.requires_grad = requires_grad
	t.device = device
	return t
}

// Copy `data` into a freshly allocated tensor buffer (safe, always owns its data).
from_data_copy :: proc(data: []f32, shape: []i32, requires_grad := false, device := default_device) -> ^Tensor {
	assert(i32(len(data)) == numel(shape), "from_data_copy: data length != product(shape)")
	assert(len(shape) <= MAX_DIMS, "from_data_copy: too many dims")
	t := new_tensor(shape, requires_grad, device)
	for i in 0..<len(data) do t.data[i] = data[i]
	return t
}

// Deep-copy shape + data. Does not copy the compute-graph (grad/ctx are nil).
clone :: proc(t: ^Tensor, requires_grad := false) -> ^Tensor {
	return from_data_copy(t.data, t.shape[:], requires_grad)
}

// Contiguous layout: strides match dense row-major for the current shape.
is_contiguous :: proc(t: ^Tensor) -> bool {
	expected: i32 = 1
	for i := len(t.shape) - 1; i >= 0; i -= 1 {
		if t.shape[i] == 0 do return true
		if t.strides[i] != expected do return false
		expected *= t.shape[i]
	}
	return true
}

// Elementwise close: |a-b| <= atol + rtol*|b| for every element. Same numel required.
allclose :: proc(a, b: ^Tensor, rtol: f32 = 1e-5, atol: f32 = 1e-6) -> bool {
	if len(a.data) != len(b.data) do return false
	for i in 0..<len(a.data) {
		diff := a.data[i] - b.data[i]
		if diff < 0 do diff = -diff
		bound := atol + rtol * (b.data[i] < 0 ? -b.data[i] : b.data[i])
		if diff > bound do return false
	}
	return true
}

// ---- creators -------------------------------------------------------------

zeros :: proc(shape: []i32, requires_grad := false, device := default_device) -> ^Tensor {
	return new_tensor(shape, requires_grad, device)
}

ones :: proc(shape: []i32, requires_grad := false, device := default_device) -> ^Tensor {
	t := new_tensor(shape, requires_grad, device)
	for i in 0..<len(t.data) do t.data[i] = 1.0
	return t
}

uniform :: proc(shape: []i32, low, high: f32, requires_grad := false, device := default_device) -> ^Tensor {
	t := new_tensor(shape, requires_grad, device)
	scale := high - low
	for i in 0..<len(t.data) do t.data[i] = low + scale * rand.float32()
	return t
}

randn :: proc(shape: []i32, mean, std: f32, requires_grad := false, device := default_device) -> ^Tensor {
	t := new_tensor(shape, requires_grad, device)
	for i in 0..<len(t.data) do t.data[i] = mean + std * rand.float32_normal(0, 1)
	return t
}

// ---- grad management ------------------------------------------------------

// Nil out .grad on a list of parameter tensors. Call this after arena
// free_all — the grad memory was just reclaimed, so the pointers are dangling.
clear_grads :: proc(params: ..^Tensor) {
	for p in params do p.grad = nil
}

// Zero out existing .grad data (without freeing). Use when reusing grad buffers
// across steps instead of the arena pattern.
zero_grad :: proc(tensors: ..^Tensor) {
	for t in tensors {
		if t.grad != nil {
			for i in 0..<len(t.grad.data) do t.grad.data[i] = 0
		}
	}
}

// ---- pretty printing ------------------------------------------------------

println :: proc(t: ^Tensor) {
	fmt.printfln("Tensor(shape=%v, requires_grad=%v)", t.shape, t.requires_grad)
	fmt.printfln("  data: %v", t.data)
	if t.grad != nil {
		fmt.printfln("  grad: %v", t.grad.data)
	}
}
