package ml

// ============================================================================
// data — MNIST IDX loader, minibatching, evaluation, shuffling.
//
// IDX is the canonical binary format from Yann LeCun's MNIST page. Layout:
//   [0,0,type_code,n_dims] (4 bytes, big-endian)
//   [dim_0, dim_1, ..., dim_n] (4 bytes each, big-endian i32)
//   payload — raw unsigned bytes
//
// Images: magic 0x00000803, dims = [count, rows, cols], payload = count*rows*cols ubytes
// Labels: magic 0x00000801, dims = [count], payload = count ubytes
//
// We convert ubytes to f32 and normalize to [0, 1] (divide by 255). For a
// simple MLP this is sufficient to reach ~97%. Standardization (mean 0.1307,
// std 0.3081) can be added later.
//
// All tensors returned here are leaves with requires_grad = false, allocated
// with the persistent allocator (call before setting the per-step arena).
// ============================================================================

import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

// ---- IDX file parsing -----------------------------------------------------

// Read 4 big-endian bytes as an i32.
big_endian_i32 :: proc(b: []u8, offset: int) -> i32 {
	return (i32(b[offset]) << 24) |
		(i32(b[offset + 1]) << 16) |
		(i32(b[offset + 2]) << 8) |
		i32(b[offset + 3])
}

// Load IDX3 images. If flatten = true, shape is [N, rows*cols]; if false, [N, 1, rows, cols].
load_idx_images :: proc(path: string, flatten: bool = true) -> ^Tensor {
	data, err := os.read_entire_file_from_path(path, context.allocator)
	assert(err == nil, "load_idx_images: failed to read file")
	defer delete(data, context.allocator)

	assert(len(data) >= 16, "load_idx_images: file too short")
	assert(data[0] == 0 && data[1] == 0 && data[2] == 0x08, "load_idx_images: not ubyte idx3")

	n := big_endian_i32(data, 4)
	rows := big_endian_i32(data, 8)
	cols := big_endian_i32(data, 12)
	pixel_count := n * rows * cols
	assert(len(data) >= 16 + int(pixel_count), "load_idx_images: data truncated")

	if flatten {
		out := new_tensor({n, rows * cols}, requires_grad = false)
		for i in 0..<int(pixel_count) {
			out.data[i] = f32(data[16 + i]) / 255.0
		}
		return out
	} else {
		out := new_tensor({n, 1, rows, cols}, requires_grad = false)
		for i in 0..<int(pixel_count) {
			out.data[i] = f32(data[16 + i]) / 255.0
		}
		return out
	}
}

// Load IDX1 labels as a []u8 slice (persistent allocator).
load_idx_labels :: proc(path: string) -> []u8 {
	data, err := os.read_entire_file_from_path(path, context.allocator)
	assert(err == nil, "load_idx_labels: failed to read file")
	defer delete(data, context.allocator)

	assert(len(data) >= 8, "load_idx_labels: file too short")
	assert(data[0] == 0 && data[1] == 0 && data[2] == 0x08, "load_idx_labels: not ubyte idx1")

	n := big_endian_i32(data, 4)
	assert(len(data) >= 8 + int(n), "load_idx_labels: data truncated")

	labels := make([]u8, n)
	for i in 0..<int(n) do labels[i] = data[8 + i]
	return labels
}

// ---- minibatching ---------------------------------------------------------

// Extract a contiguous batch from X and Y. If perm is non-nil, uses perm[offset+i]
// as the source index (for shuffled iteration). X can be any rank; the batch
// dimension is axis 0. Uses context.allocator — in training this is the arena.
minibatch :: proc(
	X: ^Tensor, Y: []u8, batch_idx: int, batch_size: int, perm: []i32 = nil,
) -> (Xb: ^Tensor, Yb: []u8) {
	n := int(X.shape[0])
	per_sample := len(X.data) / n

	batch_shape: [MAX_DIMS]i32
	batch_shape[0] = i32(batch_size)
	for d in 1..<len(X.shape) do batch_shape[d] = X.shape[d]

	Xb = new_tensor(batch_shape[:len(X.shape)], requires_grad = false)
	Yb = make([]u8, batch_size)

	offset := batch_idx * batch_size
	for i in 0..<batch_size {
		idx := offset + i
		if idx >= n do break
		src_idx := idx
		if perm != nil do src_idx = int(perm[idx])
		if src_idx >= n do continue

		src_start := src_idx * per_sample
		dst_start := i * per_sample
		copy(Xb.data[dst_start:dst_start + per_sample], X.data[src_start:src_start + per_sample])
		Yb[i] = Y[src_idx]
	}
	return Xb, Yb
}

// ---- shuffling ------------------------------------------------------------

// Fisher-Yates shuffle of [0, n) using the default PRNG. Allocates with
// context.allocator — call with the persistent allocator (outside the arena).
random_permutation :: proc(n: int) -> []i32 {
	perm := make([]i32, n)
	for i in 0..<n do perm[i] = i32(i)
	for i in 0..<n - 1 {
		j := i + int(rand.float32() * f32(n - i))
		perm[i], perm[j] = perm[j], perm[i]
	}
	return perm
}

// ---- evaluation -----------------------------------------------------------

// Run forward over X in batches, compute argmax accuracy vs Y. Uses its own
// Dynamic_Arena so it doesn't leak intermediate tensors into the caller's
// allocator. The forward proc is the model's inference function (no backward).
eval_accuracy :: proc(
	forward: proc(x: ^Tensor) -> ^Tensor, X: ^Tensor, Y: []u8, batch_size: int,
) -> f32 {
	arena: mem.Dynamic_Arena
	mem.dynamic_arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)

	n := int(X.shape[0])
	correct: int = 0
	total: int = 0

	old_alloc := context.allocator
	context.allocator = mem.dynamic_arena_allocator(&arena)

	for b in 0..<n / batch_size {
		Xb, Yb := minibatch(X, Y, b, batch_size)
		logits := forward(Xb)

		bs := int(logits.shape[0])
		c := int(logits.shape[1])
		for i in 0..<bs {
			best: int = 0
			best_val := logits.data[i * c]
			for j in 1..<c {
				if logits.data[i * c + j] > best_val {
					best_val = logits.data[i * c + j]
					best = j
				}
			}
			if best == int(Yb[i]) do correct += 1
			total += 1
		}
		mem.dynamic_arena_free_all(&arena)
	}

	context.allocator = old_alloc
	if total == 0 do return 0.0
	return f32(correct) / f32(total) * 100.0
}