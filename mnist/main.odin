package main

// ============================================================================
// MNIST MLP — 784 -> 128 (ReLU) -> 10 (softmax cross-entropy)
//
// Canonical "hello world" of ML. Tests the full stack: data loading, layers,
// autograd, optimizer, eval. Targets ~97% test accuracy in ~20 epochs.
//
//   odin run mnist -o:speed -no-bounds-check
//
// Uses the same arena pattern as regression.odin:
//   - Dataset and model parameters live in the persistent allocator.
//   - Each training step builds the forward+backward graph inside a
//     Dynamic_Arena, runs backward, applies gradients, then free_all.
// ============================================================================

import "core:fmt"
import "core:mem"
import ml "../ml"

// ---- model (file-scope so forward() can be passed to eval_accuracy) -------

l1: ml.Linear
l2: ml.Linear

forward :: proc(x: ^ml.Tensor) -> ^ml.Tensor {
	h := ml.relu(ml.linear_forward(&l1, x))    // [B, 784] -> [B, 128]
	return ml.linear_forward(&l2, h)             // [B, 128] -> [B, 10]
}

main :: proc() {
	fmt.println("=== MNIST MLP: 784 -> 128 -> 10 ===")

	// ---- load data (persistent allocator) ----
	X_train := ml.load_idx_images("data/train-images.idx3-ubyte", flatten = true)
	Y_train := ml.load_idx_labels("data/train-labels.idx1-ubyte")
	X_test := ml.load_idx_images("data/t10k-images.idx3-ubyte", flatten = true)
	Y_test := ml.load_idx_labels("data/t10k-labels.idx1-ubyte")
	fmt.printfln("train: %v  test: %v", X_train.shape, X_test.shape)

	// ---- model parameters (persistent) ----
	l1 = ml.linear(784, 128, .He)
	l2 = ml.linear(128, 10, .Xavier)
	opt := ml.new_sgd(0.05, 0.9, l1.W, l1.b, l2.W, l2.b)

	// ---- arena for per-step graph ----
	arena: mem.Dynamic_Arena
	mem.dynamic_arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)

	BS: int = 128
	EPOCHS: int = 20
	N := int(X_train.shape[0])
	batches := N / BS

	for epoch in 0..<EPOCHS {
		// Shuffle training indices each epoch (persistent allocator — outside arena).
		perm := ml.random_permutation(N)

		old_alloc := context.allocator
		last_loss: f32 = 0

		for b in 0..<batches {
			context.allocator = mem.dynamic_arena_allocator(&arena)

			Xb, Yb := ml.minibatch(X_train, Y_train, b, BS, perm)
			logits := forward(Xb)
			loss := ml.cross_entropy(logits, Yb)
			last_loss = loss.data[0]

			ml.clear_grads(l1.W, l1.b, l2.W, l2.b)
			ml.backward(loss)
			ml.sgd_step(opt)

			context.allocator = old_alloc
			mem.dynamic_arena_free_all(&arena)
			ml.clear_grads(l1.W, l1.b, l2.W, l2.b)
		}

		delete(perm)

		acc := ml.eval_accuracy(forward, X_test, Y_test, 512)
		fmt.printfln("epoch %2d  loss=%.4f  test_acc=%.2f%%", epoch, last_loss, acc)
	}

	fmt.println("done.")
}