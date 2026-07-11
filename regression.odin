package main

// ============================================================================
// Linear regression sanity test.
//
//   model:   y_hat = x * w + b           (w, b are trainable scalars)
//   loss:    L = mean((y_hat - y)^2)
//   train:   SGD with momentum on (w, b)
//
// We make noisy data from y = 2.5*x + 1.0 and check that gradient descent
// recovers w ~= 2.5 and b ~= 1.0.
//
// ARENA PATTERN: each training step builds a forward+backward graph inside a
// Dynamic_Arena. After reading the loss and applying gradients, we call
// dynamic_arena_free_all to reclaim everything at once. Parameters (w, b) live
// in the persistent allocator and survive across steps; we nil their .grad
// pointers after free_all since that memory was just reclaimed.
// ============================================================================

import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"
import "ml"

main :: proc() {

	fmt.println("=== Linear regression: y = w*x + b ===")

	true_w: f32 = 2.5
	true_b: f32 = 1.0
	N: int = 64

	// Synthetic data: x ~ Uniform[-1, 2), y = 2.5*x + 1 + N(0, 0.1)
	x_data := make([]f32, N)
	y_data := make([]f32, N)
	for i in 0..<N {
		x := rand.float32() * 3.0 - 1.0
		noise := rand.float32_normal(0, 0.1)
		x_data[i] = x
		y_data[i] = true_w * x + true_b + noise
	}

	// Inputs and labels: leaves, no grad. Allocated with persistent allocator.
	X := ml.from_data(x_data, {i32(N)}, requires_grad = false)
	Y := ml.from_data(y_data, {i32(N)}, requires_grad = false)

	// Parameters: leaves with grad. Also persistent.
	w := ml.randn({1}, 0.0, 0.5, requires_grad = true)
	b := ml.zeros({1}, requires_grad = true)

	fmt.printfln("init    w=%.4f  b=%.4f", w.data[0], b.data[0])

	opt := ml.new_sgd(0.05, 0.9, w, b)

	// Arena for per-step graph. Grows on demand, freed all at once each step.
	arena: mem.Dynamic_Arena
	mem.dynamic_arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)

	epochs: int = 300
	for epoch in 0..<epochs {
		// --- switch to arena allocator for this step's graph ---
		old_alloc := context.allocator
		context.allocator = mem.dynamic_arena_allocator(&arena)

		// ---- forward ----
		wx := ml.mul(X, w)        // [N] * [1] -> [N]  (broadcast)
		pred := ml.add(wx, b)     // [N] + [1] -> [N]
		resid := ml.sub(pred, Y)  // [N]
		sq := ml.mul(resid, resid) // [N]
		loss := ml.mean(sq)       // [1]

		// snapshot loss before freeing
		loss_val := loss.data[0]

		// ---- backward + step ----
		ml.clear_grads(w, b)
		ml.backward(loss)
		ml.sgd_step(opt)

		// --- restore allocator and reclaim the whole graph ---
		context.allocator = old_alloc
		mem.dynamic_arena_free_all(&arena)
		ml.clear_grads(w, b)  // nil out dangling grad pointers

		if epoch % 30 == 0 || epoch == epochs - 1 {
			fmt.printfln(
				"epoch %3d  loss=%.6f  w=%.4f  b=%.4f",
				epoch, loss_val, w.data[0], b.data[0],
			)
		}
	}

	fmt.println("---")
	fmt.printfln("truth   w=%.4f  b=%.4f", true_w, true_b)
	fmt.printfln("learned w=%.4f  b=%.4f", w.data[0], b.data[0])
	fmt.println("done.")
}
