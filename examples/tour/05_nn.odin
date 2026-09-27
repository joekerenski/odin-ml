package main

// nn is tiny: plain structs, no Module. One Linear, one Conv, one SGD step.
// The arena pattern: graph lives on a Dynamic_Arena, params live on the heap.
//
//   odin run examples/tour/05_nn.odin -file

import "core:fmt"
import "core:mem"
import ml "../../ml"

main :: proc() {
	fmt.println("=== 05 nn + one training step ===\n")
	ml.seed(0)

	// ---- Linear: out = x @ W + b ----
	fc := ml.linear(4, 2, .Xavier)
	fmt.printfln("Linear  W %v  b %v", fc.W.shape, fc.b.shape)

	x := ml.from_data_copy({1, 0, 0, 1}, {1, 4}) // batch 1
	logits := ml.linear_forward(&fc, x)
	ml.realize(logits)
	fmt.printfln("  logits = %v", logits.data)

	// ---- Conv2d: NCHW, bias as [1,C,1,1] ----
	conv := ml.conv2d_layer(1, 2, 3, stride = 1, padding = 1, init = .He)
	img := ml.randn({1, 1, 8, 8}, 0, 1)
	h := ml.relu(ml.conv2d_forward(&conv, img))
	h = ml.max_pool2d(h, 2)
	ml.realize(h)
	fmt.printfln("Conv 1→2 k=3 pad=1, pool2: in [1,1,8,8] → %v", h.shape)

	// ---- one CE step on the Linear, with the arena ----
	fmt.println("\n-- one SGD step (cross-entropy) --")
	params: [dynamic]^ml.Tensor
	ml.linear_params(&params, fc)

	labels := []u8{0}
	opt := ml.new_sgd_list(0.1, 0.0, params[:])

	arena: mem.Dynamic_Arena
	mem.dynamic_arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)

	old := context.allocator
	context.allocator = mem.dynamic_arena_allocator(&arena)

	logits2 := ml.linear_forward(&fc, x)
	loss := ml.cross_entropy(logits2, labels)
	ml.clear_grads(..params[:])
	ml.backward(loss)
	v := ml.item(loss)
	ml.sgd_step(opt)

	context.allocator = old
	mem.dynamic_arena_reset(&arena)
	ml.clear_grads(..params[:]) // grads pointed into the arena we just reset

	fmt.printfln("  loss = %.4f", v)
	fmt.println("  params (W, b) live on the heap and were updated in place")
	fmt.println("  graph + grads were on the arena and are gone")

	fmt.println("\nTrainer wraps this loop: trainer_epoch_ce in mnist/main.odin")
}
