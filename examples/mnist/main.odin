package main

// ============================================================================
// MNIST MLP — 784 -> 128 (ReLU) -> 10 (softmax cross-entropy)
//
//   make mnist                 (optimized build)
//   odin run examples/mnist -o:speed
//   ML_DEBUG=2 odin run examples/mnist -o:speed
// ============================================================================

import "core:fmt"
import ml "../../ml"

l1: ml.Linear
l2: ml.Linear

forward :: proc(x: ^ml.Tensor) -> ^ml.Tensor {
	h := ml.relu(ml.linear_forward(&l1, x))
	return ml.linear_forward(&l2, h)
}

main :: proc() {
	fmt.println("=== MNIST MLP: 784 -> 128 -> 10 ===")
	ml.setup_from_env()
	ml.warn_if_unoptimized()
	ml.seed(42)

	X_train := ml.load_idx_images("data/mnist/train-images.idx3-ubyte", flatten = true)
	Y_train := ml.load_idx_labels("data/mnist/train-labels.idx1-ubyte")
	X_test := ml.load_idx_images("data/mnist/t10k-images.idx3-ubyte", flatten = true)
	Y_test := ml.load_idx_labels("data/mnist/t10k-labels.idx1-ubyte")
	fmt.printfln("train: %v  test: %v  ML_DEBUG=%d", X_train.shape, X_test.shape, ml.debug_level)

	l1 = ml.linear(784, 128, .He)
	l2 = ml.linear(128, 10, .Xavier)
	params: [dynamic]^ml.Tensor
	ml.linear_params(&params, l1)
	ml.linear_params(&params, l2)

	tr: ml.Trainer
	ml.trainer_init(&tr, params[:], lr = 0.05, momentum = 0.9, batch_size = 128)
	defer ml.trainer_destroy(&tr)

	if ml.debug_level > 0 {
		// context switch must be in THIS proc (Odin context is by-value).
		old := context.allocator
		context.allocator = ml.trainer_allocator(&tr)
		Xb, Yb := ml.minibatch(X_train, Y_train, 0, tr.batch_size, nil)
		loss := ml.cross_entropy(forward(Xb), Yb)
		ml.counters_reset()
		ml.print_graph(loss, "loss")
		_ = ml.trainer_backward_step(&tr, loss)
		ml.counters_print("first_step")
		context.allocator = old
		ml.trainer_reclaim(&tr)
		ml.debug_level = 0
	}

	EPOCHS :: 20
	for epoch in 0..<EPOCHS {
		loss := ml.trainer_epoch_ce(&tr, X_train, Y_train, forward)
		acc := ml.eval_accuracy(forward, X_test, Y_test, 512)
		fmt.printfln("epoch %2d  loss=%.4f  test_acc=%.2f%%", epoch, loss, acc)
	}
	fmt.println("done.")
}
