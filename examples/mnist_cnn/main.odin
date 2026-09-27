package main

// ============================================================================
// MNIST CNN — tiny LeNet-ish stack to exercise Conv2d + MaxPool + Linear.
//
//   make cnn                 (optimized build)
//   odin run examples/mnist_cnn -o:speed
//   ML_DEBUG=2 odin run examples/mnist_cnn -o:speed
// ============================================================================

import "core:fmt"
import ml "../../ml"

c1: ml.Conv2d
c2: ml.Conv2d
fc: ml.Linear

forward :: proc(x: ^ml.Tensor) -> ^ml.Tensor {
	img := x
	if len(x.shape) == 2 {
		img = ml.reshape(x, {x.shape[0], 1, 28, 28})
	}
	h := ml.relu(ml.conv2d_forward(&c1, img))
	h = ml.max_pool2d(h, 2)
	h = ml.relu(ml.conv2d_forward(&c2, h))
	h = ml.max_pool2d(h, 2)
	return ml.linear_forward(&fc, ml.flatten(h))
}

main :: proc() {
	fmt.println("=== MNIST CNN: conv→pool→conv→pool→fc ===")
	ml.setup_from_env()
	ml.warn_if_unoptimized()
	ml.seed(42)

	X_train := ml.load_idx_images("data/mnist/train-images.idx3-ubyte", flatten = false)
	Y_train := ml.load_idx_labels("data/mnist/train-labels.idx1-ubyte")
	X_test := ml.load_idx_images("data/mnist/t10k-images.idx3-ubyte", flatten = false)
	Y_test := ml.load_idx_labels("data/mnist/t10k-labels.idx1-ubyte")
	fmt.printfln("train: %v  test: %v  ML_DEBUG=%d", X_train.shape, X_test.shape, ml.debug_level)

	c1 = ml.conv2d_layer(1, 8, 3, stride = 1, padding = 1, init = .He)
	c2 = ml.conv2d_layer(8, 16, 3, stride = 1, padding = 1, init = .He)
	fc = ml.linear(16 * 7 * 7, 10, .Xavier)

	params: [dynamic]^ml.Tensor
	ml.conv_params(&params, c1)
	ml.conv_params(&params, c2)
	ml.linear_params(&params, fc)

	tr: ml.Trainer
	ml.trainer_init(&tr, params[:], lr = 0.05, momentum = 0.9, batch_size = 128)
	defer ml.trainer_destroy(&tr)
	fmt.printfln("params: %d tensors", len(params))

	if ml.debug_level > 0 {
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

	EPOCHS :: 3
	for epoch in 0..<EPOCHS {
		loss := ml.trainer_epoch_ce(&tr, X_train, Y_train, forward)
		acc := ml.eval_accuracy(forward, X_test, Y_test, 256)
		fmt.printfln("epoch %d  loss=%.4f  test_acc=%.2f%%", epoch, loss, acc)
	}
	fmt.println("done.")
}
