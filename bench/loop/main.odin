package main

// Compiler-loop scoreboard. Same binary before and after IR/fusion work.
//
//   odin run bench/loop -o:speed -no-bounds-check -disable-assert
//
// Columns:
//   kernels   realize launches (fused group = 1; views are not kernels)
//   fused     extra ewise ops absorbed into fused kernels (0 before fusion)
//   alloc_kb  output buffers allocated in realize
//   ms        mean wall time per iteration
//
// Workloads:
//   ewise_chain  relu(relu(a+b)*c + d)     fusion-heavy
//   linear_relu  relu(X@W + b)             GEMM + ewise epilogue
//   mse_chain    mean((x*w+b - y)^2)       regression forward
//   mlp_fwd      784→128 ReLU → 10         MNIST-shaped forward
//   mlp_step     same + CE + backward      training step

import "core:fmt"
import "core:time"
import ml "../../ml"

ITERS :: 30
WARMUP :: 5

main :: proc() {
	ml.seed(0)
	fmt.println("name            kernels  fused  alloc_kb      ms")
	bench_ewise_chain()
	bench_linear_relu()
	bench_mse_chain()
	bench_mlp_fwd()
	bench_mlp_step()
}

report :: proc(name: string, kernels, fused: int, alloc_kb, ms: f64) {
	fmt.printfln("BENCH %-12s %7d %6d %9.1f %7.3f", name, kernels, fused, alloc_kb, ms)
}

bench_ewise_chain :: proc() {
	n: i32 = 1024
	a := ml.randn({n, n}, 0, 1)
	b := ml.randn({n, n}, 0, 1)
	c := ml.randn({n, n}, 0, 1)
	d := ml.randn({n, n}, 0, 1)
	for _ in 0 ..< WARMUP {
		ml.realize(ml.relu(ml.add(ml.mul(ml.relu(ml.add(a, b)), c), d)))
	}
	ml.counters_reset()
	t0 := time.tick_now()
	for _ in 0 ..< ITERS {
		ml.realize(ml.relu(ml.add(ml.mul(ml.relu(ml.add(a, b)), c), d)))
	}
	ms := f64(time.tick_since(t0)) / 1e6 / f64(ITERS)
	report("ewise_chain", ml.counters.kernels / ITERS, ml.counters.fused_ops / ITERS,
		f64(ml.counters.bytes_alloc) / 1024.0 / f64(ITERS), ms)
}

bench_linear_relu :: proc() {
	B, In, Out: i32 = 128, 784, 128
	X := ml.randn({B, In}, 0, 1)
	W := ml.randn({In, Out}, 0, 0.05, requires_grad = true)
	b := ml.zeros({Out}, requires_grad = true)
	for _ in 0 ..< WARMUP {
		ml.realize(ml.relu(ml.add(ml.matmul(X, W), b)))
	}
	ml.counters_reset()
	t0 := time.tick_now()
	for _ in 0 ..< ITERS {
		ml.realize(ml.relu(ml.add(ml.matmul(X, W), b)))
	}
	ms := f64(time.tick_since(t0)) / 1e6 / f64(ITERS)
	report("linear_relu", ml.counters.kernels / ITERS, ml.counters.fused_ops / ITERS,
		f64(ml.counters.bytes_alloc) / 1024.0 / f64(ITERS), ms)
}

bench_mse_chain :: proc() {
	n: i32 = 1 << 16
	x := ml.randn({n}, 0, 1)
	y := ml.randn({n}, 0, 1)
	w := ml.randn({1}, 0, 0.5, requires_grad = true)
	b := ml.zeros({1}, requires_grad = true)
	for _ in 0 ..< WARMUP {
		pred := ml.add(ml.mul(x, w), b)
		resid := ml.sub(pred, y)
		ml.realize(ml.mean(ml.mul(resid, resid)))
	}
	ml.counters_reset()
	t0 := time.tick_now()
	for _ in 0 ..< ITERS {
		pred := ml.add(ml.mul(x, w), b)
		resid := ml.sub(pred, y)
		ml.realize(ml.mean(ml.mul(resid, resid)))
	}
	ms := f64(time.tick_since(t0)) / 1e6 / f64(ITERS)
	report("mse_chain", ml.counters.kernels / ITERS, ml.counters.fused_ops / ITERS,
		f64(ml.counters.bytes_alloc) / 1024.0 / f64(ITERS), ms)
}

bench_mlp_fwd :: proc() {
	B: i32 = 128
	X := ml.randn({B, 784}, 0, 1)
	W1 := ml.randn({784, 128}, 0, 0.05, requires_grad = true)
	b1 := ml.zeros({128}, requires_grad = true)
	W2 := ml.randn({128, 10}, 0, 0.05, requires_grad = true)
	b2 := ml.zeros({10}, requires_grad = true)
	for _ in 0 ..< WARMUP {
		h := ml.relu(ml.add(ml.matmul(X, W1), b1))
		ml.realize(ml.add(ml.matmul(h, W2), b2))
	}
	ml.counters_reset()
	t0 := time.tick_now()
	for _ in 0 ..< ITERS {
		h := ml.relu(ml.add(ml.matmul(X, W1), b1))
		ml.realize(ml.add(ml.matmul(h, W2), b2))
	}
	ms := f64(time.tick_since(t0)) / 1e6 / f64(ITERS)
	report("mlp_fwd", ml.counters.kernels / ITERS, ml.counters.fused_ops / ITERS,
		f64(ml.counters.bytes_alloc) / 1024.0 / f64(ITERS), ms)
}

bench_mlp_step :: proc() {
	B: i32 = 128
	X := ml.randn({B, 784}, 0, 1)
	W1 := ml.randn({784, 128}, 0, 0.05, requires_grad = true)
	b1 := ml.zeros({128}, requires_grad = true)
	W2 := ml.randn({128, 10}, 0, 0.05, requires_grad = true)
	b2 := ml.zeros({10}, requires_grad = true)
	labels := make([]u8, B)
	for i in 0 ..< int(B) do labels[i] = u8(i % 10)

	for _ in 0 ..< WARMUP {
		ml.clear_grads(W1, b1, W2, b2)
		h := ml.relu(ml.add(ml.matmul(X, W1), b1))
		logits := ml.add(ml.matmul(h, W2), b2)
		ml.backward(ml.cross_entropy(logits, labels))
	}
	ml.counters_reset()
	t0 := time.tick_now()
	for _ in 0 ..< ITERS {
		ml.clear_grads(W1, b1, W2, b2)
		h := ml.relu(ml.add(ml.matmul(X, W1), b1))
		logits := ml.add(ml.matmul(h, W2), b2)
		ml.backward(ml.cross_entropy(logits, labels))
	}
	ms := f64(time.tick_since(t0)) / 1e6 / f64(ITERS)
	report("mlp_step", ml.counters.kernels / ITERS, ml.counters.fused_ops / ITERS,
		f64(ml.counters.bytes_alloc) / 1024.0 / f64(ITERS), ms)
}
