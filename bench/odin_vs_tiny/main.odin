package main

// Dump a few golden results + timings for tinygrad_compare.py.
//   odin run bench/odin_vs_tiny -o:speed
// Format (stdout):
//   RESULT <name>
//   <f32 space-separated>
//   TIME <name> <ms>

import "core:fmt"
import "core:time"
import ml "../../ml"

print_result :: proc(name: string, t: ^ml.Tensor) {
	ml.realize(t)
	fmt.printf("RESULT %s\n", name)
	for i in 0..<len(t.data) {
		if i > 0 do fmt.print(" ")
		fmt.printf("%.8g", t.data[i])
	}
	fmt.println()
}

main :: proc() {
	// --- correctness (fixed inputs, match tinygrad_compare.py) ---
	a := ml.from_data_copy({1, 2, 3, 4}, {2, 2})
	b := ml.from_data_copy({10, 20, 30, 40}, {2, 2})
	print_result("add_same", ml.add(a, b))
	print_result("mul_same", ml.mul(a, b))

	A := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {2, 3})
	row := ml.from_data_copy({10, 100, 1000}, {3})
	print_result("mul_row", ml.mul(A, row))

	col := ml.from_data_copy({1, 2}, {2, 1})
	print_result("add_col", ml.add(A, col))

	M := ml.from_data_copy({1, 2, 3, 4}, {2, 2})
	N := ml.from_data_copy({5, 6, 7, 8}, {2, 2})
	print_result("matmul_2x2", ml.matmul(M, N))

	X := ml.from_data_copy({1, 0, 0, 1, -1, 2}, {3, 2})
	w := ml.from_data_copy({1, -1}, {2, 1})
	bias := ml.from_data_copy({0.5}, {1, 1})
	print_result("relu_xw_b", ml.relu(ml.add(ml.matmul(X, w), bias)))

	// MSE mean: mean((x*w+b - y)^2)  — regression forward shape
	x := ml.from_data_copy({0, 1, 2, 3}, {4})
	y := ml.from_data_copy({1, 3.5, 6, 8.5}, {4}) // ~ 2.5x + 1
	ww := ml.from_data_copy({2.5}, {1})
	bb := ml.from_data_copy({1.0}, {1})
	pred := ml.add(ml.mul(x, ww), bb)
	resid := ml.sub(pred, y)
	print_result("mse_mean", ml.mean(ml.mul(resid, resid)))

	// one backward: L = mean((x*w)^2), dL/dw  (w requires grad)
	xw := ml.from_data_copy({1, 2, 3, 4}, {4})
	pw := ml.from_data_copy({0.5}, {1}, requires_grad = true)
	prod := ml.mul(xw, pw)
	loss := ml.mean(ml.mul(prod, prod))
	ml.backward(loss)
	fmt.printf("RESULT grad_w\n%.8g\n", pw.grad.data[0])

	// --- speed (pre-allocated leaves; time realize of single op) ---
	bench_matmul(256)
	bench_matmul(512)
	bench_add(1 << 20) // 1M
}

bench_matmul :: proc(n: i32) {
	A := ml.randn({n, n}, 0, 1)
	B := ml.randn({n, n}, 0, 1)
	// warmup
	ml.realize(ml.matmul(A, B))
	best := time.Duration(max(i64))
	for _ in 0..<5 {
		C := ml.matmul(A, B)
		t0 := time.tick_now()
		ml.realize(C)
		dt := time.tick_since(t0)
		if dt < best do best = dt
	}
	ms := f64(best) / 1e6
	fmt.printf("TIME matmul_%d %.4f\n", n, ms)
}

bench_add :: proc(n: i32) {
	A := ml.randn({n}, 0, 1)
	B := ml.randn({n}, 0, 1)
	ml.realize(ml.add(A, B))
	best := time.Duration(max(i64))
	for _ in 0..<10 {
		C := ml.add(A, B)
		t0 := time.tick_now()
		ml.realize(C)
		dt := time.tick_since(t0)
		if dt < best do best = dt
	}
	ms := f64(best) / 1e6
	fmt.printf("TIME add_%d %.4f\n", n, ms)
}
