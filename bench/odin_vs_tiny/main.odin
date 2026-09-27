package main

// Dump a few golden results + timings for tinygrad_compare.py.
//   odin run bench/odin_vs_tiny -o:speed
// Format (stdout):
//   RESULT <name>
//   <f32 space-separated>
//   TIME <name> <ms>

import "core:fmt"
import "core:math"
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

// Deterministic inputs shared with tinygrad_compare.py: sin(0.7*i + k).
seq :: proc(shape: []i32, k: f64, requires_grad := false) -> ^ml.Tensor {
	t := ml.new_tensor(shape, requires_grad)
	for i in 0 ..< len(t.data) do t.data[i] = f32(math.sin(f64(i) * 0.7 + k))
	return t
}

// Milestone 1: new ops, compositions, batched matmul, Adam — fwd + grads.
milestone1 :: proc() {
	// log / sqrt
	x := ml.add(ml.mul(seq({4}, 0), ml.scalar(0.5)), ml.scalar(1)) // positive
	xl := ml.clone(x, requires_grad = true)
	ml.backward(ml.sum(ml.mul(ml.log(xl), ml.sqrt(xl))))
	print_result("log_sqrt_dx", xl.grad)

	// reduce max with ties, fwd + grad
	mx := ml.from_data_copy({1, 5, 5, 2, 0, -1}, {2, 3}, requires_grad = true)
	mm := ml.max_axis(mx, 1)
	print_result("max_axis", mm)
	ml.backward(ml.sum(ml.mul(mm, ml.from_data_copy({1, 2}, {2, 1}))))
	print_result("max_axis_dx", mx.grad)

	// softmax family
	s := seq({3, 4}, 1, requires_grad = true)
	print_result("softmax", ml.softmax(s, 1))
	print_result("log_softmax", ml.log_softmax(s, 1))
	print_result("logsumexp", ml.logsumexp(s, 1))
	ml.backward(ml.sum(ml.mul(ml.softmax(s, 1), seq({3, 4}, 2))))
	print_result("softmax_dx", s.grad)

	// layer norm
	ln := seq({2, 5}, 3, requires_grad = true)
	print_result("layernorm", ml.layer_norm(ln))
	ml.backward(ml.sum(ml.mul(ml.layer_norm(ln), seq({2, 5}, 4))))
	print_result("layernorm_dx", ln.grad)

	// cross-entropy (composition)
	lg := seq({3, 4}, 5, requires_grad = true)
	ce := ml.cross_entropy(lg, {2, 0, 3})
	print_result("cross_entropy", ce)
	ml.backward(ce)
	print_result("cross_entropy_dx", lg.grad)

	// batched matmul + 2D weight + broadcast batch, with grads
	ba := seq({2, 2, 3}, 6, requires_grad = true)
	bb := seq({2, 3, 2}, 7, requires_grad = true)
	bw := seq({3, 2}, 8, requires_grad = true)
	bc := seq({1, 3, 2}, 9, requires_grad = true)
	l := ml.add(ml.add(ml.sum(ml.matmul(ba, bb)), ml.sum(ml.mul(ml.matmul(ba, bw), ml.scalar(2)))),
		ml.sum(ml.mul(ml.matmul(ba, bc), ml.scalar(3))))
	print_result("bmm", ml.matmul(ba, bb))
	ml.backward(l)
	print_result("bmm_da", ba.grad)
	print_result("bmm_db", bb.grad)
	print_result("bmm_dw", bw.grad)
	print_result("bmm_dc", bc.grad)

	// attention: softmax(q k^T / sqrt(d)) v   [B=2, T=3, D=4]
	q := seq({2, 3, 4}, 10, requires_grad = true)
	k := seq({2, 3, 4}, 11, requires_grad = true)
	v := seq({2, 3, 4}, 12, requires_grad = true)
	att := ml.softmax(ml.mul(ml.matmul(q, ml.mT(k)), ml.scalar(0.5)), 2)
	o := ml.matmul(att, v)
	print_result("attention", o)
	ml.backward(ml.sum(ml.mul(o, seq({2, 3, 4}, 13))))
	print_result("attention_dq", q.grad)
	print_result("attention_dk", k.grad)
	print_result("attention_dv", v.grad)

	// Adam / AdamW: 5 steps on sum((w - t)^2)
	for wd, name in ([]f32{0, 0.1}) {
		w := seq({5}, 14, requires_grad = true)
		t := seq({5}, 15)
		params := []^ml.Tensor{w}
		opt := ml.new_adam(params, lr = 0.1, weight_decay = wd)
		for _ in 0 ..< 5 {
			ml.clear_grads(w)
			ml.backward(ml.sum(ml.square(ml.sub(w, t))))
			ml.adam_step(opt)
		}
		print_result(name == 0 ? "adam_w" : "adamw_w", w)
	}
}

main :: proc() {
	ml.setup_from_env()
	milestone1()

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

	// conv2d: 1x1x3x3 * 1x1x2x2 ones → window sums
	cx := ml.from_data_copy({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3})
	cw := ml.from_data_copy({1, 1, 1, 1}, {1, 1, 2, 2})
	print_result("conv2d_k2", ml.conv2d(cx, cw, stride = 1, padding = 0))

	// max_pool2d 2x2 stride 2
	px := ml.from_data_copy({
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, {1, 1, 4, 4})
	print_result("maxpool2d", ml.max_pool2d(px, kernel_size = 2))

	// conv backward: dW
	gx := ml.from_data_copy({1, 2, 3, 4}, {1, 1, 2, 2}, requires_grad = true)
	gw := ml.from_data_copy({0.5, 0.5, 0.5, 0.5}, {1, 1, 2, 2}, requires_grad = true)
	gy := ml.conv2d(gx, gw, 1, 0)
	ml.backward(ml.sum(gy))
	print_result("conv_dW", gw.grad)

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
