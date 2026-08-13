package main

// ============================================================================
// Tensor op suite — shape creation, elementwise, broadcast, reductions,
// shape ops, matmul, activations, and a tiny SIMD kernel check.
//
// Run from repo root:
//   odin run tensor_ops
// ============================================================================

import "core:fmt"
import "core:simd"
import "core:time"
import "core:math"
import ml "../ml"

failed: int
passed: int

expect :: proc(cond: bool, msg: string) {
	if cond {
		passed += 1
		fmt.printfln("  ok  %s", msg)
	} else {
		failed += 1
		fmt.printfln("  FAIL %s", msg)
	}
}

expect_close :: proc(got, want: ^ml.Tensor, msg: string) {
	if ml.allclose(got, want, 1e-5, 1e-5) {
		passed += 1
		fmt.printfln("  ok  %s", msg)
	} else {
		failed += 1
		fmt.printfln("  FAIL %s  got=%v want=%v", msg, got.data, want.data)
	}
}

main :: proc() {
	fmt.println("=== tensor ops ===")
	fmt.printfln("HAS_HARDWARE_SIMD=%v  (portable #simd → NEON on this Mac)", simd.HAS_HARDWARE_SIMD)
	fmt.println()

	test_create_and_shape()
	test_from_data_roundtrip()
	test_elementwise_same_shape()
	test_broadcast()
	test_unary()
	test_reductions()
	test_reshape_transpose()
	test_matmul()
	test_simd_kernels_match_scalar()
	test_broadcast_simd_correctness()
	test_broadcast_simd_vs_scalar_perf()
	test_chained()
	test_conv_pool()
	test_conv_grad()
	test_params_helper()
	test_div_broadcast_grad()
	test_gradcheck()
	test_cross_entropy()
	test_views()

	fmt.println()
	fmt.printfln("=== %d passed, %d failed ===", passed, failed)
	if failed > 0 do return
}

// ---- individual groups ----------------------------------------------------

test_create_and_shape :: proc() {
	fmt.println("-- create / shape --")
	z := ml.zeros({2, 3})
	expect(len(z.data) == 6, "zeros numel")
	expect(z.shape[0] == 2 && z.shape[1] == 3, "zeros shape")
	expect(ml.is_contiguous(z), "zeros contiguous")
	expect(z.strides[0] == 3 && z.strides[1] == 1, "zeros strides row-major")

	o := ml.ones({4})
	expect(o.data[0] == 1 && o.data[3] == 1, "ones fill")

	c := ml.clone(o)
	expect(ml.allclose(c, o), "clone matches")
	c.data[0] = 99
	expect(o.data[0] == 1, "clone is a deep copy")
}

test_from_data_roundtrip :: proc() {
	fmt.println("-- from_data / from_data_copy --")
	buf := []f32{1, 2, 3, 4, 5, 6}
	view := ml.from_data(buf, {2, 3})
	expect(view.data[0] == 1 && view.data[5] == 6, "from_data view")
	// mutation of underlying buffer is visible
	buf[0] = 42
	expect(view.data[0] == 42, "from_data shares buffer")

	owned := ml.from_data_copy({10, 20, 30}, {3})
	expect(owned.data[0] == 10 && owned.data[2] == 30, "from_data_copy values")
	expect(ml.is_contiguous(owned), "from_data_copy contiguous")
}

test_elementwise_same_shape :: proc() {
	fmt.println("-- elementwise same-shape --")
	a := ml.from_data_copy({1, 2, 3, 4}, {2, 2})
	b := ml.from_data_copy({10, 20, 30, 40}, {2, 2})

	s := ml.add(a, b)
	expect_close(s, ml.from_data_copy({11, 22, 33, 44}, {2, 2}), "add")

	d := ml.sub(b, a)
	expect_close(d, ml.from_data_copy({9, 18, 27, 36}, {2, 2}), "sub")

	m := ml.mul(a, b)
	expect_close(m, ml.from_data_copy({10, 40, 90, 160}, {2, 2}), "mul")

	q := ml.div(b, a)
	expect_close(q, ml.from_data_copy({10, 10, 10, 10}, {2, 2}), "div")

	n := ml.neg(a)
	expect_close(n, ml.from_data_copy({-1, -2, -3, -4}, {2, 2}), "neg")
}

test_broadcast :: proc() {
	fmt.println("-- broadcast --")
	// [3] + [1] → [3]
	a := ml.from_data_copy({1, 2, 3}, {3})
	bias := ml.from_data_copy({10}, {1})
	out := ml.add(a, bias)
	expect_close(out, ml.from_data_copy({11, 12, 13}, {3}), "add [3]+[1]")

	// [2,3] * [3] → [2,3]  (right-aligned, last dim matches)
	A := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {2, 3})
	row := ml.from_data_copy({10, 100, 1000}, {3})
	scaled := ml.mul(A, row)
	expect_close(scaled, ml.from_data_copy({10, 200, 3000, 40, 500, 6000}, {2, 3}), "mul [2,3]*[3]")

	// [2,3] + [2,1] → [2,3]
	col := ml.from_data_copy({1, 2}, {2, 1})
	bc := ml.add(A, col)
	expect_close(bc, ml.from_data_copy({2, 3, 4, 6, 7, 8}, {2, 3}), "add [2,3]+[2,1]")
}

test_unary :: proc() {
	fmt.println("-- unary / activations --")
	x := ml.from_data_copy({-2, -0.5, 0, 0.5, 3}, {5})
	r := ml.relu(x)
	expect_close(r, ml.from_data_copy({0, 0, 0, 0.5, 3}, {5}), "relu")

	s := ml.sigmoid(ml.from_data_copy({0}, {1}))
	ml.realize(s)
	// sigmoid(0) = 0.5
	expect(s.data[0] > 0.499 && s.data[0] < 0.501, "sigmoid(0)≈0.5")
}

test_reductions :: proc() {
	fmt.println("-- reductions --")
	t := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {2, 3})
	all := ml.sum(t, -1)
	ml.realize(all)
	expect(all.shape[0] == 1 && all.data[0] == 21, "sum all")

	// sum over cols → shape [2,1]
	rows := ml.sum(t, 1)
	ml.realize(rows)
	expect(rows.shape[0] == 2 && rows.shape[1] == 1, "sum axis=1 shape")
	expect(rows.data[0] == 6 && rows.data[1] == 15, "sum axis=1 values")

	// mean of [1,2,3,4] = 2.5
	m := ml.mean(ml.from_data_copy({1, 2, 3, 4}, {4}))
	ml.realize(m)
	expect(m.data[0] > 2.499 && m.data[0] < 2.501, "mean")
}

test_reshape_transpose :: proc() {
	fmt.println("-- reshape / transpose --")
	t := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {2, 3})
	r := ml.reshape(t, {3, 2})
	ml.realize(r)
	expect(r.shape[0] == 3 && r.shape[1] == 2, "reshape shape")
	expect(r.data[0] == 1 && r.data[5] == 6, "reshape data (row-major flat)")

	// 2x3 → 3x2 transpose
	// [[1,2,3],[4,5,6]]^T = [[1,4],[2,5],[3,6]]
	tt := ml.T(t)
	expect(tt.shape[0] == 3 && tt.shape[1] == 2, "T shape")
	want := ml.from_data_copy({1, 4, 2, 5, 3, 6}, {3, 2})
	expect_close(tt, want, "T values")
}

test_matmul :: proc() {
	fmt.println("-- matmul --")
	// [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
	A := ml.from_data_copy({1, 2, 3, 4}, {2, 2})
	B := ml.from_data_copy({5, 6, 7, 8}, {2, 2})
	C := ml.matmul(A, B)
	expect_close(C, ml.from_data_copy({19, 22, 43, 50}, {2, 2}), "matmul 2x2")

	// [3,2] @ [2,1] → [3,1]
	X := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {3, 2})
	w := ml.from_data_copy({10, 1}, {2, 1})
	y := ml.matmul(X, w)
	// rows: 1*10+2*1=12, 3*10+4*1=34, 5*10+6*1=56
	expect_close(y, ml.from_data_copy({12, 34, 56}, {3, 1}), "matmul [3,2]@[2,1]")
}

// Kernel path vs explicit scalar: same result.
test_simd_kernels_match_scalar :: proc() {
	fmt.println("-- simd kernels == scalar --")
	n := 17 // not a multiple of 4 → exercises the tail
	a := make([]f32, n)
	b := make([]f32, n)
	out_simd := make([]f32, n)
	out_scalar := make([]f32, n)
	for i in 0..<n {
		a[i] = f32(i) * 0.5 - 3
		b[i] = f32(i) * 0.25 + 1
	}

	ml.add_f32_contiguous(out_simd, a, b)
	for i in 0..<n do out_scalar[i] = a[i] + b[i]
	ok := true
	for i in 0..<n {
		d := out_simd[i] - out_scalar[i]
		if d < 0 do d = -d
		if d > 1e-6 do ok = false
	}
	expect(ok, "add_f32_contiguous matches scalar (n=17)")

	ml.mul_f32_contiguous(out_simd, a, b)
	for i in 0..<n do out_scalar[i] = a[i] * b[i]
	ok = true
	for i in 0..<n {
		d := out_simd[i] - out_scalar[i]
		if d < 0 do d = -d
		if d > 1e-6 do ok = false
	}
	expect(ok, "mul_f32_contiguous matches scalar (n=17)")

	ml.relu_f32_contiguous(out_simd, a)
	for i in 0..<n do out_scalar[i] = a[i] > 0 ? a[i] : 0
	ok = true
	for i in 0..<n {
		d := out_simd[i] - out_scalar[i]
		if d < 0 do d = -d
		if d > 1e-6 do ok = false
	}
	expect(ok, "relu_f32_contiguous matches scalar (n=17)")
}

test_chained :: proc() {
	fmt.println("-- chained expression --")
	// y = relu(X @ w + b) for a tiny batch
	X := ml.from_data_copy({1, 0, 0, 1, -1, 2}, {3, 2})
	w := ml.from_data_copy({1, -1}, {2, 1})
	b := ml.from_data_copy({0.5}, {1, 1})
	// Xw: [1, -1, -3]^T then +0.5 → [1.5, -0.5, -2.5] relu → [1.5, 0, 0]
	y := ml.relu(ml.add(ml.matmul(X, w), b))
	expect_close(y, ml.from_data_copy({1.5, 0, 0}, {3, 1}), "relu(Xw+b)")
}

// ---- SIMD broadcast correctness ----

test_broadcast_simd_correctness :: proc() {
	fmt.println("-- broadcast SIMD correctness --")

	// Scalar broadcast: [N] + [1]
	{
		n := 1024
		a_data := make([]f32, n)
		for i in 0..<n do a_data[i] = f32(i) * 0.1
		a := ml.from_data_copy(a_data, {i32(n)})
		b := ml.from_data_copy({3.14}, {1})
		out := ml.add(a, b)
		want := make([]f32, n)
		for i in 0..<n do want[i] = a_data[i] + 3.14
		w := ml.from_data_copy(want, {i32(n)})
		expect(ml.allclose(out, w), "scalar broadcast add [N]+[1]")
	}

	// Row broadcast: [M,N] + [N]
	{
		M, N := 32, 128
		a_data := make([]f32, M * N)
		for i in 0..<M*N do a_data[i] = f32(i)
		a := ml.from_data_copy(a_data, {i32(M), i32(N)})
		b_data := make([]f32, N)
		for i in 0..<N do b_data[i] = f32(i) * 0.01
		b := ml.from_data_copy(b_data, {i32(N)})
		out := ml.add(a, b)
		ml.realize(out)
		// verify
		ok := true
		for i in 0..<M {
			for j in 0..<N {
				idx := i * N + j
				expected := a_data[idx] + b_data[j]
				if math.abs(out.data[idx] - expected) > 1e-5 do ok = false
			}
		}
		expect(ok, "row broadcast add [32,128]+[128]")
	}

	// Row broadcast mul
	{
		M, N := 16, 64
		a_data := make([]f32, M * N)
		for i in 0..<M*N do a_data[i] = f32(i) + 1
		a := ml.from_data_copy(a_data, {i32(M), i32(N)})
		b_data := make([]f32, N)
		for i in 0..<N do b_data[i] = f32(i) + 1
		b := ml.from_data_copy(b_data, {i32(N)})
		out := ml.mul(a, b)
		ml.realize(out)
		ok := true
		for i in 0..<M {
			for j in 0..<N {
				idx := i * N + j
				expected := a_data[idx] * b_data[j]
				if math.abs(out.data[idx] - expected) > 1e-4 do ok = false
			}
		}
		expect(ok, "row broadcast mul [16,64]*[64]")
	}

	// Scalar broadcast sub: [N] - [1]
	{
		n := 512
		a_data := make([]f32, n)
		for i in 0..<n do a_data[i] = f32(i) * 2
		a := ml.from_data_copy(a_data, {i32(n)})
		b := ml.from_data_copy({1.5}, {1})
		out := ml.sub(a, b)
		ml.realize(out)
		ok := true
		for i in 0..<n {
			expected := a_data[i] - 1.5
			if math.abs(out.data[i] - expected) > 1e-5 do ok = false
		}
		expect(ok, "scalar broadcast sub [N]-[1]")
	}

	// Col broadcast: [M,N] + [M,1]
	{
		M, N := 32, 64
		a_data := make([]f32, M * N)
		col := make([]f32, M)
		for i in 0..<M*N do a_data[i] = f32(i)
		for i in 0..<M do col[i] = f32(i) * 0.5
		a := ml.from_data_copy(a_data, {i32(M), i32(N)})
		b := ml.from_data_copy(col, {i32(M), 1})
		out := ml.add(a, b)
		ml.realize(out)
		ok := true
		for i in 0..<M {
			for j in 0..<N {
				idx := i * N + j
				expected := a_data[idx] + col[i]
				if math.abs(out.data[idx] - expected) > 1e-5 do ok = false
			}
		}
		expect(ok, "col broadcast add [32,64]+[32,1]")
	}

	// Col broadcast mul
	{
		M, N := 16, 32
		a_data := make([]f32, M * N)
		col := make([]f32, M)
		for i in 0..<M*N do a_data[i] = f32(i) + 1
		for i in 0..<M do col[i] = f32(i) + 1
		a := ml.from_data_copy(a_data, {i32(M), i32(N)})
		b := ml.from_data_copy(col, {i32(M), 1})
		out := ml.mul(a, b)
		ml.realize(out)
		ok := true
		for i in 0..<M {
			for j in 0..<N {
				idx := i * N + j
				expected := a_data[idx] * col[i]
				if math.abs(out.data[idx] - expected) > 1e-4 do ok = false
			}
		}
		expect(ok, "col broadcast mul [16,32]*[16,1]")
	}
}

// ---- SIMD broadcast perf comparison ----
//
// Compare the new SIMD broadcast fast paths against the old scalar-broadcast
// path. Numbers give us a concrete answer to "did SIMD help?".

bench_ms :: proc(label: string, n_iters: int, body: proc(a: ^ml.Tensor, b: ^ml.Tensor), a: ^ml.Tensor, b: ^ml.Tensor) {
	// warmup
	for _ in 0..<3 do body(a, b)
	times := make([]f64, n_iters)
	defer delete(times)
	for i in 0..<n_iters {
		t0 := time.tick_now()
		body(a, b)
		dt := time.duration_seconds(time.tick_since(t0))
		times[i] = dt * 1e3  // ms
	}
	// sort
	for i in 1..<n_iters {
		v := times[i]
		j := i
		for j > 0 && times[j - 1] > v {
			times[j] = times[j - 1]
			j -= 1
		}
		times[j] = v
	}
	med := times[n_iters / 2]
	fmt.printfln("  %-40s  best=%.3f ms  med=%.3f ms", label, times[0], med)
}

test_broadcast_simd_vs_scalar_perf :: proc() {
	fmt.println("-- broadcast SIMD vs scalar perf --")
	N := 1 << 20  // 1M elements

	// Setup data once
	a_data := make([]f32, N)
	b_data := make([]f32, N)
	scalar_data := []f32{0.5}
	for i in 0..<N {
		a_data[i] = f32(i) * 0.001
		b_data[i] = f32(i) * 0.0005 + 1
	}
	defer delete(a_data)
	defer delete(b_data)

	// [N] + [1]: SIMD scalar broadcast
	{
		a := ml.from_data_copy(a_data, {i32(N)})
		b := ml.from_data_copy(scalar_data, {1})
		bench_ms("scalar add [N]+[1]  (SIMD path)", 30, proc(a, b: ^ml.Tensor) {
			r := ml.add(a, b); ml.realize(r)
		}, a, b)
	}

	// [N] + [N]: same-shape SIMD add
	{
		a := ml.from_data_copy(a_data, {i32(N)})
		b := ml.from_data_copy(b_data, {i32(N)})
		bench_ms("same-shape add [N]+[N] (SIMD)", 30, proc(a, b: ^ml.Tensor) {
			r := ml.add(a, b); ml.realize(r)
		}, a, b)
	}

	// [M,N] + [N]: row broadcast SIMD
	{
		M, N2 := 1024, 1024
		row := make([]f32, M * N2)
		col_vec := make([]f32, N2)
		for i in 0..<M*N2 do row[i] = f32(i) * 0.001
		for j in 0..<N2 do col_vec[j] = f32(j) * 0.01
		defer delete(row)
		defer delete(col_vec)

		a := ml.from_data_copy(row, {i32(M), i32(N2)})
		b := ml.from_data_copy(col_vec, {i32(N2)})
		bench_ms("row broadcast add [1024,1024]+[1024]", 20, proc(a, b: ^ml.Tensor) {
			r := ml.add(a, b); ml.realize(r)
		}, a, b)
	}

// [M,N] + [M,1]: column broadcast — SIMD path
	{
		M, N2 := 1024, 1024
		big := make([]f32, M * N2)
		col := make([]f32, M)
		for i in 0..<M*N2 do big[i] = f32(i) * 0.001
		for i in 0..<M do col[i] = f32(i) * 0.01
		defer delete(big)
		defer delete(col)

		a := ml.from_data_copy(big, {i32(M), i32(N2)})
		b := ml.from_data_copy(col, {i32(M), 1})
		bench_ms("col broadcast add [1024,1024]+[1024,1] (SIMD)", 20, proc(a, b: ^ml.Tensor) {
			r := ml.add(a, b); ml.realize(r)
		}, a, b)
	}
}

// ---- conv / pool ----------------------------------------------------------

test_conv_pool :: proc() {
	fmt.println("-- conv2d / max_pool2d --")
	// x: 1x1x3x3 identity-ish, w: 1x1x2x2 all ones, stride 1 pad 0 → 2x2 of window sums
	// x = [[1,2,3],[4,5,6],[7,8,9]]
	x := ml.from_data_copy({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3})
	w := ml.from_data_copy({1, 1, 1, 1}, {1, 1, 2, 2})
	y := ml.conv2d(x, w, stride = 1, padding = 0)
	// windows: 1+2+4+5=12, 2+3+5+6=16, 4+5+7+8=24, 5+6+8+9=28
	expect_close(y, ml.from_data_copy({12, 16, 24, 28}, {1, 1, 2, 2}), "conv2d 3x3 k=2")

	// max pool 2x2 stride 2 on 1x1x4x4
	// [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
	p_in := ml.from_data_copy({
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, {1, 1, 4, 4})
	p := ml.max_pool2d(p_in, kernel_size = 2)
	expect_close(p, ml.from_data_copy({6, 8, 14, 16}, {1, 1, 2, 2}), "max_pool2d 2x2")

	flat := ml.flatten(p)
	expect(flat.shape[0] == 1 && flat.shape[1] == 4, "flatten shape")
	ml.realize(flat)
	expect(flat.data[0] == 6 && flat.data[3] == 16, "flatten values")
}

test_conv_grad :: proc() {
	fmt.println("-- conv2d backward --")
	// Tiny: x 1x1x2x2, w 1x1x2x2, out 1x1x1x1 = sum(x*w)
	// L = out, dL/dw = x, dL/dx = w
	x := ml.from_data_copy({1, 2, 3, 4}, {1, 1, 2, 2}, requires_grad = true)
	w := ml.from_data_copy({0.5, 0.5, 0.5, 0.5}, {1, 1, 2, 2}, requires_grad = true)
	y := ml.conv2d(x, w, stride = 1, padding = 0)
	loss := ml.sum(y, -1)
	ml.backward(loss)
	expect_close(w.grad, ml.from_data_copy({1, 2, 3, 4}, {1, 1, 2, 2}), "conv dW = x")
	expect_close(x.grad, ml.from_data_copy({0.5, 0.5, 0.5, 0.5}, {1, 1, 2, 2}), "conv dX = w")

	// maxpool backward: gradient routes to argmax only
	t := ml.from_data_copy({1, 3, 2, 0}, {1, 1, 2, 2}, requires_grad = true)
	m := ml.max_pool2d(t, kernel_size = 2)
	loss2 := ml.sum(m, -1)
	ml.backward(loss2)
	// max is 3 at index 1
	expect_close(t.grad, ml.from_data_copy({0, 1, 0, 0}, {1, 1, 2, 2}), "maxpool grad to argmax")
}

test_params_helper :: proc() {
	fmt.println("-- collect_params --")
	l := ml.linear(4, 2, .Zeros)
	c := ml.conv2d_layer(1, 3, 3, stride = 1, padding = 1, init = .Zeros)
	params: [dynamic]^ml.Tensor
	ml.linear_params(&params, l)
	ml.conv_params(&params, c)
	expect(len(params) == 4, "linear+conv → 4 params")
	expect(params[0] == l.W && params[1] == l.b, "linear params order")
	expect(params[2] == c.W && params[3] == c.b, "conv params order")
}

// Div with broadcast: a[2,3] / b[1] and a[2,3] / b[3]
test_div_broadcast_grad :: proc() {
	fmt.println("-- div broadcast grad --")
	// a / scalar: dL/da = 1/s, dL/ds = -sum(a)/s²  for L=sum(a/s)
	a := ml.from_data_copy({2, 4, 6, 8, 10, 12}, {2, 3}, requires_grad = true)
	s := ml.from_data_copy({2}, {1}, requires_grad = true)
	y := ml.div(a, s)
	loss := ml.sum(y, -1)
	ml.backward(loss)
	// y = [1,2,3,4,5,6], dL/da = 0.5 each, dL/ds = -sum(a)/4 = -42/4 = -10.5
	expect_close(a.grad, ml.from_data_copy({0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, {2, 3}), "div scalar dA")
	expect_close(s.grad, ml.from_data_copy({-10.5}, {1}), "div scalar dS")

	// a / row: b = [1,2,3], a = [[1,2,3],[4,5,6]]
	a2 := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {2, 3}, requires_grad = true)
	b2 := ml.from_data_copy({1, 2, 3}, {3}, requires_grad = true)
	y2 := ml.div(a2, b2)
	loss2 := ml.sum(y2, -1)
	ml.backward(loss2)
	// dL/da = 1/b = [1, 0.5, 1/3] per row
	expect_close(a2.grad, ml.from_data_copy({1, 0.5, 1.0 / 3, 1, 0.5, 1.0 / 3}, {2, 3}), "div row dA")
	// dL/db_j = sum_i (-a_ij / b_j²) = - (a0j+a1j)/b_j²
	// j0: -(1+4)/1 = -5; j1: -(2+5)/4 = -1.75; j2: -(3+6)/9 = -1
	expect_close(b2.grad, ml.from_data_copy({-5, -1.75, -1}, {3}), "div row dB")
}

// Numerical gradient check: L = sum(relu(x @ w + b))
test_gradcheck :: proc() {
	fmt.println("-- numerical gradcheck --")
	eps: f32 = 1e-3
	x_d := [4]f32{0.5, -0.3, 0.8, 0.1}
	w_d := [6]f32{0.2, -0.4, 0.6, 0.1, -0.2, 0.3}
	b_d := [3]f32{0.1, -0.1, 0.05}
	x_sh := []i32{2, 2}
	w_sh := []i32{2, 3}
	b_sh := []i32{3}

	fwd :: proc(xd, wd, bd: []f32, x_sh, w_sh, b_sh: []i32) -> f32 {
		return ml.item(ml.sum(ml.relu(ml.add(
			ml.matmul(ml.from_data_copy(xd, x_sh), ml.from_data_copy(wd, w_sh)),
			ml.from_data_copy(bd, b_sh),
		)), -1))
	}

	// analytic
	x := ml.from_data_copy(x_d[:], x_sh, requires_grad = true)
	w := ml.from_data_copy(w_d[:], w_sh, requires_grad = true)
	b := ml.from_data_copy(b_d[:], b_sh, requires_grad = true)
	ml.backward(ml.sum(ml.relu(ml.add(ml.matmul(x, w), b)), -1))

	// numerical dW
	ok_w := true
	for i in 0..<6 {
		wp := w_d
		wm := w_d
		wp[i] += eps
		wm[i] -= eps
		num := (fwd(x_d[:], wp[:], b_d[:], x_sh, w_sh, b_sh) - fwd(x_d[:], wm[:], b_d[:], x_sh, w_sh, b_sh)) / (2 * eps)
		d := w.grad.data[i] - num
		if d < 0 do d = -d
		if d > 2e-2 do ok_w = false
	}
	expect(ok_w, "gradcheck dW vs numerical")

	// numerical dB
	ok_b := true
	for i in 0..<3 {
		bp := b_d
		bm := b_d
		bp[i] += eps
		bm[i] -= eps
		num := (fwd(x_d[:], w_d[:], bp[:], x_sh, w_sh, b_sh) - fwd(x_d[:], w_d[:], bm[:], x_sh, w_sh, b_sh)) / (2 * eps)
		d := b.grad.data[i] - num
		if d < 0 do d = -d
		if d > 2e-2 do ok_b = false
	}
	expect(ok_b, "gradcheck dB vs numerical")
}

test_cross_entropy :: proc() {
	fmt.println("-- cross_entropy --")
	// logits that put all mass on class 1 after softmax-ish: large on idx 1
	logits := ml.from_data_copy({0, 10, 0, 0, 10, 0}, {2, 3}, requires_grad = true)
	labels := []u8{1, 1}
	loss := ml.cross_entropy(logits, labels)
	// near-zero loss
	v := ml.item(loss)
	expect(v < 0.01, "CE low when correct class dominates")
	ml.backward(loss)
	expect(logits.grad != nil, "CE produces logit grads")
	// grad at correct class should be negative-ish (softmax-1)/B < 0
	expect(logits.grad.data[1] < 0 && logits.grad.data[4] < 0, "CE grad negative on true class")
}

// Reshape/Transpose are views (share storage); matmul densifies as needed.
test_views :: proc() {
	fmt.println("-- views (reshape/transpose) --")
	a := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {2, 3})
	r := ml.reshape(a, {3, 2})
	ml.realize(r)
	expect(raw_data(r.data) == raw_data(a.data), "reshape shares storage")
	expect(ml.is_contiguous(r), "reshape of contig stays contig")
	expect_close(r, ml.from_data_copy({1, 2, 3, 4, 5, 6}, {3, 2}), "reshape values")

	t := ml.transpose(a, 0, 1) // [3,2]
	ml.realize(t)
	expect(raw_data(t.data) == raw_data(a.data), "transpose shares storage")
	expect(!ml.is_contiguous(t), "transpose is non-contig")
	// dense values of T: [[1,4],[2,5],[3,6]]
	expect_close(t, ml.from_data_copy({1, 4, 2, 5, 3, 6}, {3, 2}), "transpose values")

	// matmul with transposed right factor densifies under the hood
	// a [2,3] @ t[3,2] but t is view of a^T... a @ a.T → [2,2]
	aT := ml.T(a)
	y := ml.matmul(a, aT)
	// a @ a.T = [[1+4+9, 4+10+18],[4+10+18, 16+25+36]] = [[14,32],[32,77]]
	expect_close(y, ml.from_data_copy({14, 32, 32, 77}, {2, 2}), "matmul with T view")

	// grad through transpose view
	x := ml.from_data_copy({1, 2, 3, 4}, {2, 2}, requires_grad = true)
	xt := ml.T(x)
	loss := ml.sum(xt, -1)
	ml.backward(loss)
	// dL/dx = ones (transpose of ones)
	expect_close(x.grad, ml.from_data_copy({1, 1, 1, 1}, {2, 2}), "grad through T")
}