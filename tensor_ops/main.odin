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