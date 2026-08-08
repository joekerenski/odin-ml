package main

// Broadcast bench vs NumPy — runs the exact same patterns as numpy_bench_bcast.py.
//
//   cd bench && odin run odin_matmul_bcast -o:speed -no-bounds-check -disable-assert
//
// Output lines match numpy_bench_bcast.py format so a compare script can grep both.

import "core:fmt"
import "core:time"
import ml "../../ml"

// Package-level globals so body_proc can reference them without capturing.
PAT :: enum {
	SameN,
	Scalar1,
	Row1024,
	Col1024,
	Scalar2D,
	Row3D,
}

bench_tensors :: struct {
	a:        ^ml.Tensor,
	b_:       ^ml.Tensor,
	c:        ^ml.Tensor,
	big:      ^ml.Tensor,
	row:      ^ml.Tensor,
	col:      ^ml.Tensor,
	scalar2d: ^ml.Tensor,
	big3:     ^ml.Tensor,
	row3:     ^ml.Tensor,
}

g_tensors: bench_tensors
g_which: PAT = .SameN

body_proc :: proc() {
	switch g_which {
	case .SameN:    r := ml.add(g_tensors.a, g_tensors.b_); ml.realize(r)
	case .Scalar1:  r := ml.add(g_tensors.a, g_tensors.c); ml.realize(r)
	case .Row1024:  r := ml.add(g_tensors.big, g_tensors.row); ml.realize(r)
	case .Col1024:  r := ml.add(g_tensors.big, g_tensors.col); ml.realize(r)
	case .Scalar2D: r := ml.add(g_tensors.big, g_tensors.scalar2d); ml.realize(r)
	case .Row3D:    r := ml.add(g_tensors.big3, g_tensors.row3); ml.realize(r)
	}
}

main :: proc() {
	fmt.println("odin-ml broadcast bench (f32)")
	fmt.println("flags: -o:speed -no-bounds-check -disable-assert")
	fmt.println()

	// Match numpy_bench_bcast.py exactly
	N :: 1 << 20  // 1M elements
	M :: 1024
	K :: 1024
	B :: 32

	a := make([]f32, N)
	b := make([]f32, N)
	c := make([]f32, 1)
	defer delete(a); defer delete(b); defer delete(c)
	fill_rand(a, 0); fill_rand(b, 1); fill_rand(c, 2)

	big := make([]f32, M * K)
	row := make([]f32, K)
	col := make([]f32, M)
	scalar_2d := make([]f32, 1)
	defer delete(big); defer delete(row); defer delete(col); defer delete(scalar_2d)
	fill_rand(big, 3); fill_rand(row, 4); fill_rand(col, 5); fill_rand(scalar_2d, 6)

	big3 := make([]f32, B * M * K)
	row3 := make([]f32, K)
	defer delete(big3); defer delete(row3)
	fill_rand(big3, 7); fill_rand(row3, 8)

	A := ml.from_data_copy(a, {i32(N)})
	B_ := ml.from_data_copy(b, {i32(N)})
	C := ml.from_data_copy(c, {1})

	BIG := ml.from_data_copy(big, {i32(M), i32(K)})
	ROW := ml.from_data_copy(row, {i32(K)})
	COL := ml.from_data_copy(col, {i32(M), 1})
	SCALAR2D := ml.from_data_copy(scalar_2d, {1, 1})

	BIG3 := ml.from_data_copy(big3, {i32(B), i32(M), i32(K)})
	ROW3 := ml.from_data_copy(row3, {1, i32(K)})

// Populate package-level tensors from these locals (body_proc references them).
	g_tensors = bench_tensors{A, B_, C, BIG, ROW, COL, SCALAR2D, BIG3, ROW3}

	// Run a single pattern. Sets g_which, runs body_proc `batch` times per sample.
	run_one :: proc(label: string, n_elems: int, pat: PAT) {
		g_which = pat

		// measure one body call to estimate batch
		t0 := time.tick_now()
		body_proc()
		one_us := int(time.duration_microseconds(time.tick_since(t0)))
		batch := 8 * 1000 / max(one_us, 1)
		if batch < 1 do batch = 1
		if batch > 256 do batch = 256

		// warmup
		for _ in 0..<5 do body_proc()

		times := make([]f64, 30)
		defer delete(times)
		for i in 0..<30 {
			t1 := time.tick_now()
			for _ in 0..<batch do body_proc()
			dt := time.duration_seconds(time.tick_since(t1)) / f64(batch)
			times[i] = dt * 1e3
		}
		// sort
		for i in 1..<len(times) {
			v := times[i]
			j := i
			for j > 0 && times[j - 1] > v {
				times[j] = times[j - 1]
				j -= 1
			}
			times[j] = v
		}
		med := times[len(times) / 2]
		best := times[0]
		gbps := (3.0 * f64(n_elems) * 4.0) / med / 1e6
		fmt.printfln(
			"  %-40s  best=%.3f ms  med=%.3f ms  %.1f GB/s",
			label, best, med, gbps,
		)
	}

	run_one("add [1048576]+[1048576]",    N,         .SameN)
	run_one("add [1048576]+[1]",          N,         .Scalar1)
	run_one("add [1024,1024]+[1024]",     M * K,     .Row1024)
	run_one("add [1024,1024]+[1024,1]",   M * K,     .Col1024)
	run_one("add [1024,1024]+[1,1]",      M * K,     .Scalar2D)
	run_one("add [32,1024,1024]+[1,1024]", B * M * K, .Row3D)

	fmt.println("done.")
}

fill_rand :: proc(buf: []f32, seed: u64) {
	s := seed
	for i in 0..<len(buf) {
		s = s * 6364136223846793005 + 1
		buf[i] = f32((s >> 40) & 0xffffff) / f32(0xffffff) * 2 - 1
	}
}