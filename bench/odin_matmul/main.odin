package main

// Pure-kernel matmul / elemwise bench vs NumPy.
//   cd bench && odin run odin_matmul -o:speed -no-bounds-check -disable-assert

import "core:fmt"
import "core:time"
import ml "../../ml"

Main_Config :: struct {
	m, k, n: i32,
}

main :: proc() {
	fmt.println("odin-ml kernel bench (f32)")
	fmt.println("flags: -o:speed -no-bounds-check -disable-assert")

	configs := []Main_Config{
		{256, 256, 256},
		{512, 512, 512},
		{1024, 1024, 1024},
		{2048, 2048, 2048},
		{64, 1024, 64},
		{1024, 64, 1024},
	}

	// elementwise first (doesn't thrash big GEMM caches / heat as hard)
	fmt.println("--- elementwise (pure SIMD) ---")
	ns := []int{1_048_576, 4_194_304, 16_777_216}
	ops := []string{"add", "mul"}
	for n in ns {
		for op in ops {
			// more samples for small n (noise + frequency scaling)
			reps := 80 if n <= 1_048_576 else 40
			r := bench_elem(n, op, reps, 20)
			fmt.printfln(
				"%s n=%d  best=%.3f ms  med=%.3f ms  %.1f GB/s",
				op, n, r.best_ms, r.med_ms, r.gbps,
			)
		}
	}

	// --- default backend (Accelerate on Darwin) ---
	fmt.printfln("--- backend: %v (default) ---", ml.matmul_get_backend())
	run_matmul_suite(configs)

	// --- pure tiled SIMD last (educational; heats CPU hard) ---
	ml.matmul_set_backend(.Pure)
	fmt.printfln("--- backend: %v ---", ml.matmul_get_backend())
	run_matmul_suite(configs)

	// restore default for library demos
	ml.matmul_init_backend()
}

run_matmul_suite :: proc(configs: []Main_Config) {
	for conf in configs {
		m, k, n := conf.m, conf.k, conf.n
		// batch multiple GEMMs per sample when the op is small, so timers breathe
		batch: int = 1
		flops := 2.0 * f64(m) * f64(k) * f64(n)
		// aim for ~5+ ms of work per sample
		for flops * f64(batch) / 1e9 < 8.0 do batch *= 2 // rough: <8 GFLOP total → double
		if batch > 64 do batch = 64
		r := bench_matmul(m, k, n, 15, 8, batch)
		fmt.printfln(
			"matmul %dx%d@%dx%d  best=%.3f ms  med=%.3f ms  %.1f GFLOP/s  |C[0]|=%.4f  (batch=%d)",
			m, k, k, n, r.best_ms, r.med_ms, r.gflops, r.checksum, batch,
		)
	}
}

Bench_Result :: struct {
	best_ms:  f64,
	med_ms:   f64,
	gflops:   f64,
	gbps:     f64,
	checksum: f32,
}

fill_rand :: proc(buf: []f32, seed: u64) {
	s := seed
	for i in 0..<len(buf) {
		s = s * 6364136223846793005 + 1
		u := f32((s >> 40) & 0xffffff) / f32(0xffffff)
		buf[i] = u * 2 - 1
	}
}

bench_matmul :: proc(M, K, N: i32, repeats: int, warmup: int, batch: int = 1) -> Bench_Result {
	a := make([]f32, M * K)
	b := make([]f32, K * N)
	c := make([]f32, M * N)
	defer delete(a)
	defer delete(b)
	defer delete(c)
	fill_rand(a, 1)
	fill_rand(b, 2)

	for _ in 0..<warmup {
		for _ in 0..<batch do ml.matmul_f32(c, a, b, M, K, N)
	}
	sink := c[0]

	times := make([]f64, repeats)
	defer delete(times)
	for r in 0..<repeats {
		t0 := time.tick_now()
		for _ in 0..<batch do ml.matmul_f32(c, a, b, M, K, N)
		dt := time.duration_seconds(time.tick_since(t0))
		// time per single matmul
		times[r] = dt / f64(batch)
		sink += c[0]
	}

	sort_f64(times)
	best := times[0]
	med := times[len(times) / 2]
	// gflops from *median* (stable)
	gflops := (2.0 * f64(M) * f64(K) * f64(N)) / med / 1e9
	return Bench_Result{
		best_ms = best * 1e3,
		med_ms = med * 1e3,
		gflops = gflops,
		checksum = sink,
	}
}

bench_elem :: proc(n: int, op: string, repeats: int, warmup: int) -> Bench_Result {
	a := make([]f32, n)
	b := make([]f32, n)
	out := make([]f32, n)
	defer delete(a)
	defer delete(b)
	defer delete(out)
	fill_rand(a, 3)
	fill_rand(b, 4)

	run :: proc(op: string, out, a, b: []f32) {
		switch op {
		case "add":
			ml.add_f32_contiguous(out, a, b)
		case "mul":
			ml.mul_f32_contiguous(out, a, b)
		}
	}

	for _ in 0..<warmup do run(op, out, a, b)
	sink := out[0]

	times := make([]f64, repeats)
	defer delete(times)
	for r in 0..<repeats {
		t0 := time.tick_now()
		run(op, out, a, b)
		dt := time.duration_seconds(time.tick_since(t0))
		times[r] = dt
		sink += out[0]
	}
	sort_f64(times)
	best := times[0]
	med := times[len(times) / 2]
	gbps := (3.0 * f64(n) * 4.0) / med / 1e9
	return Bench_Result{
		best_ms = best * 1e3,
		med_ms = med * 1e3,
		gbps = gbps,
		checksum = sink,
	}
}

sort_f64 :: proc(times: []f64) {
	for i in 1..<len(times) {
		v := times[i]
		j := i
		for j > 0 && times[j - 1] > v {
			times[j] = times[j - 1]
			j -= 1
		}
		times[j] = v
	}
}