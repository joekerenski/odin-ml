package main

// Odin native `matrix` type vs our explicit-#simd kernels — does the language
// give us SIMD for free?
//
// Three findings to verify:
//   1. Native `matrix[M,N]T` is capped at 64 elements (matrix[8,8]f32 is the
//      largest square) → it is for small linear algebra, NOT tensors.
//   2. `a + b` on a native matrix compiles to NEON `fadd.4s` (LLVM
//      auto-vectorizes) — see study_mat_* in study.s / study.ll.
//   3. A *plain scalar loop* also compiles to NEON — see study_plain_* and the
//      bench below, where plain loops tie our hand-written #simd kernels.
//
// Run:    odin run . -o:speed -no-bounds-check -disable-assert
// Inspect: sh build_asm.sh   (writes study.s / study_ir/, greps study procs)

import "core:fmt"
import "core:math/linalg"
import "core:time"
import ml "../../ml"

// Largest allowed square matrix: 8x8 = 64 elements.
Mat :: matrix[8, 8]f32

// ---- the three ways to write the same elementwise op ----------------------

// 1. native matrix ops (what the language gives us)
mat_add :: proc(out, a, b: ^Mat) {
	out^ = a^ + b^ // elementwise +
}
mat_hadamard :: proc(out, a, b: ^Mat) {
	out^ = linalg.hadamard_product(a^, b^) // elementwise * (`*` is matrix product)
}

// 2. plain scalar loop — no #simd, no intrinsics, just a for loop
plain_add :: proc(out, a, b: []f32) {
	assert(len(out) == len(a) && len(a) == len(b))
	for i in 0..<len(out) do out[i] = a[i] + b[i]
}
plain_mul :: proc(out, a, b: []f32) {
	assert(len(out) == len(a) && len(a) == len(b))
	for i in 0..<len(out) do out[i] = a[i] * b[i]
}

// 3. our kernels: explicit f32x4 SIMD
slice_add :: proc(out, a, b: []f32) {
	ml.add_f32_contiguous(out, a, b)
}
slice_mul :: proc(out, a, b: []f32) {
	ml.mul_f32_contiguous(out, a, b)
}

// ---- asm / IR study procs (grep these out of study.s / study_ir/.ll) ------
// @(export) keeps them as standalone symbols (no inlining / dead-stripping).

@(export)
study_mat_add :: proc(a, b: ^Mat) -> Mat {
	return a^ + b^
}
@(export)
study_mat_mul :: proc(a, b: ^Mat) -> Mat {
	return linalg.hadamard_product(a^, b^)
}
@(export)
study_plain_add :: proc(out, a, b: []f32) {
	for i in 0..<len(out) do out[i] = a[i] + b[i]
}
@(export)
study_plain_mul :: proc(out, a, b: []f32) {
	for i in 0..<len(out) do out[i] = a[i] * b[i]
}
@(export)
study_slice_add :: proc(out, a, b: []f32) {
	ml.add_f32_contiguous(out, a, b)
}
@(export)
study_slice_mul :: proc(out, a, b: []f32) {
	ml.mul_f32_contiguous(out, a, b)
}

// ---- helpers -------------------------------------------------------------

fill_rand :: proc(buf: []f32, seed: u64) {
	s := seed
	for i in 0..<len(buf) {
		s = s * 6364136223846793005 + 1
		u := f32((s >> 40) & 0xffffff) / f32(0xffffff)
		buf[i] = u * 2 - 1
	}
}

measure_ms :: proc(reps: int, op: proc([]f32, []f32, []f32), a, b, c: []f32, sink: ^f32) -> f64 {
	for _ in 0..<200 do op(a, b, c) // warmup
	t0 := time.tick_now()
	for _ in 0..<reps {
		op(a, b, c)
		sink^ += c[0]
		a[0] = c[0]
	}
	dt := time.duration_seconds(time.tick_since(t0))
	return dt / f64(reps) * 1e3
}

// ---- correctness: all three produce the same result -----------------------

check :: proc() {
	a, b, c_mat, c_slice: Mat
	fill_rand((^[64]f32)(&a)[:], 1)
	fill_rand((^[64]f32)(&b)[:], 2)

	// native matrix vs our flattened kernel
	mat_add(&c_mat, &a, &b)
	a_sl := (^[64]f32)(&a)[:]
	b_sl := (^[64]f32)(&b)[:]
	c_sl := (^[64]f32)(&c_slice)[:]
	ml.add_f32_contiguous(c_sl, a_sl, b_sl)
	ok := true
	for i in 0..<len(a_sl) do if c_sl[i] != (^[64]f32)(&c_mat)[i] do ok = false
	fmt.println("matrix +  == #simd slice add: ", ok)

	mat_hadamard(&c_mat, &a, &b)
	ml.mul_f32_contiguous(c_sl, a_sl, b_sl)
	ok = true
	for i in 0..<len(a_sl) do if c_sl[i] != (^[64]f32)(&c_mat)[i] do ok = false
	fmt.println("hadamard == #simd slice mul: ", ok)
}

// ---- bench: plain scalar loop vs our #simd kernel, streaming size ---------

main :: proc() {
	fmt.println("odin-ml scalar-vs-#simd bench (f32)")
	fmt.println("flags: -o:speed -no-bounds-check -disable-assert")
	check()

	n := 1_048_576 // 4 MB buffers — memory-bandwidth bound, same as bench suite
	a := make([]f32, n)
	b := make([]f32, n)
	c := make([]f32, n)
	defer delete(a)
	defer delete(b)
	defer delete(c)
	fill_rand(a, 3)
	fill_rand(b, 4)

	reps := 60
	sink: f32
	gbps := proc(ms: f64, n: int) -> f64 { return (3.0 * f64(n) * 4.0) / (ms * 1e-3) / 1e9 }

	fmt.println("--- plain scalar loop (no #simd) ---")
	ms := measure_ms(reps, plain_add, a, b, c, &sink)
	fmt.printfln("plain add   %8.3f ms  %7.1f GB/s", ms, gbps(ms, n))
	ms = measure_ms(reps, plain_mul, a, b, c, &sink)
	fmt.printfln("plain mul   %8.3f ms  %7.1f GB/s", ms, gbps(ms, n))

	fmt.println("--- our #simd kernels ---")
	ms = measure_ms(reps, slice_add, a, b, c, &sink)
	fmt.printfln("#simd add   %8.3f ms  %7.1f GB/s", ms, gbps(ms, n))
	ms = measure_ms(reps, slice_mul, a, b, c, &sink)
	fmt.printfln("#simd mul   %8.3f ms  %7.1f GB/s", ms, gbps(ms, n))

	fmt.println("--- for reference: NumPy f32 (same 3-array pattern) ---")
	fmt.println("    add n=1048576 ~ 60 GB/s, mul ~ 95 GB/s (bench suite)")
	fmt.printfln("sink=%.3f (keep study procs referenced)", sink)
}
