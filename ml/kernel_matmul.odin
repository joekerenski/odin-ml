package ml

// ============================================================================
// matmul_f32 — dense contiguous GEMM: C[M,N] = A[M,K] @ B[K,N]
//
// Backends:
//   - .Accelerate on Darwin (links Apple BLAS, same stack as NumPy) → best perf
//   - .AVX2          x86-64 with AVX2+FMA (kernel_gemm_amd64.odin): packed, all cores
//   - .Pure          portable tiled SIMD (NEON via #simd) for learning / other CPUs
//
// Select with `matmul_set_backend`. Default: Accelerate on Darwin, AVX2 on
// x86-64 CPUs that have it, Pure elsewhere.
// ============================================================================

import "base:intrinsics"
import "core:simd"

Matmul_Backend :: enum {
	Pure,
	Accelerate,
	AVX2,
}

matmul_backend: Matmul_Backend = .Pure

@(init)
matmul_init_backend :: proc "contextless" () {
	// Prefer system BLAS on Apple Silicon/macOS when available.
	when ODIN_OS == .Darwin {
		matmul_backend = .Accelerate
	} else when ODIN_ARCH == .amd64 {
		matmul_backend = gemm_x86_available ? .AVX2 : .Pure
	} else {
		matmul_backend = .Pure
	}
}

matmul_set_backend :: proc(b: Matmul_Backend) {
	when ODIN_ARCH == .amd64 {
		if b == .AVX2 && !gemm_x86_available do return
	} else {
		if b == .AVX2 do return
	}
	matmul_backend = b
}

matmul_get_backend :: proc() -> Matmul_Backend {
	return matmul_backend
}

// --- public entry ----------------------------------------------------------

// C[M,N] = op(A) @ op(B), C overwritten. op(X) = X^T when trans_*:
// A stored [M,K] (or [K,M] if trans_a), B stored [K,N] (or [N,K] if trans_b).
// Uses all cores unless called from inside a parallel_for.
matmul_f32 :: proc(C, A, B: []f32, M, K, N: i32, trans_a := false, trans_b := false) {
	assert(len(A) >= int(M * K))
	assert(len(B) >= int(K * N))
	assert(len(C) >= int(M * N))
	switch matmul_backend {
	case .Accelerate:
		when ODIN_OS == .Darwin {
			accelerate_sgemm(C, A, B, M, K, N, trans_a, trans_b)
			return
		}
	case .AVX2:
		when ODIN_ARCH == .amd64 {
			gemm_x86(C, A, B, int(M), int(K), int(N), trans_a, trans_b, parallel = !in_parallel_for)
			return
		}
	case .Pure:
	}
	a, b := A, B
	if trans_a {
		a = make([]f32, M * K, scratch())
		permute_kernel(a, A, {K, M}, {1, 0})
	}
	if trans_b {
		b = make([]f32, K * N, scratch())
		permute_kernel(b, B, {N, K}, {1, 0})
	}
	for i in 0..<int(M * N) do C[i] = 0
	matmul_f32_accum_pure(C, a, b, M, K, N)
	if trans_a do delete(a, scratch())
	if trans_b do delete(b, scratch())
}

// ============================================================================
// Pure tiled SIMD backend
// ============================================================================

matmul_f32_accum_pure :: proc(C, A, B: []f32, M, K, N: i32) {
	// Bigger tiles help L2; MR microkernel (4 rows x 8 cols of f32 / 2× f32x4).
	BM :: 128
	BN :: 128
	BK :: 64

	m := int(M)
	k := int(K)
	n := int(N)

	for i0 := 0; i0 < m; i0 += BM {
		i_max := min(i0 + BM, m)
		for j0 := 0; j0 < n; j0 += BN {
			j_max := min(j0 + BN, n)
			for p0 := 0; p0 < k; p0 += BK {
				p_max := min(p0 + BK, k)
				matmul_tile_f32_mr4(C, A, B, m, k, n, i0, i_max, j0, j_max, p0, p_max)
			}
		}
	}
}

// 4-row microkernel over the panel, 8-wide SIMD (2× f32x4) on N.
matmul_tile_f32_mr4 :: proc(
	C, A, B: []f32,
	m, k, n: int,
	i0, i_max, j0, j_max, p0, p_max: int,
) {
	i := i0
	for ; i + 4 <= i_max; i += 4 {
		j := j0
		for ; j + 8 <= j_max; j += 8 {
			// two f32x4 accumulators per row → 8 columns
			c00 := intrinsics.unaligned_load((^simd.f32x4)(&C[(i + 0) * n + j + 0]))
			c01 := intrinsics.unaligned_load((^simd.f32x4)(&C[(i + 0) * n + j + 4]))
			c10 := intrinsics.unaligned_load((^simd.f32x4)(&C[(i + 1) * n + j + 0]))
			c11 := intrinsics.unaligned_load((^simd.f32x4)(&C[(i + 1) * n + j + 4]))
			c20 := intrinsics.unaligned_load((^simd.f32x4)(&C[(i + 2) * n + j + 0]))
			c21 := intrinsics.unaligned_load((^simd.f32x4)(&C[(i + 2) * n + j + 4]))
			c30 := intrinsics.unaligned_load((^simd.f32x4)(&C[(i + 3) * n + j + 0]))
			c31 := intrinsics.unaligned_load((^simd.f32x4)(&C[(i + 3) * n + j + 4]))

			for p := p0; p < p_max; p += 1 {
				b0 := intrinsics.unaligned_load((^simd.f32x4)(&B[p * n + j + 0]))
				b1 := intrinsics.unaligned_load((^simd.f32x4)(&B[p * n + j + 4]))
				a0: simd.f32x4 = A[(i + 0) * k + p]
				a1: simd.f32x4 = A[(i + 1) * k + p]
				a2: simd.f32x4 = A[(i + 2) * k + p]
				a3: simd.f32x4 = A[(i + 3) * k + p]
				c00 = fma4(a0, b0, c00)
				c01 = fma4(a0, b1, c01)
				c10 = fma4(a1, b0, c10)
				c11 = fma4(a1, b1, c11)
				c20 = fma4(a2, b0, c20)
				c21 = fma4(a2, b1, c21)
				c30 = fma4(a3, b0, c30)
				c31 = fma4(a3, b1, c31)
			}

			intrinsics.unaligned_store((^simd.f32x4)(&C[(i + 0) * n + j + 0]), c00)
			intrinsics.unaligned_store((^simd.f32x4)(&C[(i + 0) * n + j + 4]), c01)
			intrinsics.unaligned_store((^simd.f32x4)(&C[(i + 1) * n + j + 0]), c10)
			intrinsics.unaligned_store((^simd.f32x4)(&C[(i + 1) * n + j + 4]), c11)
			intrinsics.unaligned_store((^simd.f32x4)(&C[(i + 2) * n + j + 0]), c20)
			intrinsics.unaligned_store((^simd.f32x4)(&C[(i + 2) * n + j + 4]), c21)
			intrinsics.unaligned_store((^simd.f32x4)(&C[(i + 3) * n + j + 0]), c30)
			intrinsics.unaligned_store((^simd.f32x4)(&C[(i + 3) * n + j + 4]), c31)
		}
		// remaining columns for these 4 rows
		for ; j < j_max; j += 1 {
			s0 := C[(i + 0) * n + j]
			s1 := C[(i + 1) * n + j]
			s2 := C[(i + 2) * n + j]
			s3 := C[(i + 3) * n + j]
			for p := p0; p < p_max; p += 1 {
				bp := B[p * n + j]
				s0 += A[(i + 0) * k + p] * bp
				s1 += A[(i + 1) * k + p] * bp
				s2 += A[(i + 2) * k + p] * bp
				s3 += A[(i + 3) * k + p] * bp
			}
			C[(i + 0) * n + j] = s0
			C[(i + 1) * n + j] = s1
			C[(i + 2) * n + j] = s2
			C[(i + 3) * n + j] = s3
		}
	}
	// leftover rows (1..3)
	for ; i < i_max; i += 1 {
		j := j0
		for ; j + 4 <= j_max; j += 4 {
			acc := intrinsics.unaligned_load((^simd.f32x4)(&C[i * n + j]))
			for p := p0; p < p_max; p += 1 {
				av: simd.f32x4 = A[i * k + p]
				bv := intrinsics.unaligned_load((^simd.f32x4)(&B[p * n + j]))
				acc = fma4(av, bv, acc)
			}
			intrinsics.unaligned_store((^simd.f32x4)(&C[i * n + j]), acc)
		}
		for ; j < j_max; j += 1 {
			s := C[i * n + j]
			for p := p0; p < p_max; p += 1 {
				s += A[i * k + p] * B[p * n + j]
			}
			C[i * n + j] = s
		}
	}
}


// FMA where the baseline ISA has it (arm64); x86-64 without -microarch would
// lower fused_mul_add to a libm call per lane, so multiply then add there.
@(private = "file")
fma4 :: #force_inline proc "contextless" (a, b, c: simd.f32x4) -> simd.f32x4 {
	when ODIN_ARCH == .amd64 do return a * b + c
	else do return intrinsics.fused_mul_add(a, b, c)
}

// Batched GEMM: C[i] = op(A[i]) @ op(B[i]) for i in 0..<batch.
// (Accelerate is fast even for attention-sized 5×8 matrices; a hand-written
// loop kernel was 2× slower.)
Bmm_Job :: struct {
	C, A, B:          []f32,
	M, K, N:          i32,
	trans_a, trans_b: bool,
}

matmul_batched :: proc(C, A, B: []f32, batch: int, M, K, N: i32, trans_a, trans_b: bool) {
	// Few big matrices: one at a time, each on all cores. Many: one per thread.
	if batch == 1 || (batch < 2 * thread_count() && int(M) * int(K) * int(N) >= 1 << 20) {
		for i in 0 ..< batch {
			matmul_f32(C[i * int(M * N):], A[i * int(M * K):], B[i * int(K * N):], M, K, N, trans_a, trans_b)
		}
		return
	}
	job := Bmm_Job{C, A, B, M, K, N, trans_a, trans_b}
	when ODIN_ARCH == .amd64 {
		if matmul_backend == .AVX2 && int(M) * int(K) * int(N) <= 16 * 16 * 64 {
			parallel_for(batch, max(1, PAR_GRAIN / int(M * K * N)), proc(data: rawptr, lo, hi: int) {
				j := (^Bmm_Job)(data)
				gemm_x86_batched_small(j.C, j.A, j.B, lo, hi, int(j.M), int(j.K), int(j.N), j.trans_a, j.trans_b)
			}, &job)
			return
		}
	}
	parallel_for(batch, max(1, PAR_GRAIN / int(M * K * N)), proc(data: rawptr, lo, hi: int) {
		using j := (^Bmm_Job)(data)
		for i in lo ..< hi {
			matmul_f32(C[i * int(M * N):], A[i * int(M * K):], B[i * int(K * N):], M, K, N, trans_a, trans_b)
		}
	}, &job)
}

// Backend.matmul on the CPU. The dense batched layout goes to matmul_batched;
// strided operands (views read in place) run item by item: through BLAS
// leading dims (Accelerate), else packed into dense copies.
cpu_gemm :: proc(g: ^Gemm) {
	ta, lda, aok := gemm_layout(g.a, g.M, g.K)
	tb, ldb, bok := gemm_layout(g.b, g.K, g.N)
	tc, ldc, cok := gemm_layout(g.c, g.M, g.N)
	if aok && bok && cok && !tc && lda == (ta ? g.M : g.K) && ldb == (tb ? g.K : g.N) && ldc == g.N &&
	   gemm_batch_flat(g, g.a, g.M, g.K) && gemm_batch_flat(g, g.b, g.K, g.N) && gemm_batch_flat(g, g.c, g.M, g.N) {
		matmul_batched(g.c.data, g.a.data, g.b.data, g.Z0 * g.Z1, i32(g.M), i32(g.K), i32(g.N), ta, tb)
		return
	}
	Z := g.Z0 * g.Z1
	if Z == 1 {
		gemm_item(g, 0)
		return
	}
	parallel_for(Z, max(1, PAR_GRAIN / (g.M * g.K * g.N)), proc(data: rawptr, lo, hi: int) {
		g := (^Gemm)(data)
		for z in lo ..< hi do gemm_item(g, z)
	}, g)
}

// One batch item of a strided GEMM.
@(private = "file")
gemm_item :: proc(g: ^Gemm, z: int) {
	M, K, N := g.M, g.K, g.N
	a := g.a.data[gemm_offset(g, g.a, z):]
	b := g.b.data[gemm_offset(g, g.b, z):]
	c := g.c.data[gemm_offset(g, g.c, z):]
	ta, lda, aok := gemm_layout(g.a, M, K)
	tb, ldb, bok := gemm_layout(g.b, K, N)
	tc, ldc, cok := gemm_layout(g.c, M, N)
	when ODIN_OS == .Darwin {
		if matmul_backend == .Accelerate && aok && bok && cok {
			op :: proc(t: bool) -> i32 { return t ? CblasTrans : CblasNoTrans }
			if !tc {
				cblas_sgemm(CblasRowMajor, op(ta), op(tb), i32(M), i32(N), i32(K), 1, raw_data(a), i32(lda), raw_data(b), i32(ldb), 0, raw_data(c), i32(ldc))
			} else { // C column-major: Cᵀ = Bᵀ·Aᵀ, row-major
				cblas_sgemm(CblasRowMajor, op(!tb), op(!ta), i32(N), i32(M), i32(K), 1, raw_data(b), i32(ldb), raw_data(a), i32(lda), 0, raw_data(c), i32(ldc))
			}
			return
		}
	}
	// operands that aren't plain dense matrices go through dense copies
	plain :: proc(t: bool, ld, R, C: int, ok: bool) -> bool { return ok && ld == (t ? R : C) }
	pa, pb := a, b
	if !plain(ta, lda, M, K, aok) {
		pa = make([]f32, M * K, scratch())
		strided_copy(pa, a, {M, K}, {g.a.rs, g.a.cs})
		ta = false
	}
	if !plain(tb, ldb, K, N, bok) {
		pb = make([]f32, K * N, scratch())
		strided_copy(pb, b, {K, N}, {g.b.rs, g.b.cs})
		tb = false
	}
	direct := plain(tc, ldc, M, N, cok) && !tc
	pc := direct ? c : make([]f32, M * N, scratch())
	matmul_f32(pc, pa, pb, i32(M), i32(K), i32(N), ta, tb)
	if !direct {
		for i in 0 ..< M do for j in 0 ..< N do c[i * g.c.rs + j * g.c.cs] = pc[i * N + j]
		delete(pc, scratch())
	}
	if raw_data(pa) != raw_data(a) do delete(pa, scratch())
	if raw_data(pb) != raw_data(b) do delete(pb, scratch())
}
