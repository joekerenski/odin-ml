package ml

// ============================================================================
// matmul_f32 — dense contiguous GEMM: C[M,N] = A[M,K] @ B[K,N]
//
// Backends:
//   - .Accelerate on Darwin (links Apple BLAS, same stack as NumPy) → best perf
//   - .Pure          portable tiled SIMD (NEON via #simd) for learning / other OS
//
// Select with `matmul_set_backend`. Default: Accelerate on Darwin, Pure elsewhere.
// ============================================================================

import "base:intrinsics"
import "core:simd"

Matmul_Backend :: enum {
	Pure,
	Accelerate,
}

matmul_backend: Matmul_Backend = .Pure

@(init)
matmul_init_backend :: proc "contextless" () {
	// Prefer system BLAS on Apple Silicon/macOS when available.
	when ODIN_OS == .Darwin {
		matmul_backend = .Accelerate
	} else {
		matmul_backend = .Pure
	}
}

matmul_set_backend :: proc(b: Matmul_Backend) {
	matmul_backend = b
}

matmul_get_backend :: proc() -> Matmul_Backend {
	return matmul_backend
}

// --- public entry ----------------------------------------------------------

// C = A @ B with C zeroed first. All row-major contiguous.
matmul_f32 :: proc(C, A, B: []f32, M, K, N: i32) {
	assert(len(A) >= int(M * K))
	assert(len(B) >= int(K * N))
	assert(len(C) >= int(M * N))
	switch matmul_backend {
	case .Accelerate:
		when ODIN_OS == .Darwin {
			// C := 1*A*B + 0*C
			accelerate_sgemm(C, A, B, M, K, N, 1, 0)
			return
		} else {
			// fall through
		}
	case .Pure:
	}
	for i in 0..<int(M * N) do C[i] = 0
	matmul_f32_accum_pure(C, A, B, M, K, N)
}

// C += A @ B
matmul_f32_accum :: proc(C, A, B: []f32, M, K, N: i32) {
	switch matmul_backend {
	case .Accelerate:
		when ODIN_OS == .Darwin {
			accelerate_sgemm(C, A, B, M, K, N, 1, 1)
			return
		}
	case .Pure:
	}
	matmul_f32_accum_pure(C, A, B, M, K, N)
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
				c00 = intrinsics.fused_mul_add(a0, b0, c00)
				c01 = intrinsics.fused_mul_add(a0, b1, c01)
				c10 = intrinsics.fused_mul_add(a1, b0, c10)
				c11 = intrinsics.fused_mul_add(a1, b1, c11)
				c20 = intrinsics.fused_mul_add(a2, b0, c20)
				c21 = intrinsics.fused_mul_add(a2, b1, c21)
				c30 = intrinsics.fused_mul_add(a3, b0, c30)
				c31 = intrinsics.fused_mul_add(a3, b1, c31)
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
				acc = intrinsics.fused_mul_add(av, bv, acc)
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

// Naive triple loop — reference.
matmul_f32_naive :: proc(C, A, B: []f32, M, K, N: i32) {
	for i in 0..<M {
		for j in 0..<N {
			s: f32 = 0
			for p in 0..<K do s += A[i * K + p] * B[p * N + j]
			C[i * N + j] = s
		}
	}
}

