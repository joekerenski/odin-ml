package ml

// ============================================================================
// Apple Accelerate CBLAS bindings (Darwin only).
// Same library NumPy uses for matmul on macOS.
// ============================================================================

when ODIN_OS == .Darwin {

	// cblas.h enums (partial)
	CblasRowMajor :: 101
	CblasNoTrans  :: 111

	foreign import accelerate "system:Accelerate.framework"

	foreign accelerate {
		cblas_sgemm :: proc "c" (
			Order: i32,
			TransA: i32,
			TransB: i32,
			M: i32,
			N: i32,
			K: i32,
			alpha: f32,
			A: [^]f32,
			lda: i32,
			B: [^]f32,
			ldb: i32,
			beta: f32,
			C: [^]f32,
			ldc: i32,
		) ---
	}

	// C (M×N) = alpha * A(M×K) @ B(K×N) + beta * C
	// All row-major contiguous.
	accelerate_sgemm :: proc(C, A, B: []f32, M, K, N: i32, alpha, beta: f32) {
		cblas_sgemm(
			CblasRowMajor,
			CblasNoTrans,
			CblasNoTrans,
			M, N, K,
			alpha,
			raw_data(A), K, // lda = K (row-major A is M×K)
			raw_data(B), N, // ldb = N
			beta,
			raw_data(C), N, // ldc = N
		)
	}

} else {

	accelerate_sgemm :: proc(C, A, B: []f32, M, K, N: i32, alpha, beta: f32) {
		panic("Accelerate only available on Darwin")
	}

}
