package ml

// ============================================================================
// Apple Accelerate CBLAS bindings (Darwin only).
// Same library NumPy uses for matmul on macOS.
// ============================================================================

when ODIN_OS == .Darwin {

	// cblas.h enums (partial)
	CblasRowMajor :: 101
	CblasNoTrans  :: 111
	CblasTrans    :: 112

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

	// C[M,N] = op(A) @ op(B), row-major. Leading dims are the stored row lengths.
	accelerate_sgemm :: proc(C, A, B: []f32, M, K, N: i32, trans_a, trans_b: bool) {
		cblas_sgemm(
			CblasRowMajor,
			trans_a ? CblasTrans : CblasNoTrans,
			trans_b ? CblasTrans : CblasNoTrans,
			M, N, K,
			1,
			raw_data(A), trans_a ? M : K,
			raw_data(B), trans_b ? K : N,
			0,
			raw_data(C), N,
		)
	}

} else {

	accelerate_sgemm :: proc(C, A, B: []f32, M, K, N: i32, trans_a, trans_b: bool) {
		panic("Accelerate only available on Darwin")
	}

}
