package ml

// ============================================================================
// Contiguous f32 kernels.
//
// Portable Odin #simd vectors (`#simd[4]f32`) lower to ARM NEON on Apple
// Silicon (and SSE/AVX on x86). This is the right first step — write portable
// SIMD, measure, only drop to core:simd/arm/neon if you need instructions the
// portable layer does not expose.
//
// Scope today:
//   - same-length contiguous elementwise add/mul (the bulk of tensor math)
//   - scalar tail for leftovers
//
// Broadcast, strided, and matmul microkernels come later.
// ============================================================================

import "base:intrinsics"
import "core:simd"

// Contiguous out[i] = a[i] + b[i] for len(a) == len(b) == len(out).
add_f32_contiguous :: proc(out, a, b: []f32) {
	assert(len(out) == len(a) && len(a) == len(b))
	n := len(out)
	i := 0
	for i + 4 <= n {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		vb := intrinsics.unaligned_load((^simd.f32x4)(&b[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&out[i]), simd.add(va, vb))
		i += 4
	}
	for ; i < n; i += 1 do out[i] = a[i] + b[i]
}

// Contiguous out[i] = a[i] * b[i].
mul_f32_contiguous :: proc(out, a, b: []f32) {
	assert(len(out) == len(a) && len(a) == len(b))
	n := len(out)
	i := 0
	for i + 4 <= n {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		vb := intrinsics.unaligned_load((^simd.f32x4)(&b[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&out[i]), simd.mul(va, vb))
		i += 4
	}
	for ; i < n; i += 1 do out[i] = a[i] * b[i]
}

// Contiguous out[i] = a[i] - b[i].
sub_f32_contiguous :: proc(out, a, b: []f32) {
	assert(len(out) == len(a) && len(a) == len(b))
	n := len(out)
	i := 0
	for i + 4 <= n {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		vb := intrinsics.unaligned_load((^simd.f32x4)(&b[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&out[i]), simd.sub(va, vb))
		i += 4
	}
	for ; i < n; i += 1 do out[i] = a[i] - b[i]
}

// Contiguous out[i] = a[i] / b[i].
div_f32_contiguous :: proc(out, a, b: []f32) {
	assert(len(out) == len(a) && len(a) == len(b))
	n := len(out)
	i := 0
	for i + 4 <= n {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		vb := intrinsics.unaligned_load((^simd.f32x4)(&b[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&out[i]), simd.div(va, vb))
		i += 4
	}
	for ; i < n; i += 1 do out[i] = a[i] / b[i]
}

// Contiguous out[i] = -a[i].
neg_f32_contiguous :: proc(out, a: []f32) {
	assert(len(out) == len(a))
	n := len(out)
	i := 0
	for i + 4 <= n {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&out[i]), simd.neg(va))
		i += 4
	}
	for ; i < n; i += 1 do out[i] = -a[i]
}

// Contiguous out[i] = max(a[i], 0).
relu_f32_contiguous :: proc(out, a: []f32) {
	assert(len(out) == len(a))
	n := len(out)
	zero: simd.f32x4 = 0
	i := 0
	for i + 4 <= n {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&out[i]), simd.max(va, zero))
		i += 4
	}
	for ; i < n; i += 1 do out[i] = a[i] > 0 ? a[i] : 0
}