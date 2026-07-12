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

// ============================================================================
// Broadcast SIMD kernels — fast paths for common NumPy patterns.
//
// Why SIMD helps broadcasting: adjacent output elements read adjacent operand
// elements ONLY when broadcasted dims are outer. These are the most common
// patterns in ML: bias terms ([M,N]+[N]), row-wise scaling ([M,N]+[M,1] → no,
// that one's stride access into b; covered by the scalar path), and full
// scalar broadcasts ([M,N]+[1,1]).
// ============================================================================

// Pattern A: full scalar broadcast.
//   out[i] = a[i] + s   (s read once, broadcast in vector)
// Covers [N]+[1], [M,N]+[1,1], [1]+anything, etc.
add_scalar_contiguous :: proc(out, a: []f32, s: f32) {
	n := len(out)
	sv: simd.f32x4 = s
	i := 0
	for ; i + 4 <= n; i += 4 {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&out[i]), simd.add(va, sv))
	}
	for ; i < n; i += 1 do out[i] = a[i] + s
}

mul_scalar_contiguous :: proc(out, a: []f32, s: f32) {
	n := len(out)
	sv: simd.f32x4 = s
	i := 0
	for ; i + 4 <= n; i += 4 {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&out[i]), simd.mul(va, sv))
	}
	for ; i < n; i += 1 do out[i] = a[i] * s
}

// Pattern B: row broadcast — b is the inner-dim vector, repeats for each outer row.
//   For each row m:  out[m, 0..N] = a[m, 0..N] + b[0..N]
// Covers [M,N]+[N], [B,M,N]+[1,N], [M,N]+[1,N], etc.
// `inner_n` = the matching inner dimension (length of b).
add_row_broadcast :: proc(out, a, b: []f32, outer_rows: i32, inner_n: i32) {
	bv_aligned := inner_n % 4 == 0
	or := int(outer_rows)
	inn := int(inner_n)
	for row in 0..<or {
		a_off := row * inn
		o_off := a_off
		if bv_aligned {
			i := 0
			for ; i + 4 <= inn; i += 4 {
				va := intrinsics.unaligned_load((^simd.f32x4)(&a[a_off + i]))
				vb := intrinsics.unaligned_load((^simd.f32x4)(&b[i]))
				intrinsics.unaligned_store((^simd.f32x4)(&out[o_off + i]), simd.add(va, vb))
			}
			for ; i < inn; i += 1 do out[o_off + i] = a[a_off + i] + b[i]
		} else {
			for i in 0..<inn do out[o_off + i] = a[a_off + i] + b[i]
		}
	}
}

mul_row_broadcast :: proc(out, a, b: []f32, outer_rows: i32, inner_n: i32) {
	bv_aligned := inner_n % 4 == 0
	or := int(outer_rows)
	inn := int(inner_n)
	for row in 0..<or {
		a_off := row * inn
		o_off := a_off
		if bv_aligned {
			i := 0
			for ; i + 4 <= inn; i += 4 {
				va := intrinsics.unaligned_load((^simd.f32x4)(&a[a_off + i]))
				vb := intrinsics.unaligned_load((^simd.f32x4)(&b[i]))
				intrinsics.unaligned_store((^simd.f32x4)(&out[o_off + i]), simd.mul(va, vb))
			}
			for ; i < inn; i += 1 do out[o_off + i] = a[a_off + i] * b[i]
		} else {
			for i in 0..<inn do out[o_off + i] = a[a_off + i] * b[i]
		}
	}
}

add_col_broadcast :: proc(out, a, b: []f32, outer_rows: i32, inner_n: i32) {
	bv_aligned := inner_n % 4 == 0
	or := int(outer_rows)
	inn := int(inner_n)
	for row in 0..<or {
		a_off := row * inn
		o_off := a_off
		bv: simd.f32x4 = b[row]
		if bv_aligned {
			i := 0
			for ; i + 4 <= inn; i += 4 {
				va := intrinsics.unaligned_load((^simd.f32x4)(&a[a_off + i]))
				intrinsics.unaligned_store((^simd.f32x4)(&out[o_off + i]), simd.add(va, bv))
			}
			for ; i < inn; i += 1 do out[o_off + i] = a[a_off + i] + b[row]
		} else {
			for i in 0..<inn do out[o_off + i] = a[a_off + i] + b[row]
		}
	}
}

mul_col_broadcast :: proc(out, a, b: []f32, outer_rows: i32, inner_n: i32) {
	bv_aligned := inner_n % 4 == 0
	or := int(outer_rows)
	inn := int(inner_n)
	for row in 0..<or {
		a_off := row * inn
		o_off := a_off
		bv: simd.f32x4 = b[row]
		if bv_aligned {
			i := 0
			for ; i + 4 <= inn; i += 4 {
				va := intrinsics.unaligned_load((^simd.f32x4)(&a[a_off + i]))
				intrinsics.unaligned_store((^simd.f32x4)(&out[o_off + i]), simd.mul(va, bv))
			}
			for ; i < inn; i += 1 do out[o_off + i] = a[a_off + i] * b[row]
		} else {
			for i in 0..<inn do out[o_off + i] = a[a_off + i] * b[row]
		}
	}
}