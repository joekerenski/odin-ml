package ml

// ============================================================================
// nn — minimal layer helpers (Linear) and weight initialization.
//
// Linear holds weight W [in, out] and bias b [out] as leaf tensors with
// requires_grad = true. linear_forward applies x @ W + b (bias broadcasts
// across the batch via the row-broadcast SIMD fast path in add()).
//
// Init schemes:
//   .He     — std = sqrt(2 / fan_in)      (good for ReLU layers)
//   .Xavier — std = sqrt(2 / (fan_in+fan_out))  (good for tanh/sigmoid/softmax)
//   .Zeros  — all zeros (rarely useful; for bias use zeros() directly)
//
// Parameters are allocated with whatever allocator is current at call time —
// typically the persistent allocator (before the per-step arena is set up).
// ============================================================================

import "core:math"

Init :: enum { He, Xavier, Zeros }

Linear :: struct {
	W: ^Tensor, // [in_dim, out_dim]
	b: ^Tensor, // [out_dim]
}

linear :: proc(in_dim, out_dim: i32, init: Init = .He) -> Linear {
	l: Linear
	switch init {
	case .He:
		std := math.sqrt(2.0 / f32(in_dim))
		l.W = randn({in_dim, out_dim}, 0.0, std, requires_grad = true)
	case .Xavier:
		std := math.sqrt(2.0 / f32(in_dim + out_dim))
		l.W = randn({in_dim, out_dim}, 0.0, std, requires_grad = true)
	case .Zeros:
		l.W = zeros({in_dim, out_dim}, requires_grad = true)
	}
	l.b = zeros({out_dim}, requires_grad = true)
	return l
}

// out = x @ W + b   —  x:[B, in]  W:[in, out]  b:[out]  ->  [B, out]
// The bias [out] broadcasts against [B, out] via the row-broadcast fast path.
linear_forward :: proc(l: ^Linear, x: ^Tensor) -> ^Tensor {
	return add(matmul(x, l.W), l.b)
}