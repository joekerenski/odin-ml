package ml

// ============================================================================
// nn — layer helpers (Linear, Conv2d, LayerNorm) + param collection.
//
// No Module base class (tinygrad style): layers are plain structs holding
// ^Tensor params. Train loops collect params into a list for the optimizer:
//
//   params: [dynamic]^Tensor
//   linear_params(&params, l1)
//   conv_params(&params, c1)
//   opt := new_sgd_list(lr, mom, params[:])
//
// Init: .He (ReLU), .Xavier (linear/softmax heads), .Zeros.
// ============================================================================

import "core:math"

Init :: enum { He, Xavier, Zeros }

// ---- param bag ------------------------------------------------------------

// Append parameter tensors into a dynamic list (persistent allocator).
collect_params :: proc(dst: ^[dynamic]^Tensor, params: ..^Tensor) {
	for p in params {
		if p != nil do append(dst, p)
	}
}

// ---- Linear ---------------------------------------------------------------

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
linear_forward :: proc(l: ^Linear, x: ^Tensor) -> ^Tensor {
	return add(matmul(x, l.W), l.b)
}

// Append Linear params into dst.
linear_params :: proc(dst: ^[dynamic]^Tensor, l: Linear) {
	collect_params(dst, l.W, l.b)
}

// ---- Conv2d (NCHW) --------------------------------------------------------

Conv2d :: struct {
	W:       ^Tensor, // [out_c, in_c, kH, kW]
	b:       ^Tensor, // [out_c]
	stride:  i32,
	padding: i32,
}

// Square kernel. bias always on (MNIST-style).
conv2d_layer :: proc(
	in_c, out_c, kernel_size: i32,
	stride: i32 = 1,
	padding: i32 = 0,
	init: Init = .He,
) -> Conv2d {
	fan_in := in_c * kernel_size * kernel_size
	fan_out := out_c * kernel_size * kernel_size
	c: Conv2d
	c.stride = stride
	c.padding = padding
	switch init {
	case .He:
		std := math.sqrt(2.0 / f32(fan_in))
		c.W = randn({out_c, in_c, kernel_size, kernel_size}, 0.0, std, requires_grad = true)
	case .Xavier:
		std := math.sqrt(2.0 / f32(fan_in + fan_out))
		c.W = randn({out_c, in_c, kernel_size, kernel_size}, 0.0, std, requires_grad = true)
	case .Zeros:
		c.W = zeros({out_c, in_c, kernel_size, kernel_size}, requires_grad = true)
	}
	c.b = zeros({out_c}, requires_grad = true)
	return c
}

// x:[N,Ci,H,W] → [N,Co,Ho,Wo]; bias broadcasts over N,H,W via reshape to [1,Co,1,1]
conv2d_forward :: proc(c: ^Conv2d, x: ^Tensor) -> ^Tensor {
	y := conv2d(x, c.W, c.stride, c.padding)
	// bias [Co] → [1, Co, 1, 1] for NCHW broadcast
	b4 := reshape(c.b, {1, c.b.shape[0], 1, 1})
	return add(y, b4)
}

// Append Conv2d params into dst.
conv_params :: proc(dst: ^[dynamic]^Tensor, c: Conv2d) {
	collect_params(dst, c.W, c.b)
}

// ---- LayerNorm (last axis, learnable gain + bias) -------------------------

LayerNorm :: struct {
	g:   ^Tensor, // [dim], init 1
	b:   ^Tensor, // [dim], init 0
	eps: f32,
}

layer_norm_layer :: proc(dim: i32, eps: f32 = 1e-5) -> LayerNorm {
	return LayerNorm{ones({dim}, requires_grad = true), zeros({dim}, requires_grad = true), eps}
}

layer_norm_forward :: proc(l: ^LayerNorm, x: ^Tensor) -> ^Tensor {
	return add(mul(layer_norm(x, l.eps), l.g), l.b)
}

layer_norm_params :: proc(dst: ^[dynamic]^Tensor, l: LayerNorm) {
	collect_params(dst, l.g, l.b)
}
