package ml

// ============================================================================
// Spatial kernels — Conv2d (im2col + matmul_f32 / Accelerate) + MaxPool2d.
// ============================================================================

out_spatial :: proc(in_size, k, stride, pad: i32) -> i32 {
	return (in_size + 2 * pad - k) / stride + 1
}

// col is [K, P] row-major, K=Ci*kH*kW, P=N*Ho*Wo
im2col :: proc(
	col, x: []f32,
	N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo: i32,
) {
	K := Ci * kH * kW
	P := N * Ho * Wo
	for i in 0..<int(K * P) do col[i] = 0
	for n in 0..<N {
		for ho in 0..<Ho {
			for wo in 0..<Wo {
				p := (n * Ho + ho) * Wo + wo
				in_h0 := ho * sH - pH
				in_w0 := wo * sW - pW
				for ci in 0..<Ci {
					for kh in 0..<kH {
						ih := in_h0 + kh
						for kw in 0..<kW {
							iw := in_w0 + kw
							k := (ci * kH + kh) * kW + kw
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								xi := ((n * Ci + ci) * H + ih) * W + iw
								col[k * P + p] = x[xi]
							}
						}
					}
				}
			}
		}
	}
}

// Accumulate col [K,P] back into dx [N,Ci,H,W]
col2im :: proc(
	dx, col: []f32,
	N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo: i32,
) {
	P := N * Ho * Wo
	for n in 0..<N {
		for ho in 0..<Ho {
			for wo in 0..<Wo {
				p := (n * Ho + ho) * Wo + wo
				in_h0 := ho * sH - pH
				in_w0 := wo * sW - pW
				for ci in 0..<Ci {
					for kh in 0..<kH {
						ih := in_h0 + kh
						for kw in 0..<kW {
							iw := in_w0 + kw
							k := (ci * kH + kh) * kW + kw
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								xi := ((n * Ci + ci) * H + ih) * W + iw
								dx[xi] += col[k * P + p]
							}
						}
					}
				}
			}
		}
	}
}

// out_mat [Co,P] → out NCHW [N,Co,Ho,Wo]
scatter_nchw_from_mat :: proc(out, mat: []f32, N, Co, Ho, Wo: i32) {
	P := N * Ho * Wo
	for n in 0..<N {
		for co in 0..<Co {
			for ho in 0..<Ho {
				for wo in 0..<Wo {
					p := (n * Ho + ho) * Wo + wo
					out[((n * Co + co) * Ho + ho) * Wo + wo] = mat[co * P + p]
				}
			}
		}
	}
}

// dout NCHW → mat [Co,P]
gather_mat_from_nchw :: proc(mat, dout: []f32, N, Co, Ho, Wo: i32) {
	P := N * Ho * Wo
	for n in 0..<N {
		for co in 0..<Co {
			for ho in 0..<Ho {
				for wo in 0..<Wo {
					p := (n * Ho + ho) * Wo + wo
					mat[co * P + p] = dout[((n * Co + co) * Ho + ho) * Wo + wo]
				}
			}
		}
	}
}

// Transpose row-major A[M,N] → AT[N,M]
transpose_mn :: proc(AT, A: []f32, M, N: i32) {
	for i in 0..<M {
		for j in 0..<N {
			AT[j * M + i] = A[i * N + j]
		}
	}
}

// out [N,Co,Ho,Wo] = conv(x, w) via im2col + matmul_f32 (Accelerate on Darwin)
conv2d_f32 :: proc(
	out, x, w: []f32,
	N, Ci, H, W, Co, kH, kW, sH, sW, pH, pW: i32,
) {
	Ho := out_spatial(H, kH, sH, pH)
	Wo := out_spatial(W, kW, sW, pW)
	K := Ci * kH * kW
	P := N * Ho * Wo

	col := make([]f32, K * P)
	im2col(col, x, N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo)

	// out_mat[Co,P] = W[Co,K] @ col[K,P]
	out_mat := make([]f32, Co * P)
	matmul_f32(out_mat, w, col, Co, K, P)
	scatter_nchw_from_mat(out, out_mat, N, Co, Ho, Wo)
}

// dX via W^T @ dout_mat then col2im
conv2d_backward_input :: proc(
	dx, dout, w: []f32,
	N, Ci, H, W, Co, kH, kW, sH, sW, pH, pW: i32,
) {
	for i in 0..<len(dx) do dx[i] = 0
	Ho := out_spatial(H, kH, sH, pH)
	Wo := out_spatial(W, kW, sW, pW)
	K := Ci * kH * kW
	P := N * Ho * Wo

	dout_mat := make([]f32, Co * P)
	gather_mat_from_nchw(dout_mat, dout, N, Co, Ho, Wo)

	// dcol[K,P] = W^T[K,Co] @ dout[Co,P]
	wT := make([]f32, K * Co)
	transpose_mn(wT, w, Co, K)
	dcol := make([]f32, K * P)
	matmul_f32(dcol, wT, dout_mat, K, Co, P)
	col2im(dx, dcol, N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo)
}

// dW = dout_mat @ col^T
conv2d_backward_weight :: proc(
	dw, dout, x: []f32,
	N, Ci, H, W, Co, kH, kW, sH, sW, pH, pW: i32,
) {
	Ho := out_spatial(H, kH, sH, pH)
	Wo := out_spatial(W, kW, sW, pW)
	K := Ci * kH * kW
	P := N * Ho * Wo

	col := make([]f32, K * P)
	im2col(col, x, N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo)

	dout_mat := make([]f32, Co * P)
	gather_mat_from_nchw(dout_mat, dout, N, Co, Ho, Wo)

	// dw[Co,K] = dout[Co,P] @ col^T[P,K]
	colT := make([]f32, P * K)
	transpose_mn(colT, col, K, P)
	matmul_f32(dw, dout_mat, colT, Co, P, K)
}

// out[n,c,h,w] += bias[c]  — fast NCHW channel bias
bias_add_nchw :: proc(out, bias: []f32, N, C, H, W: i32) {
	HW := H * W
	for n in 0..<N {
		for c in 0..<C {
			b := bias[c]
			base := (n * C + c) * HW
			for i in 0..<HW do out[base + i] += b
		}
	}
}

// MaxPool2d forward; indices[out_i] = flat index into x (same NCHW layout).
maxpool2d_f32 :: proc(
	out, x: []f32, indices: []i32,
	N, C, H, W, kH, kW, sH, sW, pH, pW: i32,
) {
	Ho := out_spatial(H, kH, sH, pH)
	Wo := out_spatial(W, kW, sW, pW)
	for n in 0..<N {
		for c in 0..<C {
			for ho in 0..<Ho {
				for wo in 0..<Wo {
					best: f32 = -3.4e38
					best_idx: i32 = 0
					in_h0 := ho * sH - pH
					in_w0 := wo * sW - pW
					for kh in 0..<kH {
						ih := in_h0 + kh
						if ih < 0 || ih >= H do continue
						for kw in 0..<kW {
							iw := in_w0 + kw
							if iw < 0 || iw >= W do continue
							xi := ((n * C + c) * H + ih) * W + iw
							v := x[xi]
							if v > best {
								best = v
								best_idx = xi
							}
						}
					}
					oi := ((n * C + c) * Ho + ho) * Wo + wo
					out[oi] = best
					indices[oi] = best_idx
				}
			}
		}
	}
}

maxpool2d_backward :: proc(dx, dout: []f32, indices: []i32) {
	for i in 0..<len(dx) do dx[i] = 0
	for i in 0..<len(dout) {
		dx[indices[i]] += dout[i]
	}
}
