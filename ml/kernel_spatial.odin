package ml

// ============================================================================
// Spatial kernels — Conv2d (im2col + matmul_f32 / Accelerate) + MaxPool2d.
// ============================================================================

// All loops below are independent per image (each writes only its own part
// of the output), so they run over the batch on all cores.
@(private)
Spatial_Job :: struct {
	dst, src:                                  []f32,
	Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo: i32,
	C, P:                                      i32, // channels of mat ↔ NCHW; N·Ho·Wo
	win:                                       Window,
}

// col is [K, P] row-major, K=Ci*kH*kW, P=N*Ho*Wo
im2col :: proc(
	col, x: []f32,
	N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo: i32,
) {
	job := Spatial_Job{dst = col, src = x, Ci = Ci, H = H, W = W, kH = kH, kW = kW, sH = sH, sW = sW, pH = pH, pW = pW, Ho = Ho, Wo = Wo, P = N * Ho * Wo}
	parallel_for(int(N), 1, proc(data: rawptr, lo, hi: int) {
		using j := (^Spatial_Job)(data)
		col, x := dst, src
		for n in i32(lo) ..< i32(hi) {
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
								inside := ih >= 0 && ih < H && iw >= 0 && iw < W
								col[k * P + p] = inside ? x[((n * Ci + ci) * H + ih) * W + iw] : 0
							}
						}
					}
				}
			}
		}
	}, &job)
}

// Accumulate col [K,P] back into dx [N,Ci,H,W]
col2im :: proc(
	dx, col: []f32,
	N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo: i32,
) {
	job := Spatial_Job{dst = dx, src = col, Ci = Ci, H = H, W = W, kH = kH, kW = kW, sH = sH, sW = sW, pH = pH, pW = pW, Ho = Ho, Wo = Wo, P = N * Ho * Wo}
	parallel_for(int(N), 1, proc(data: rawptr, lo, hi: int) {
		using j := (^Spatial_Job)(data)
		dx, col := dst, src
		for n in i32(lo) ..< i32(hi) {
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
	}, &job)
}

// out_mat [Co,P] → out NCHW [N,Co,Ho,Wo]
scatter_nchw_from_mat :: proc(out, mat: []f32, N, Co, Ho, Wo: i32) {
	job := Spatial_Job{dst = out, src = mat, C = Co, Ho = Ho, Wo = Wo, P = N * Ho * Wo}
	parallel_for(int(N), 1, proc(data: rawptr, lo, hi: int) {
		using j := (^Spatial_Job)(data)
		hw := Ho * Wo
		for n in i32(lo) ..< i32(hi) {
			for co in 0..<C {
				copy(dst[(n * C + co) * hw:][:hw], src[co * P + n * hw:][:hw])
			}
		}
	}, &job)
}

// dout NCHW → mat [Co,P]
gather_mat_from_nchw :: proc(mat, dout: []f32, N, Co, Ho, Wo: i32) {
	job := Spatial_Job{dst = mat, src = dout, C = Co, Ho = Ho, Wo = Wo, P = N * Ho * Wo}
	parallel_for(int(N), 1, proc(data: rawptr, lo, hi: int) {
		using j := (^Spatial_Job)(data)
		hw := Ho * Wo
		for n in i32(lo) ..< i32(hi) {
			for co in 0..<C {
				copy(dst[co * P + n * hw:][:hw], src[(n * C + co) * hw:][:hw])
			}
		}
	}, &job)
}

// out [N,Co,Ho,Wo] = conv(x, w) via im2col + GEMM
conv2d_f32 :: proc(out, x, w: []f32, N, Ci, H, W, Co: i32, win: Window) {
	kH, kW, sH, sW, pH, pW := win.kH, win.kW, win.sH, win.sW, win.pH, win.pW
	Ho := out_spatial(H, kH, sH, pH)
	Wo := out_spatial(W, kW, sW, pW)
	K := Ci * kH * kW
	P := N * Ho * Wo

	col := make([]f32, K * P, scratch())
	defer delete(col, scratch())
	im2col(col, x, N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo)

	// out_mat[Co,P] = W[Co,K] @ col[K,P]
	out_mat := make([]f32, Co * P, scratch())
	defer delete(out_mat, scratch())
	matmul_f32(out_mat, w, col, Co, K, P)
	scatter_nchw_from_mat(out, out_mat, N, Co, Ho, Wo)
}

// dx [N,Ci,H,W]: dcol[K,P] = W^T @ dout_mat, then col2im
conv2d_backward_input :: proc(dx, dout, w: []f32, N, Ci, H, W, Co: i32, win: Window) {
	kH, kW, sH, sW, pH, pW := win.kH, win.kW, win.sH, win.sW, win.pH, win.pW
	for i in 0 ..< len(dx) do dx[i] = 0
	Ho := out_spatial(H, kH, sH, pH)
	Wo := out_spatial(W, kW, sW, pW)
	K := Ci * kH * kW
	P := N * Ho * Wo

	dout_mat := make([]f32, Co * P, scratch())
	defer delete(dout_mat, scratch())
	gather_mat_from_nchw(dout_mat, dout, N, Co, Ho, Wo)
	dcol := make([]f32, K * P, scratch())
	defer delete(dcol, scratch())
	matmul_f32(dcol, w, dout_mat, K, Co, P, trans_a = true)
	col2im(dx, dcol, N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo)
}

// dw [Co,Ci,kH,kW] = dout_mat[Co,P] @ col^T
conv2d_backward_weight :: proc(dw, dout, x: []f32, N, Ci, H, W, Co: i32, win: Window) {
	kH, kW, sH, sW, pH, pW := win.kH, win.kW, win.sH, win.sW, win.pH, win.pW
	Ho := out_spatial(H, kH, sH, pH)
	Wo := out_spatial(W, kW, sW, pW)
	K := Ci * kH * kW
	P := N * Ho * Wo

	col := make([]f32, K * P, scratch())
	defer delete(col, scratch())
	im2col(col, x, N, Ci, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo)
	dout_mat := make([]f32, Co * P, scratch())
	defer delete(dout_mat, scratch())
	gather_mat_from_nchw(dout_mat, dout, N, Co, Ho, Wo)
	matmul_f32(dw, dout_mat, col, Co, P, K, trans_b = true)
}

// Flat index into x of the max in output window (n,c,ho,wo). First max wins.
maxpool_argmax :: #force_inline proc(x: []f32, n, c, ho, wo, C, H, W: i32, win: Window) -> i32 {
	kH, kW, sH, sW, pH, pW := win.kH, win.kW, win.sH, win.sW, win.pH, win.pW
	best: f32 = -3.4e38
	best_idx: i32 = 0
	in_h0 := ho * sH - pH
	in_w0 := wo * sW - pW
	for kh in 0 ..< kH {
		ih := in_h0 + kh
		if ih < 0 || ih >= H do continue
		for kw in 0 ..< kW {
			iw := in_w0 + kw
			if iw < 0 || iw >= W do continue
			xi := ((n * C + c) * H + ih) * W + iw
			if x[xi] > best {
				best = x[xi]
				best_idx = xi
			}
		}
	}
	return best_idx
}

maxpool2d_f32 :: proc(out, x: []f32, N, C, H, W: i32, win: Window) {
	job := Spatial_Job{dst = out, src = x, C = C, H = H, W = W, win = win,
		Ho = out_spatial(H, win.kH, win.sH, win.pH), Wo = out_spatial(W, win.kW, win.sW, win.pW)}
	parallel_for(int(N), 1, proc(data: rawptr, lo, hi: int) {
		using j := (^Spatial_Job)(data)
		for n in i32(lo) ..< i32(hi) do for c in 0 ..< C do for ho in 0 ..< Ho do for wo in 0 ..< Wo {
			dst[((n * C + c) * Ho + ho) * Wo + wo] = src[maxpool_argmax(src, n, c, ho, wo, C, H, W, win)]
		}
	}, &job)
}

// Route each output grad to its window's argmax (recomputed from x).
@(private)
Pool_Bwd_Job :: struct {
	dx, dout, x:     []f32,
	C, H, W, Ho, Wo: i32,
	win:             Window,
}

maxpool2d_backward :: proc(dx, dout, x: []f32, N, C, H, W: i32, win: Window) {
	job := Pool_Bwd_Job{dx, dout, x, C, H, W, out_spatial(H, win.kH, win.sH, win.pH), out_spatial(W, win.kW, win.sW, win.pW), win}
	parallel_for(int(N), 1, proc(data: rawptr, lo, hi: int) {
		using j := (^Pool_Bwd_Job)(data)
		chw := int(C * H * W)
		for i in lo * chw ..< hi * chw do dx[i] = 0
		for n in i32(lo) ..< i32(hi) do for c in 0 ..< C do for ho in 0 ..< Ho do for wo in 0 ..< Wo {
			dx[maxpool_argmax(x, n, c, ho, wo, C, H, W, win)] += dout[((n * C + c) * Ho + ho) * Wo + wo]
		}
	}, &job)
}
