package dt

// ============================================================================
// Full-covariance Gaussian mixtures in D dimensions.
//
// Each component is parameterized by the Cholesky factor U of its PRECISION
// (Λ = U Uᵀ, U lower triangular, positive diagonal):
//
//   log N(x; μ, Λ⁻¹) = Σ_j log U_jj − ½ ‖Uᵀ(x − μ)‖² − (D/2) log 2π
//
// so the loss needs no triangular solve: Uᵀr is a diagonal scale plus a sum
// over the strict lower triangle, which two constant 0/1 matmuls express with
// ops the library already has. (The paper factors the covariance instead; both
// span all full-covariance Gaussians.)
//
// Per component the network predicts: logit (weights = softmax over the K
// components), mean [D], log diag U [D], strict lower triangle of U [M],
// M = D(D−1)/2, ordered row by row: (1,0) (2,0) (2,1) (3,0) ...
//
// The host side (f64, fixed D) converts between mixtures, covariances and the
// flat feature vectors fed to the network, and evaluates densities.
// ============================================================================

import "core:math"
import "core:math/linalg"
import ml "../ml"

D :: 4 // state dimension of the tracking experiment
M :: D * (D - 1) / 2
N_FEAT :: 1 + D + D + M // per component: log w, μ, log diag U, lower U

Mat :: matrix[D, D]f64
Vec :: [D]f64

// Log-diagonal clamp for the network's output (reference impl: [-11, 14] on
// the covariance factor); keeps exp() finite early in training.
LOG_DIAG_MIN :: -7.0
LOG_DIAG_MAX :: 7.0
LOG_W_MIN :: -14.0

// ---- graph side ------------------------------------------------------------

Gmm_Full :: struct {
	logits:   ^ml.Tensor, // [B, K, 1]
	means:    ^ml.Tensor, // [B, K, D]
	log_diag: ^ml.Tensor, // [B, K, D]  (clipped)
	lower:    ^ml.Tensor, // [B, K, M]
}

// Constant matrices: sel [D, M] picks r_i for pair p=(i,j); put [M, D] adds it to row j.
Tril_Maps :: struct {
	sel, put: ^ml.Tensor,
}

tril_maps :: proc() -> Tril_Maps {
	sel, put := make([]f32, D * M), make([]f32, M * D)
	p := 0
	for i in 1 ..< D {
		for j in 0 ..< i {
			sel[i * M + p] = 1
			put[p * D + j] = 1
			p += 1
		}
	}
	return {ml.from_data(sel, {D, M}), ml.from_data(put, {M, D})}
}

// log q(x) per row, x [B, 1, D] → [B, 1, 1]
gmm_full_log_prob :: proc(g: Gmm_Full, x: ^ml.Tensor, maps: Tril_Maps) -> ^ml.Tensor {
	r := ml.sub(x, g.means) // [B, K, D]
	off := ml.matmul(ml.mul(g.lower, ml.matmul(r, maps.sel)), maps.put) // Σ_{i>j} U_ij r_i
	y := ml.add(ml.mul(ml.exp(g.log_diag), r), off) // Uᵀr
	comp := ml.sub(ml.sum(g.log_diag, -1), ml.mul(ml.sum(ml.square(y), -1), ml.scalar(0.5)))
	lse := ml.logsumexp(ml.add(ml.log_softmax(g.logits, 1), comp), 1)
	return ml.sub(lse, ml.scalar(f32(D) * LOG_SQRT_2PI))
}

gmm_full_nll :: proc(g: Gmm_Full, x: ^ml.Tensor, maps: Tril_Maps) -> ^ml.Tensor {
	return ml.neg(ml.mean(gmm_full_log_prob(g, x, maps)))
}

// ---- host side ---------------------------------------------------------------

Component :: struct {
	w:  f64,
	mu: Vec,
	U:  Mat, // Cholesky factor of the precision
}

Mixture :: []Component

// Lower Cholesky factor of a symmetric positive-definite matrix.
cholesky :: proc(a: Mat) -> (L: Mat, ok: bool) {
	for j in 0 ..< D {
		s := a[j, j]
		for k in 0 ..< j do s -= L[j, k] * L[j, k]
		if s <= 0 do return {}, false
		L[j, j] = math.sqrt(s)
		for i in j + 1 ..< D {
			t := a[i, j]
			for k in 0 ..< j do t -= L[i, k] * L[j, k]
			L[i, j] = t / L[j, j]
		}
	}
	return L, true
}

covariance :: proc(c: Component) -> Mat {
	return linalg.inverse(c.U * linalg.transpose(c.U))
}

// Component with covariance Σ (precision factor via Cholesky of Σ⁻¹).
component_from_cov :: proc(w: f64, mu: Vec, cov: Mat) -> (c: Component, ok: bool) {
	prec := linalg.inverse(cov)
	prec = 0.5 * (prec + linalg.transpose(prec))
	U: Mat
	U, ok = cholesky(prec)
	return {w, mu, U}, ok
}

component_log_pdf :: proc(c: Component, x: Vec) -> f64 {
	r := x - c.mu
	s: f64 = 0
	for j in 0 ..< D {
		y: f64 = 0
		for i in j ..< D do y += c.U[i, j] * r[i] // (Uᵀr)_j
		s += y * y
	}
	ld: f64 = 0
	for j in 0 ..< D do ld += math.ln(c.U[j, j])
	return ld - 0.5 * s - f64(D) * LOG_SQRT_2PI
}

mixture_log_pdf :: proc(mix: Mixture, x: Vec) -> f64 {
	m := math.inf_f64(-1)
	lp: [16]f64
	for c, k in mix {
		lp[k] = math.ln(max(c.w, 1e-300)) + component_log_pdf(c, x)
		m = max(m, lp[k])
	}
	s: f64 = 0
	for k in 0 ..< len(mix) do s += math.exp(lp[k] - m)
	return m + math.ln(s)
}

// Flat network features of one component: log w, μ, log diag U, lower U.
component_features :: proc(c: Component, out: []f32) {
	out[0] = f32(max(math.ln(max(c.w, 1e-300)), LOG_W_MIN))
	for j in 0 ..< D do out[1 + j] = f32(c.mu[j])
	for j in 0 ..< D do out[1 + D + j] = f32(math.ln(c.U[j, j]))
	p := 1 + 2 * D
	for i in 1 ..< D do for j in 0 ..< i {
		out[p] = f32(c.U[i, j])
		p += 1
	}
}

// Realized network output, row b → mixture (weights by softmax, clamps as in the loss).
gmm_full_row :: proc(g: Gmm_Full, b: int, out: Mixture) {
	K := len(out)
	lg := g.logits.data[b * K:][:K]
	mx := f64(lg[0])
	for l in lg do mx = max(mx, f64(l))
	z: f64 = 0
	for l in lg do z += math.exp(f64(l) - mx)
	for k in 0 ..< K {
		c := &out[k]
		c.w = math.exp(f64(lg[k]) - mx) / z
		c.U = {}
		for j in 0 ..< D {
			c.mu[j] = f64(g.means.data[(b * K + k) * D + j])
			ld := clamp(f64(g.log_diag.data[(b * K + k) * D + j]), LOG_DIAG_MIN, LOG_DIAG_MAX)
			c.U[j, j] = math.exp(ld)
		}
		p := 0
		for i in 1 ..< D do for j in 0 ..< i {
			c.U[i, j] = f64(g.lower.data[(b * K + k) * M + p])
			p += 1
		}
	}
}
