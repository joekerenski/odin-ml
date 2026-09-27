package dt

// ============================================================================
// 1D Gaussian mixtures — the distribution family the paper maps between.
//
// A model predicts, per batch row, K components as three [B, K] tensors:
//   logits    mixture weights (softmax)
//   means     component means
//   log_stds  log standard deviations
//
// gmm_nll is the training loss (graph ops, differentiable); gmm_logpdf is the
// plain f64 density used for evaluation.
// ============================================================================

import "core:math"
import ml "../ml"

Gmm :: struct {
	logits, means, log_stds: ^ml.Tensor, // [B, K]
}

// Three linear heads from a hidden layer. Means get random init so the
// components start apart (identical components never separate).
Gmm_Head :: struct {
	logits, means, log_stds: ml.Linear,
}

gmm_head :: proc(hidden, K: i32) -> Gmm_Head {
	return {ml.linear(hidden, K, .Zeros), ml.linear(hidden, K, .Xavier), ml.linear(hidden, K, .Zeros)}
}

gmm_head_forward :: proc(h: ^Gmm_Head, x: ^ml.Tensor) -> Gmm {
	return {
		ml.linear_forward(&h.logits, x),
		ml.linear_forward(&h.means, x),
		ml.linear_forward(&h.log_stds, x),
	}
}

gmm_head_params :: proc(dst: ^[dynamic]^ml.Tensor, h: Gmm_Head) {
	ml.linear_params(dst, h.logits)
	ml.linear_params(dst, h.means)
	ml.linear_params(dst, h.log_stds)
}

LOG_SQRT_2PI :: 0.91893853320467274178

// log q(y) per row: logsumexp_k( log w_k - log σ_k - ½log 2π - ½((y - μ_k)/σ_k)² )
gmm_log_prob :: proc(g: Gmm, y: ^ml.Tensor) -> ^ml.Tensor {
	z := ml.mul(ml.sub(y, g.means), ml.exp(ml.neg(g.log_stds)))
	comp := ml.sub(ml.sub(ml.log_softmax(g.logits, -1), g.log_stds), ml.mul(ml.square(z), ml.scalar(0.5)))
	return ml.sub(ml.logsumexp(comp, -1), ml.scalar(LOG_SQRT_2PI))
}

// Mean negative log-likelihood of targets y [B, 1] under the predicted mixtures.
gmm_nll :: proc(g: Gmm, y: ^ml.Tensor) -> ^ml.Tensor {
	return ml.neg(ml.mean(gmm_log_prob(g, y)))
}

// One mixture (row) of realized parameters, for evaluation.
Gmm_Row :: struct {
	logits, means, log_stds: []f32,
}

gmm_row :: proc(g: Gmm, b: int) -> Gmm_Row {
	K := int(g.logits.shape[1])
	return {g.logits.data[b * K:][:K], g.means.data[b * K:][:K], g.log_stds.data[b * K:][:K]}
}

gmm_logpdf :: proc(g: Gmm_Row, y: f64) -> f64 {
	m := f64(g.logits[0])
	for l in g.logits do m = max(m, f64(l))
	zw: f64 = 0 // softmax normalizer
	for l in g.logits do zw += math.exp(f64(l) - m)
	acc: f64 = 0
	for k in 0 ..< len(g.logits) {
		s := math.exp(f64(g.log_stds[k]))
		u := (y - f64(g.means[k])) / s
		acc += math.exp(f64(g.logits[k]) - m) / zw * math.exp(-0.5 * u * u) / s
	}
	return math.ln(max(acc, 1e-300)) - LOG_SQRT_2PI
}

// Weight, mean, std of each component (for printing).
gmm_components :: proc(g: Gmm_Row, w, mu, sd: []f64) {
	m := f64(g.logits[0])
	for l in g.logits do m = max(m, f64(l))
	zw: f64 = 0
	for l in g.logits do zw += math.exp(f64(l) - m)
	for k in 0 ..< len(g.logits) {
		w[k] = math.exp(f64(g.logits[k]) - m) / zw
		mu[k] = f64(g.means[k])
		sd[k] = math.exp(f64(g.log_stds[k]))
	}
}

count_params :: proc(params: []^ml.Tensor) -> (n: int) {
	for p in params do n += len(p.data)
	return
}
