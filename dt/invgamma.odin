package dt

// ============================================================================
// The paper's conjugate check (Sec. 5.1): unknown variance of Gaussian data.
//
//   σ² ~ InvGamma(α, β)         prior, (α, β) drawn per problem from a meta-prior
//   z_i ~ N(0, σ²), i = 1..n    observations
//   σ² | z ~ InvGamma(α + n/2, β + Σz²/2)   exact posterior
//
// Models work in y = log σ² (unbounded; the paper's change of measure).
// If σ² ~ InvGamma(α, β), then  log p(y) = α log β − lgamma(α) − α y − β e^(−y).
// ============================================================================

import "core:math"
import "core:math/rand"

// Uniform ranges for log α and log β.
Meta_Prior :: struct {
	log_alpha, log_beta: [2]f64,
}

META_PRIOR :: Meta_Prior{{math.LN2 - 0.3, math.LN10 - 0.2}, {-1.2, 1.6}} // α ∈ [1.5, 8], β ∈ [0.3, 5]

Problem :: struct {
	alpha, beta: f64, // prior
	z:           []f64, // observations
	y:           f64, // the true log σ² that generated z
}

sample_problem :: proc(meta: Meta_Prior, n: int) -> Problem {
	p: Problem
	p.alpha = math.exp(rand.float64_range(meta.log_alpha[0], meta.log_alpha[1]))
	p.beta = math.exp(rand.float64_range(meta.log_beta[0], meta.log_beta[1]))
	precision := rand.float64_gamma(p.alpha, 1 / p.beta) // Gamma(shape α, rate β)
	p.y = -math.ln(precision)
	sd := math.exp(0.5 * p.y)
	p.z = make([]f64, n)
	for i in 0 ..< n do p.z[i] = sd * rand.float64_normal(0, 1)
	return p
}

posterior :: proc(p: Problem) -> (alpha, beta: f64) {
	ss: f64 = 0
	for v in p.z do ss += v * v
	return p.alpha + 0.5 * f64(len(p.z)), p.beta + 0.5 * ss
}

// Density of y = log σ² when σ² ~ InvGamma(α, β).
invgamma_logpdf_y :: proc(alpha, beta, y: f64) -> f64 {
	lg, _ := math.lgamma(alpha)
	return alpha * math.ln(beta) - lg - alpha * y - beta * math.exp(-y)
}
