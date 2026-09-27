package dt

// ============================================================================
// KL(p ‖ q) for 1D densities by quadrature on a grid around p's mass.
// p is the exact posterior over y = log σ²: InvGamma(α, β) in log space,
// mode at log(β/α), std ≈ 1/√α.
// ============================================================================

import "core:math"

GRID :: 4001

Grid :: struct {
	y, logp: [GRID]f64,
	dy:      f64,
}

invgamma_grid :: proc(alpha, beta: f64) -> (g: Grid) {
	c := math.ln(beta / alpha)
	half := 14 / math.sqrt(alpha)
	g.dy = 2 * half / f64(GRID - 1)
	mass: f64 = 0
	for i in 0 ..< GRID {
		g.y[i] = c - half + f64(i) * g.dy
		g.logp[i] = invgamma_logpdf_y(alpha, beta, g.y[i])
		mass += math.exp(g.logp[i]) * g.dy
	}
	for i in 0 ..< GRID do g.logp[i] -= math.ln(mass) // renormalize away grid error
	return
}

kl_to_gmm :: proc(g: ^Grid, q: Gmm_Row) -> f64 {
	kl: f64 = 0
	for i in 0 ..< GRID do kl += math.exp(g.logp[i]) * (g.logp[i] - gmm_logpdf(q, g.y[i])) * g.dy
	return kl
}

kl_to_invgamma :: proc(g: ^Grid, alpha, beta: f64) -> f64 {
	kl: f64 = 0
	for i in 0 ..< GRID do kl += math.exp(g.logp[i]) * (g.logp[i] - invgamma_logpdf_y(alpha, beta, g.y[i])) * g.dy
	return kl
}

// Best single Gaussian under KL(p ‖ q): match p's mean and variance.
kl_to_moment_gaussian :: proc(g: ^Grid) -> f64 {
	m, v: f64 = 0, 0
	for i in 0 ..< GRID do m += math.exp(g.logp[i]) * g.y[i] * g.dy
	for i in 0 ..< GRID do v += math.exp(g.logp[i]) * (g.y[i] - m) * (g.y[i] - m) * g.dy
	kl: f64 = 0
	for i in 0 ..< GRID {
		u := (g.y[i] - m) / math.sqrt(v)
		logq := -0.5 * u * u - 0.5 * math.ln(v) - LOG_SQRT_2PI
		kl += math.exp(g.logp[i]) * (g.logp[i] - logq) * g.dy
	}
	return kl
}
