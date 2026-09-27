package dt

// ============================================================================
// Filters for the tracking problem, for the M4 evaluation and any other user
// of a trained model: the DT update (batched over independent tracks) and a bootstrap
// particle filter. The predict step for mixtures is predict_mixture.
// ============================================================================

import "core:math"
import "core:math/linalg"
import "core:math/rand"
import ml "../ml"

// Posterior mixture of the DT for a batch of (prior, readings); graph realized.
dt_full_posterior :: proc(m: ^Dt_Full, comps, z: ^ml.Tensor) -> Gmm_Full {
	tokens := dt_full_prior_tokens(m, comps)
	q := dt_full_unembed(m, trunk_forward(&m.trunk, tokens, dt_full_obs_tokens(m, z)))
	ml.realize_all({q.logits, q.means, q.log_diag, q.lower})
	return q
}

// Network inputs for one track: its prior components and its two readings.
fusion_inputs :: proc(comps, z: ^ml.Tensor, i: int, prior: Mixture, z_range, z_bearing: f64) {
	for c, k in prior do component_features(c, comps.data[(i * len(prior) + k) * N_FEAT:][:N_FEAT])
	obs_features(z_range, z_bearing, z.data[i * N_OBS * OBS_FEAT:][:N_OBS * OBS_FEAT])
}

// One Bayesian update by the DT for B tracks: priors [B·K] → posteriors [B·K]
// (may alias). Temporaries come from context.allocator (use an arena).
dt_update :: proc(m: ^Dt_Full, priors: []Component, z_range, z_bearing: []f64, out: []Component) {
	B, K := len(z_range), len(priors) / len(z_range)
	comps := ml.new_tensor({i32(B), i32(K), N_FEAT})
	z := ml.new_tensor({i32(B), N_OBS, OBS_FEAT})
	for i in 0 ..< B do fusion_inputs(comps, z, i, priors[i * K:][:K], z_range[i], z_bearing[i])
	q := dt_full_posterior(m, comps, z)
	for i in 0 ..< B do gmm_full_row(q, i, out[i * K:][:K])
}

// ---- bootstrap particle filter -------------------------------------------------

Particle_Filter :: struct {
	ps, next: []Vec,
	lw:       []f64,
	state:    rand.Default_Random_State,
}

pf_make :: proc(n: int, seed: u64) -> ^Particle_Filter {
	pf := new(Particle_Filter)
	pf.ps, pf.next, pf.lw = make([]Vec, n), make([]Vec, n), make([]f64, n)
	pf.state = rand.create(seed)
	gen := rand.default_random_generator(&pf.state)
	for &p in pf.ps do p = X0_MEAN + std_normal_vec(gen)
	return pf
}

pf_destroy :: proc(pf: ^Particle_Filter) {
	delete(pf.ps)
	delete(pf.next)
	delete(pf.lw)
	free(pf)
}

pf_predict :: proc(pf: ^Particle_Filter) {
	gen := rand.default_random_generator(&pf.state)
	for &p in pf.ps do p = step_state(p, gen)
}

// Weight by both readings, return the weighted mean/covariance (the filter's
// Gaussian summary), then resample (systematic). pf.ps are then posterior draws.
pf_update :: proc(pf: ^Particle_Filter, z_range, z_bearing: f64) -> (mean: Vec, cov: Mat) {
	gen := rand.default_random_generator(&pf.state)
	N := len(pf.ps)
	mx := math.inf_f64(-1)
	for p, i in pf.ps {
		pf.lw[i] = range_log_lik(z_range, p) + bearing_log_lik(z_bearing, p)
		mx = max(mx, pf.lw[i])
	}
	tot: f64 = 0
	for p, i in pf.ps {
		pf.lw[i] = math.exp(pf.lw[i] - mx)
		tot += pf.lw[i]
		mean += pf.lw[i] * p
	}
	mean /= tot
	for p, i in pf.ps {
		r := p - mean
		cov += (pf.lw[i] / tot) * linalg.outer_product(r, r)
	}
	u := rand.float64(gen) / f64(N)
	acc := pf.lw[0] / tot
	j := 0
	for i in 0 ..< N {
		for u > acc && j < N - 1 {
			j += 1
			acc += pf.lw[j] / tot
		}
		pf.next[i] = pf.ps[j]
		u += 1 / f64(N)
	}
	pf.ps, pf.next = pf.next, pf.ps
	return
}
