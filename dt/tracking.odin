package dt

// ============================================================================
// Bayesian sensor fusion (paper, Sec. 4.3.1): track a 2D target, state
// x = (px, vx, py, vy), from two non-linear, non-Gaussian sensors. Settings
// follow the reference implementation (experiments/configs/lti_filter.yaml).
//
//   dynamics   x' = A x + G n,  n ~ N(0, I₂)        (linear: the predict step
//                                                     maps a GMM to a GMM exactly)
//   rangefinder  r = |(px, py)|:  0.7 N(r, (0.1(r+1))²) + 0.2 Exp(1) + 0.1 U(0, 20),
//                clamped to [0, 20]  (true echo / clutter / sensor failure)
//   bearing      atan2(px, py) + N(0, 0.1²), wrapped to [−π, π)
//
// Training problems (one DT update): prior = random 4-component GMM from the
// reference's conjugate meta-prior, x ~ prior, one reading from each sensor.
// Filtering: posterior_t = DT(prior_t, z_t); prior_{t+1} = predict(posterior_t).
// ============================================================================

import "core:math"
import "core:math/linalg"
import "core:math/rand"

K_FUSION :: 4

// The M4 model (dt/fusion) and where its trained weights live.
FUSION_MODEL :: Dt_Config{k = K_FUSION, dim = 64, heads = 8, layers = 6, mlp = 256, gelu = true}
FUSION_CHECKPOINT :: MODELS_DIR + "/fusion_dt4.safetensors"

DYN_A :: Mat{
	1, 0.6321, 0, 0,
	0, 0.3679, 0, 0,
	0, 0, 1, 0.6321,
	0, 0, 0, 0.3679,
}
DYN_G :: matrix[D, 2]f64{
	0.1, 0,
	0.2, 0,
	0, 0.1,
	0, 0.2,
}
X0_MEAN :: Vec{1, 0, 1, 0}

RANGE_SCALE :: 0.1
RANGE_RATE :: 1.0
RANGE_MAX :: 20.0
RANGE_W :: [3]f64{0.7, 0.2, 0.1} // gaussian echo, exponential clutter, uniform failure
BEARING_STD :: 0.1

// Observation tokens: [range·1, bearing·1, is_range, is_bearing] — one shared
// MLP, the sensor type as a one-hot.
N_OBS :: 2
OBS_FEAT :: 4

Echo :: enum u8 {
	True,    // the target
	Clutter, // an early echo (exponential)
	Failure, // sensor failure (uniform over the range)
}

sample_range :: proc(x: Vec, gen := context.random_generator) -> f64 {
	z, _ := sample_range_kind(x, gen)
	return z
}

sample_range_kind :: proc(x: Vec, gen := context.random_generator) -> (z: f64, kind: Echo) {
	r := math.sqrt(x[0] * x[0] + x[2] * x[2])
	u := rand.float64(gen)
	switch {
	case u < RANGE_W[0]: z, kind = rand.float64_normal(r, RANGE_SCALE * (r + 1), gen), .True
	case u < RANGE_W[0] + RANGE_W[1]: z, kind = rand.float64_exponential(RANGE_RATE, gen), .Clutter
	case: z, kind = rand.float64_uniform(0, RANGE_MAX, gen), .Failure
	}
	return clamp(z, 0, RANGE_MAX), kind
}

range_log_lik :: proc(z: f64, x: Vec) -> f64 {
	r := math.sqrt(x[0] * x[0] + x[2] * x[2])
	s := RANGE_SCALE * (r + 1)
	u := (z - r) / s
	p := RANGE_W[0] * math.exp(-0.5 * u * u - LOG_SQRT_2PI) / s
	p += RANGE_W[1] * RANGE_RATE * math.exp(-RANGE_RATE * z)
	p += RANGE_W[2] / RANGE_MAX
	return math.ln(p)
}

wrap_angle :: proc(a: f64) -> f64 {
	return a - 2 * math.PI * math.floor((a + math.PI) / (2 * math.PI))
}

sample_bearing :: proc(x: Vec, gen := context.random_generator) -> f64 {
	return wrap_angle(math.atan2(x[0], x[2]) + rand.float64_normal(0, BEARING_STD, gen))
}

bearing_log_lik :: proc(z: f64, x: Vec) -> f64 {
	d := wrap_angle(z - math.atan2(x[0], x[2]))
	u := d / BEARING_STD
	return -0.5 * u * u - math.ln(f64(BEARING_STD)) - LOG_SQRT_2PI
}

obs_features :: proc(z_range, z_bearing: f64, out: []f32) {
	out[0], out[1], out[2], out[3] = f32(z_range), 0, 1, 0
	out[4], out[5], out[6], out[7] = 0, f32(z_bearing), 0, 1
}

// ---- meta-prior over GMM priors (reference: GaussianMixtureModelConjugateMetaPrior)
// w ~ Dirichlet(1/K), μ_k ~ N(0, I + 11ᵀ), Λ_k ~ Wishart(df = D+1, 4I) + 1e-4 I

// Cholesky of I + 11ᵀ
mu_chol :: proc() -> Mat {
	c := linalg.identity(Mat)
	for i in 0 ..< D do for j in 0 ..< D do c[i, j] += 1
	L, _ := cholesky(c)
	return L
}

std_normal_vec :: proc(gen := context.random_generator) -> (v: Vec) {
	for j in 0 ..< D do v[j] = rand.float64_normal(0, 1, gen)
	return
}

sample_prior :: proc(out: Mixture, mu_L: Mat, gen := context.random_generator) {
	wsum: f64 = 0
	for &c in out {
		c.w = rand.float64_gamma(1.0 / f64(len(out)), 1, gen)
		wsum += c.w
		c.mu = mu_L * std_normal_vec(gen)
		// Bartlett: Λ = L A Aᵀ Lᵀ with L = chol(4I) = 2I, A lower, A_ii² ~ χ²(df − i)
		A: Mat
		for i in 0 ..< D {
			A[i, i] = math.sqrt(rand.float64_gamma(0.5 * f64(D + 1 - i), 2, gen))
			for j in 0 ..< i do A[i, j] = rand.float64_normal(0, 1, gen)
		}
		prec := 4 * A * linalg.transpose(A) + 1e-4 * linalg.identity(Mat)
		c.U, _ = cholesky(prec)
	}
	for &c in out do c.w /= wsum
}

// x ~ mixture: pick a component, x = μ + U⁻ᵀ ε.
sample_mixture :: proc(mix: Mixture, gen := context.random_generator) -> Vec {
	u := rand.float64(gen)
	k := len(mix) - 1
	for c, i in mix {
		if u < c.w {
			k = i
			break
		}
		u -= c.w
	}
	c := mix[k]
	return c.mu + linalg.inverse(linalg.transpose(c.U)) * std_normal_vec(gen)
}

// ---- dynamics ----------------------------------------------------------------

process_cov :: proc() -> Mat {
	return DYN_G * linalg.transpose(DYN_G)
}

step_state :: proc(x: Vec, gen := context.random_generator) -> Vec {
	n := [2]f64{rand.float64_normal(0, 1, gen), rand.float64_normal(0, 1, gen)}
	return DYN_A * x + DYN_G * n
}

// Exact predict step for a GMM: μ → Aμ, Σ → AΣAᵀ + Q per component.
predict_mixture :: proc(mix: Mixture) {
	Q := process_cov()
	for &c in mix {
		cov := DYN_A * covariance(c) * linalg.transpose(DYN_A) + Q
		cov = 0.5 * (cov + linalg.transpose(cov))
		c, _ = component_from_cov(c.w, DYN_A * c.mu, cov)
	}
}

// The filter's initial prior, as in the reference: K copies of N(x0, I) with
// weights ∝ 1..K (identical components, so only the weights tell them apart).
initial_prior :: proc(out: Mixture) {
	n := f64(len(out) * (len(out) + 1) / 2)
	for &c, k in out do c = {f64(k + 1) / n, X0_MEAN, linalg.identity(Mat)}
}

// ---- test series ---------------------------------------------------------------

Series :: struct {
	x:               []Vec, // true states
	z_range, z_bear: []f64, // readings
	echo:            []Echo, // what the rangefinder actually saw (for display)
}

sample_series :: proc(T: int, gen := context.random_generator) -> (s: Series) {
	s.x = make([]Vec, T)
	s.z_range = make([]f64, T)
	s.z_bear = make([]f64, T)
	s.echo = make([]Echo, T)
	x := X0_MEAN + std_normal_vec(gen)
	for t in 0 ..< T {
		if t > 0 do x = step_state(x, gen)
		s.x[t] = x
		s.z_range[t], s.echo[t] = sample_range_kind(x, gen)
		s.z_bear[t] = sample_bearing(x, gen)
	}
	return
}
