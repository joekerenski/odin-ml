package main

// ============================================================================
// M4 — Bayesian sensor fusion (paper, Sec. 4.3.1, Table 3).
//
// A Distribution Transformer learns ONE Bayesian update: a 4-component,
// full-covariance GMM prior over the 4D state plus one rangefinder and one
// bearing reading → the GMM posterior. Because the output is again a GMM, and
// the dynamics are linear, it runs as a filter: update with the DT, predict
// exactly, feed the result back in as the next prior, 100 times.
//
//   make dt-fusion                                  load models/fusion_dt4 (or train) + evaluate
//   odin run dt/fusion -o:speed -- train [steps]    retrain and overwrite the checkpoint
//
// Evaluation:
//   1. one update, 1000 held-out problems: E[KL(exact posterior ‖ q)], using the
//      true x (a draw from the exact posterior) and an importance-sampled
//      evidence p(z | prior)
//   2. filtering, 100 series × 100 steps: mean −log q_t(x_t) against bootstrap
//      particle filters (Gaussian fit to the particles, the paper's protocol)
//
// Paper, Table 3: DT −0.197 ± 0.040, particle filter (5000) −0.244 ± 0.047, EKF 95.9.
// ============================================================================

import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:strconv"
import "core:time"
import dt ".."
import ml "../../ml"

D :: dt.D
K :: dt.K_FUSION
Vec :: dt.Vec
Mat :: dt.Mat

CHECKPOINT :: dt.FUSION_CHECKPOINT
MODEL :: dt.FUSION_MODEL

Config :: struct {
	batch, steps, warmup, log_every: int,
	lr:                              f32,
}

Batch :: struct {
	comps: ^ml.Tensor, // [B, K, N_FEAT]   prior components
	z:     ^ml.Tensor, // [B, 2, OBS_FEAT] sensor readings
	x:     ^ml.Tensor, // [B, 1, D]        true state
	// host copies for evaluation
	priors: []dt.Component, // B·K
	xs:     []Vec,
	zr, zb: []f64,
}

make_batch :: proc(size: int) -> (b: Batch) {
	B := i32(size)
	b.comps = ml.new_tensor({B, K, dt.N_FEAT})
	b.z = ml.new_tensor({B, dt.N_OBS, dt.OBS_FEAT})
	b.x = ml.new_tensor({B, 1, D})
	b.priors = make([]dt.Component, size * K)
	b.xs = make([]Vec, size)
	b.zr, b.zb = make([]f64, size), make([]f64, size)
	mu_L := dt.mu_chol()
	for i in 0 ..< size {
		prior := b.priors[i * K:][:K]
		dt.sample_prior(prior, mu_L)
		x := dt.sample_mixture(prior)
		b.xs[i] = x
		b.zr[i] = dt.sample_range(x)
		b.zb[i] = dt.sample_bearing(x)
		dt.fusion_inputs(b.comps, b.z, i, prior, b.zr[i], b.zb[i])
		for j in 0 ..< D do b.x.data[i * D + j] = f32(x[j])
	}
	return
}



step :: proc(m: ^dt.Dt_Full, b: Batch) -> (loss, post_nll: ^ml.Tensor) {
	q_prior, q_post := dt.dt_full_forward(m, b.comps, b.z)
	post_nll = dt.gmm_full_nll(q_post, b.x, m.maps)
	return ml.add(post_nll, dt.gmm_full_nll(q_prior, b.x, m.maps)), post_nll
}



train :: proc(m: ^dt.Dt_Full, params: []^ml.Tensor, cfg: Config) -> (seconds: f64, last_nll: f64) {
	opt := ml.new_adam(params, lr = cfg.lr)
	arena: mem.Dynamic_Arena
	ml.arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)
	heap := context.allocator

	t0 := time.tick_now()
	run_post, run_prior: f64
	for i in 0 ..< cfg.steps {
		context.allocator = mem.dynamic_arena_allocator(&arena)
		b := make_batch(cfg.batch)
		loss, post_nll := step(m, b)
		ml.backward(loss)
		ml.optimizer_set_lr(opt, ml.cosine_lr(i, cfg.steps, cfg.warmup, cfg.lr))
		ml.adam_step(opt)
		p := f64(ml.item(post_nll))
		run_post += p
		run_prior += f64(ml.item(loss)) - p
		if math.is_nan(p) {
			fmt.printfln("  step %d: NaN loss, stopping", i + 1)
			break
		}
		if (i + 1) % cfg.log_every == 0 {
			n := f64(cfg.log_every)
			last_nll = run_post / n
			fmt.printfln("  step %5d  posterior nll %+.4f   prior nll %+.4f   %.1fs",
				i + 1, last_nll, run_prior / n, time.duration_seconds(time.tick_since(t0)))
			run_post, run_prior = 0, 0
		}
		context.allocator = heap
		mem.dynamic_arena_reset(&arena)
		ml.clear_grads(..params)
	}
	return time.duration_seconds(time.tick_since(t0)), last_nll
}

// ---- 1. one update vs the exact posterior ---------------------------------------

N_IS :: 20000 // prior samples per problem for the evidence p(z | prior)

Exact_Job :: struct {
	b:                               Batch,
	log_evidence:                    []f64,
	gauss_nll:                       []f64, // −log N(x; posterior mean, posterior cov)
}

// log p(x | z) = log p(x) + log p(z | x) − log p(z), p(z) = E_prior[p(z | x)].
exact_range :: proc(data: rawptr, lo, hi: int) {
	job := (^Exact_Job)(data)
	b := job.b
	for i in lo ..< hi {
		state := rand.create(u64(1_000_003 * (i + 1)))
		gen := rand.default_random_generator(&state)
		prior := b.priors[i * K:][:K]
		lw := make([]f64, N_IS)
		xs := make([]Vec, N_IS)
		defer delete(lw)
		defer delete(xs)
		mx := math.inf_f64(-1)
		for s in 0 ..< N_IS {
			xs[s] = dt.sample_mixture(prior, gen)
			lw[s] = dt.range_log_lik(b.zr[i], xs[s]) + dt.bearing_log_lik(b.zb[i], xs[s])
			mx = max(mx, lw[s])
		}
		tot: f64 = 0
		mean: Vec
		for s in 0 ..< N_IS {
			lw[s] = math.exp(lw[s] - mx)
			tot += lw[s]
			mean += lw[s] * xs[s]
		}
		mean /= tot
		cov: Mat
		for s in 0 ..< N_IS {
			r := xs[s] - mean
			cov += (lw[s] / tot) * linalg.outer_product(r, r)
		}
		job.log_evidence[i] = mx + math.ln(tot / N_IS)
		c, ok := dt.component_from_cov(1, mean, cov + 1e-9 * linalg.identity(Mat))
		job.gauss_nll[i] = ok ? -dt.component_log_pdf(c, b.xs[i]) : math.inf_f64(1)
	}
}

evaluate_update :: proc(m: ^dt.Dt_Full, n: int) -> (kl: f64) {
	arena: mem.Dynamic_Arena
	ml.arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)
	context.allocator = mem.dynamic_arena_allocator(&arena)

	rand.reset(777)
	b := make_batch(n)
	q := dt.dt_full_posterior(m, b.comps, b.z)
	job := Exact_Job{b, make([]f64, n), make([]f64, n)}
	ml.parallel_for(n, 8, exact_range, &job)

	exact, model, prior, gauss: f64
	mix := make([]dt.Component, K)
	for i in 0 ..< n {
		x := b.xs[i]
		pr := b.priors[i * K:][:K]
		log_prior := dt.mixture_log_pdf(pr, x)
		exact -= log_prior + dt.range_log_lik(b.zr[i], x) + dt.bearing_log_lik(b.zb[i], x) - job.log_evidence[i]
		dt.gmm_full_row(q, i, mix)
		model -= dt.mixture_log_pdf(mix, x)
		prior -= log_prior
		gauss += job.gauss_nll[i]
	}
	N := f64(n)
	fmt.printfln("\nOne update, %d held-out problems (E[−log q(x | prior, z)]; gap to exact = E[KL]):", n)
	fmt.printfln("  exact posterior (IS, %d samples)   %+.4f", N_IS, exact / N)
	fmt.printfln("  DT-%d                               %+.4f   E[KL] %.4f", K, model / N, (model - exact) / N)
	fmt.printfln("  best single Gaussian               %+.4f   E[KL] %.4f", gauss / N, (gauss - exact) / N)
	fmt.printfln("  prior (ignores readings)           %+.4f   E[KL] %.4f", prior / N, (prior - exact) / N)
	return (model - exact) / N
}

// ---- 2. filtering -------------------------------------------------------------

SERIES :: 100
LENGTH :: 100

Nll_Stats :: struct {
	sum, sum_sq: f64,
	n:           int,
	by_t:        [LENGTH]f64, // sums per time step
}

add_nll :: proc(s: ^Nll_Stats, v: f64, t: int) {
	s.sum += v
	s.sum_sq += v * v
	s.n += 1
	s.by_t[t] += v
}

// mean NLL over time steps [lo, hi)
window :: proc(s: Nll_Stats, lo, hi: int) -> f64 {
	acc: f64 = 0
	for t in lo ..< hi do acc += s.by_t[t]
	return acc / f64((hi - lo) * s.n / LENGTH)
}

report_nll :: proc(name: string, s: Nll_Stats, ms_per_step: f64) {
	mean := s.sum / f64(s.n)
	sd := math.sqrt(max(s.sum_sq / f64(s.n) - mean * mean, 0))
	fmt.printfln("  %-34s %+8.3f ± %.3f   %8.3f ms/step", name, mean, 1.96 * sd / math.sqrt(f64(s.n)), ms_per_step)
}

Pf_Job :: struct {
	series:      []dt.Series,
	n_particles: int,
	nll:         [][]f64, // [series][t]
}

pf_range :: proc(data: rawptr, lo, hi: int) {
	job := (^Pf_Job)(data)
	for si in lo ..< hi {
		s := job.series[si]
		pf := dt.pf_make(job.n_particles, u64(31 * (si + 1)))
		defer dt.pf_destroy(pf)
		for t in 0 ..< len(s.x) {
			if t > 0 do dt.pf_predict(pf)
			mean, cov := dt.pf_update(pf, s.z_range[t], s.z_bear[t])
			c, ok := dt.component_from_cov(1, mean, cov + 1e-3 * linalg.identity(Mat)) // the paper's jitter
			job.nll[si][t] = ok ? -dt.component_log_pdf(c, s.x[t]) : math.inf_f64(1)
		}
	}
}

run_pf :: proc(series: []dt.Series, n_particles: int) -> (st: Nll_Stats, ms: f64) {
	job := Pf_Job{series, n_particles, make([][]f64, len(series))}
	for &r in job.nll do r = make([]f64, LENGTH)
	t0 := time.tick_now()
	ml.parallel_for(len(series), 1, pf_range, &job)
	ms = time.duration_milliseconds(time.tick_since(t0)) / LENGTH
	for r in job.nll do for v, t in r do add_nll(&st, v, t)
	return
}

evaluate_filter :: proc(m: ^dt.Dt_Full) -> (dt_nll: f64) {
	rand.reset(2024)
	series := make([]dt.Series, SERIES)
	for &s in series do s = dt.sample_series(LENGTH)

	// DT filter, all series batched: update (DT) → score → predict (exact)
	priors := make([]dt.Component, SERIES * K)
	for i in 0 ..< SERIES do dt.initial_prior(priors[i * K:][:K])
	arena: mem.Dynamic_Arena
	ml.arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)
	heap := context.allocator
	st, st_pred: Nll_Stats
	dt_ms: f64
	for t in 0 ..< LENGTH {
		context.allocator = mem.dynamic_arena_allocator(&arena)
		t0 := time.tick_now()
		zr, zb := make([]f64, SERIES), make([]f64, SERIES)
		for s, i in series do zr[i], zb[i] = s.z_range[t], s.z_bear[t]
		dt.dt_update(m, priors, zr, zb, priors)
		for i in 0 ..< SERIES {
			mix := priors[i * K:][:K]
			add_nll(&st, -dt.mixture_log_pdf(mix, series[i].x[t]), t)
			dt.predict_mixture(mix)
		}
		dt_ms += time.duration_milliseconds(time.tick_since(t0))
		context.allocator = heap
		mem.dynamic_arena_reset(&arena)
	}

	// no readings at all: the prior pushed through the dynamics
	pred := dt.Component{1, dt.X0_MEAN, linalg.identity(Mat)}
	for t in 0 ..< LENGTH {
		for s in series do add_nll(&st_pred, -dt.component_log_pdf(pred, s.x[t]), t)
		p := [1]dt.Component{pred}
		dt.predict_mixture(p[:])
		pred = p[0]
	}

	fmt.printfln("\nFiltering, %d series × %d steps (mean −log q_t(x_t) ± 95%% CI; time per step, all series):", SERIES, LENGTH)
	report_nll(fmt.tprintf("DT-%d (update) + exact predict", K), st, dt_ms / LENGTH)
	pfs: [3]Nll_Stats
	for n, i in ([]int{1000, 5000, 50000}) {
		ms: f64
		pfs[i], ms = run_pf(series, n)
		report_nll(fmt.tprintf("particle filter %d, Gaussian fit", n), pfs[i], ms)
	}
	report_nll("dynamics only (no readings)", st_pred, 0)

	fmt.println("\nby time step (does error build up along the sequence?):")
	fmt.println("                         t=0      1-9    10-29    30-59    60-99")
	rows := []struct {
		name: string,
		s:    Nll_Stats,
	}{{fmt.tprintf("DT-%d", K), st}, {"particle filter 5000", pfs[1]}, {"particle filter 50000", pfs[2]}}
	for r in rows {
		fmt.printfln("  %-21s %+7.3f  %+7.3f  %+7.3f  %+7.3f  %+7.3f", r.name,
			window(r.s, 0, 1), window(r.s, 1, 10), window(r.s, 10, 30), window(r.s, 30, 60), window(r.s, 60, 100))
	}
	return st.sum / f64(st.n)
}

main :: proc() {
	cfg := Config{batch = 1024, steps = 20000, warmup = 500, log_every = 500, lr = 1e-3}
	pos, retrain := dt.args()
	if len(pos) > 0 do cfg.steps = strconv.parse_int(pos[0]) or_else cfg.steps
	cfg.log_every = max(1, min(cfg.log_every, cfg.steps / 10))

	fmt.println("=== M4: Bayesian sensor fusion, DT update inside a filter ===")
	ml.warn_if_unoptimized()
	ml.setup_from_env()
	ml.seed(0)

	m := dt.dt_full_model(MODEL, dt.OBS_FEAT)
	params := dt.dt_full_params(m)
	fmt.printfln("%d layers, d=%d, %d heads, MLP %d (GELU), K=%d full-covariance components over %dD: %d params",
		MODEL.layers, MODEL.dim, MODEL.heads, MODEL.mlp, K, D, dt.count_params(params[:]))

	if dt.try_load(CHECKPOINT, params[:], retrain) {
		evaluate_update(&m, 1000)
		evaluate_filter(&m)
		return
	}
	fmt.printfln("training: batch %d, %d steps, Adam lr %g, cosine (warmup %d); device %v",
		cfg.batch, cfg.steps, cfg.lr, cfg.warmup, ml.get_device())
	secs, nll := train(&m, params[:], cfg)
	meta: [dynamic]ml.Meta
	append(&meta,
		ml.Meta{"experiment", "sensor fusion, paper 4.3.1 (M4)"},
		ml.Meta{"arch", fmt.tprintf("DT-full k=%d dim=%d heads=%d layers=%d mlp=%d gelu=%v", MODEL.k, MODEL.dim, MODEL.heads, MODEL.layers, MODEL.mlp, MODEL.gelu)},
		ml.Meta{"train", fmt.tprintf("batch=%d steps=%d lr=%g warmup=%d", cfg.batch, cfg.steps, cfg.lr, cfg.warmup)},
		ml.Meta{"train_seconds", fmt.tprintf("%.0f on %v", secs, ml.get_device())},
		ml.Meta{"final_posterior_nll", fmt.tprintf("%.4f", nll)},
	)
	dt.save_model(CHECKPOINT, params[:], meta[:]) // keep the weights even if eval fails
	kl := evaluate_update(&m, 1000)
	filter_nll := evaluate_filter(&m)
	append(&meta, ml.Meta{"eval_update_kl", fmt.tprintf("%.4f", kl)}, ml.Meta{"eval_filter_nll", fmt.tprintf("%.4f", filter_nll)})
	dt.save_model(CHECKPOINT, params[:], meta[:])
}
