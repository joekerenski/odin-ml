package main

// ============================================================================
// M2 — amortized posterior on the paper's conjugate problem, with an MLP.
//
//   input   (log α, log β) and {z_1..z_n}   prior params + a SET of observations
//   output  K-component GMM over y = log σ²  the approximate posterior
//   loss    -log q(y_true)                  y_true is the σ² that made z
//
// Model (DeepSets): each z_i goes through one shared MLP φ, the embeddings are
// summed (order can't matter), then combined with the prior features:
//   h = relu(W_p·prior + W_s·Σ_i φ(z_i) + b) → MLP → GMM head
// The transformer (M3) keeps this shape and replaces the sum with attention.
//
// Minimizing that NLL over fresh problems = minimizing E[KL(exact ‖ q)] up to a
// constant (paper, Prop. 3.1). The exact posterior's NLL on the same batch is
// printed too: the gap between the two IS the remaining expected KL.
//
// Eval: KL(exact ‖ q) on 1000 held-out problems, by quadrature. For scale:
//   - the best single Gaussian (moment-matched to the exact posterior)
//   - the prior alone (ignoring the data)
//
//   make dt-conjugate
// ============================================================================

import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"
import "core:time"
import dt ".."
import ml "../../ml"

N_OBS :: 10
K :: 5
HIDDEN :: 256
BATCH :: 1024
STEPS :: 12000
WARMUP :: 100
LR :: 1e-3
N_TEST :: 1000

EMBED :: 64

Model :: struct {
	phi1, phi2:    ml.Linear, // per-observation encoder φ: 1 → EMBED → EMBED
	prior, pooled: ml.Linear, // → HIDDEN (their sum = a linear layer on the concat)
	l2, l3:        ml.Linear,
	head:          dt.Gmm_Head,
}

model_init :: proc() -> (m: Model) {
	m.phi1 = ml.linear(1, EMBED)
	m.phi2 = ml.linear(EMBED, EMBED)
	m.prior = ml.linear(2, HIDDEN)
	m.pooled = ml.linear(EMBED, HIDDEN)
	m.l2 = ml.linear(HIDDEN, HIDDEN)
	m.l3 = ml.linear(HIDDEN, HIDDEN)
	m.head = dt.gmm_head(HIDDEN, K)
	return
}

model_params :: proc(m: Model) -> (ps: [dynamic]^ml.Tensor) {
	for l in ([]ml.Linear{m.phi1, m.phi2, m.prior, m.pooled, m.l2, m.l3}) do ml.linear_params(&ps, l)
	dt.gmm_head_params(&ps, m.head)
	return
}

// prior [B, 2], z [B, n, 1]
forward :: proc(m: ^Model, prior, z: ^ml.Tensor) -> dt.Gmm {
	e := ml.relu(ml.linear_forward(&m.phi1, z)) // [B, n, EMBED], shared weights
	e = ml.relu(ml.linear_forward(&m.phi2, e))
	pooled := ml.reshape(ml.sum(e, 1), {z.shape[0], EMBED}) // Σ_i over the set
	h := ml.relu(ml.add(ml.linear_forward(&m.prior, prior), ml.linear_forward(&m.pooled, pooled)))
	h = ml.relu(ml.linear_forward(&m.l2, h))
	h = ml.relu(ml.linear_forward(&m.l3, h))
	return dt.gmm_head_forward(&m.head, h)
}

// prior [B, 2], z [B, n, 1], targets y [B, 1] for a batch of problems.
batch_tensors :: proc(ps: []dt.Problem) -> (prior, z, y: ^ml.Tensor) {
	B := i32(len(ps))
	prior = ml.new_tensor({B, 2})
	z = ml.new_tensor({B, N_OBS, 1})
	y = ml.new_tensor({B, 1})
	for p, b in ps {
		prior.data[2 * b] = f32(math.ln(p.alpha))
		prior.data[2 * b + 1] = f32(math.ln(p.beta))
		for v, i in p.z do z.data[b * N_OBS + i] = f32(v)
		y.data[b] = f32(p.y)
	}
	return
}

sample_batch :: proc(n: int) -> []dt.Problem {
	ps := make([]dt.Problem, n)
	for &p in ps do p = dt.sample_problem(dt.META_PRIOR, N_OBS)
	return ps
}

// Mean NLL of the true y under the exact posterior: the floor for the model's loss.
exact_nll :: proc(ps: []dt.Problem) -> f64 {
	s: f64 = 0
	for p in ps {
		a, b := dt.posterior(p)
		s -= dt.invgamma_logpdf_y(a, b, p.y)
	}
	return s / f64(len(ps))
}

main :: proc() {
	fmt.println("=== M2: amortized posterior, InvGamma prior on σ², MLP → GMM ===")
	ml.warn_if_unoptimized()
	ml.seed(0)

	m := model_init()
	params := model_params(m)
	n_params := 0
	for p in params do n_params += len(p.data)
	fmt.printfln("n=%d obs, K=%d components, DeepSets φ %d + MLP %d×3, %d params", N_OBS, K, EMBED, HIDDEN, n_params)

	opt := ml.new_adam(params[:], lr = LR)
	arena: mem.Dynamic_Arena
	mem.dynamic_arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)
	heap := context.allocator

	// ---- train on fresh problems every step ----
	t0 := time.tick_now()
	run_nll, run_exact: f64
	for step in 0 ..< STEPS {
		context.allocator = mem.dynamic_arena_allocator(&arena)
		ps := sample_batch(BATCH)
		prior, z, y := batch_tensors(ps)
		loss := dt.gmm_nll(forward(&m, prior, z), y)
		ml.backward(loss)
		ml.optimizer_set_lr(opt, ml.cosine_lr(step, STEPS, WARMUP, LR))
		ml.adam_step(opt)
		run_nll += f64(ml.item(loss))
		run_exact += exact_nll(ps)
		if (step + 1) % 1200 == 0 {
			n := 1200.0
			fmt.printfln("  step %4d  nll %.4f   exact %.4f   gap (≈ E[KL]) %.4f   %.1fs",
				step + 1, run_nll / n, run_exact / n, (run_nll - run_exact) / n, time.duration_seconds(time.tick_since(t0)))
			run_nll, run_exact = 0, 0
		}
		context.allocator = heap
		mem.dynamic_arena_reset(&arena)
		ml.clear_grads(..params[:])
	}

	// ---- eval: KL(exact ‖ ·) on held-out problems ----
	ml.seed(12345)
	context.allocator = mem.dynamic_arena_allocator(&arena)
	ps := sample_batch(N_TEST)
	prior, z, _ := batch_tensors(ps)
	q := forward(&m, prior, z)
	ml.realize(q.logits); ml.realize(q.means); ml.realize(q.log_stds)

	kl_model := make([]f64, N_TEST)
	kl_gauss := make([]f64, N_TEST)
	kl_prior := make([]f64, N_TEST)
	for p, b in ps {
		a, bb := dt.posterior(p)
		grid := new(dt.Grid)
		grid^ = dt.invgamma_grid(a, bb)
		kl_model[b] = dt.kl_to_gmm(grid, dt.gmm_row(q, b))
		kl_gauss[b] = dt.kl_to_moment_gaussian(grid)
		kl_prior[b] = dt.kl_to_invgamma(grid, p.alpha, p.beta)
	}
	report :: proc(name: string, kl: []f64) {
		mean := math.sum(kl) / f64(len(kl))
		slice.sort(kl)
		fmt.printfln("  %-30s mean %.5f   median %.5f   p95 %.5f", name, mean, kl[len(kl) / 2], kl[len(kl) * 95 / 100])
	}
	fmt.printfln("\nKL(exact posterior ‖ ·) on %d held-out problems:", N_TEST)
	report(fmt.tprintf("DeepSets MLP → GMM (K=%d)", K), kl_model)
	report("best single Gaussian", kl_gauss)
	report("prior (ignores data)", kl_prior)

	// ---- one example ----
	p := ps[0]
	a, bb := dt.posterior(p)
	w, mu, sd: [K]f64
	dt.gmm_components(dt.gmm_row(q, 0), w[:], mu[:], sd[:])
	fmt.printfln("\nexample: prior InvGamma(%.2f, %.2f), n=%d, true y=log σ²=%.3f", p.alpha, p.beta, N_OBS, p.y)
	fmt.printfln("  exact posterior  InvGamma(%.2f, %.2f): mode of y at %.3f", a, bb, math.ln(bb / a))
	for k in 0 ..< K do fmt.printfln("  component %d  w=%.3f  μ=%+.3f  σ=%.3f", k, w[k], mu[k], sd[k])

	context.allocator = heap
}
