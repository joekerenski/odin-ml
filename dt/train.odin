package dt

// ============================================================================
// Shared harness for the conjugate experiments: batches of fresh problems,
// the training loop, and evaluation against the exact posterior. A model plugs
// in two procs: step (its training loss) and posterior (its q for eval).
// ============================================================================

import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"
import "core:time"
import ml "../ml"

Batch :: struct {
	problems: []Problem,
	prior:    ^ml.Tensor, // [B, 2]     log α, log β
	z:        ^ml.Tensor, // [B, n, 1]  observations
	y:        ^ml.Tensor, // [B, 1]     true log σ²
}

make_batch :: proc(size, n_obs: int) -> (b: Batch) {
	b.problems = make([]Problem, size)
	for &p in b.problems do p = sample_problem(META_PRIOR, n_obs)
	B, n := i32(size), i32(n_obs)
	b.prior = ml.new_tensor({B, 2})
	b.z = ml.new_tensor({B, n, 1})
	b.y = ml.new_tensor({B, 1})
	for p, i in b.problems {
		b.prior.data[2 * i] = f32(math.ln(p.alpha))
		b.prior.data[2 * i + 1] = f32(math.ln(p.beta))
		for v, j in p.z do b.z.data[i * n_obs + j] = f32(v)
		b.y.data[i] = f32(p.y)
	}
	return
}

// Per-step arena with big blocks: after the first step every tensor comes from
// warm, already-mapped memory (small default blocks send big tensors to the
// heap, which maps and faults in fresh pages every step).
arena_init :: proc(a: ^mem.Dynamic_Arena) {
	mem.dynamic_arena_init(a, block_size = 64 * mem.Megabyte, out_band_size = 32 * mem.Megabyte)
}

// Mean NLL of the true y under the exact posterior: the floor for any model.
exact_nll :: proc(ps: []Problem) -> f64 {
	s: f64 = 0
	for p in ps {
		a, b := posterior(p)
		s -= invgamma_logpdf_y(a, b, p.y)
	}
	return s / f64(len(ps))
}

Config :: struct {
	n_obs, batch, steps, warmup, log_every, n_test: int,
	lr:                                             f32,
}

// Adam + cosine schedule on fresh problems every step. step returns the loss to
// minimize and the posterior NLL (printed next to the exact posterior's NLL:
// their gap is the model's remaining expected KL).
train :: proc(m: ^$M, params: []^ml.Tensor, cfg: Config, step: proc(m: ^M, b: Batch) -> (loss, post_nll: ^ml.Tensor)) {
	opt := ml.new_adam(params, lr = cfg.lr)
	arena: mem.Dynamic_Arena
	arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)
	heap := context.allocator

	t0 := time.tick_now()
	run_nll, run_exact: f64
	for i in 0 ..< cfg.steps {
		context.allocator = mem.dynamic_arena_allocator(&arena)
		b := make_batch(cfg.batch, cfg.n_obs)
		loss, post_nll := step(m, b)
		ml.backward(loss)
		ml.optimizer_set_lr(opt, ml.cosine_lr(i, cfg.steps, cfg.warmup, cfg.lr))
		ml.adam_step(opt)
		run_nll += f64(ml.item(post_nll))
		run_exact += exact_nll(b.problems)
		if (i + 1) % cfg.log_every == 0 {
			n := f64(cfg.log_every)
			fmt.printfln("  step %5d  nll %.4f   exact %.4f   gap (≈ E[KL]) %.4f   %.1fs",
				i + 1, run_nll / n, run_exact / n, (run_nll - run_exact) / n, time.duration_seconds(time.tick_since(t0)))
			run_nll, run_exact = 0, 0
		}
		context.allocator = heap
		mem.dynamic_arena_reset(&arena)
		ml.clear_grads(..params)
	}
}

// KL(exact ‖ ·) on held-out problems for the model and two references.
evaluate :: proc(m: ^$M, cfg: Config, posterior_q: proc(m: ^M, b: Batch) -> Gmm, label: string) {
	arena: mem.Dynamic_Arena
	arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)
	context.allocator = mem.dynamic_arena_allocator(&arena)

	ml.seed(12345)
	b := make_batch(cfg.n_test, cfg.n_obs)
	q := posterior_q(m, b)
	ml.realize(q.logits); ml.realize(q.means); ml.realize(q.log_stds)

	n := cfg.n_test
	kl_model, kl_gauss, kl_prior := make([]f64, n), make([]f64, n), make([]f64, n)
	grid := new(Grid)
	for p, i in b.problems {
		a, bb := posterior(p)
		grid^ = invgamma_grid(a, bb)
		kl_model[i] = kl_to_gmm(grid, gmm_row(q, i))
		kl_gauss[i] = kl_to_moment_gaussian(grid)
		kl_prior[i] = kl_to_invgamma(grid, p.alpha, p.beta)
	}
	fmt.printfln("\nKL(exact posterior ‖ ·) on %d held-out problems:", n)
	report_kl(label, kl_model)
	report_kl("best single Gaussian", kl_gauss)
	report_kl("prior (ignores data)", kl_prior)

	p := b.problems[0]
	a, bb := posterior(p)
	K := int(q.logits.shape[1])
	w, mu, sd := make([]f64, K), make([]f64, K), make([]f64, K)
	gmm_components(gmm_row(q, 0), w, mu, sd)
	fmt.printfln("\nexample: prior InvGamma(%.2f, %.2f), n=%d, true y=log σ²=%.3f", p.alpha, p.beta, cfg.n_obs, p.y)
	fmt.printfln("  exact posterior  InvGamma(%.2f, %.2f): mode of y at %.3f", a, bb, math.ln(bb / a))
	for k in 0 ..< K do fmt.printfln("  component %d  w=%.3f  μ=%+.3f  σ=%.3f", k, w[k], mu[k], sd[k])
}

report_kl :: proc(name: string, kl: []f64) {
	mean := math.sum(kl) / f64(len(kl))
	slice.sort(kl)
	fmt.printfln("  %-30s mean %.5f   median %.5f   p95 %.5f", name, mean, kl[len(kl) / 2], kl[len(kl) * 95 / 100])
}

// KL(exact prior ‖ q_prior): how well the prior tokens alone decode to the prior.
evaluate_prior_fit :: proc(m: ^$M, cfg: Config, prior_q: proc(m: ^M, b: Batch) -> Gmm) {
	arena: mem.Dynamic_Arena
	arena_init(&arena)
	defer mem.dynamic_arena_destroy(&arena)
	context.allocator = mem.dynamic_arena_allocator(&arena)

	ml.seed(12345)
	b := make_batch(cfg.n_test, cfg.n_obs)
	q := prior_q(m, b)
	ml.realize(q.logits); ml.realize(q.means); ml.realize(q.log_stds)
	kl := make([]f64, cfg.n_test)
	grid := new(Grid)
	for p, i in b.problems {
		grid^ = invgamma_grid(p.alpha, p.beta)
		kl[i] = kl_to_gmm(grid, gmm_row(q, i))
	}
	fmt.println("\nKL(exact prior ‖ prior tokens, unembedded):")
	report_kl("prior reconstruction", kl)
}
