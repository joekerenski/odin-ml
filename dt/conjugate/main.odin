package main

// ============================================================================
// M2 — amortized posterior on the paper's conjugate problem, DeepSets MLP.
//
//   input   (log α, log β) and {z_1..z_n}   prior params + a SET of observations
//   output  K-component GMM over y = log σ²  the approximate posterior
//   loss    -log q(y_true)                  y_true is the σ² that made z
//
// DeepSets: each z_i goes through one shared MLP φ, the embeddings are summed
// (order can't matter), then combined with the prior features:
//   h = relu(W_p·prior + W_s·Σ_i φ(z_i) + b) → MLP → GMM head
// The transformer (dt/table1, M3) keeps this shape and replaces the sum with
// attention.
//
// Minimizing that NLL over fresh problems = minimizing E[KL(exact ‖ q)] up to a
// constant (paper, Prop. 3.1): the printed gap to the exact posterior's NLL IS
// the remaining expected KL. Training/eval harness: dt/train.odin.
//
//   make dt-conjugate
// ============================================================================

import "core:fmt"
import dt ".."
import ml "../../ml"

K :: 5
EMBED :: 64
HIDDEN :: 256

CFG :: dt.Config{n_obs = 10, batch = 1024, steps = 12000, warmup = 100, log_every = 1200, n_test = 1000, lr = 1e-3}

Model :: struct {
	phi1, phi2:    ml.Linear, // per-observation encoder φ: 1 → EMBED → EMBED
	prior, pooled: ml.Linear, // → HIDDEN (their sum = a linear layer on the concat)
	l2, l3:        ml.Linear,
	head:          dt.Gmm_Head,
}

posterior_q :: proc(m: ^Model, b: dt.Batch) -> dt.Gmm {
	e := ml.relu(ml.linear_forward(&m.phi1, b.z)) // [B, n, EMBED], shared weights
	e = ml.relu(ml.linear_forward(&m.phi2, e))
	pooled := ml.reshape(ml.sum(e, 1), {b.z.shape[0], EMBED}) // Σ_i over the set
	h := ml.relu(ml.add(ml.linear_forward(&m.prior, b.prior), ml.linear_forward(&m.pooled, pooled)))
	h = ml.relu(ml.linear_forward(&m.l2, h))
	h = ml.relu(ml.linear_forward(&m.l3, h))
	return dt.gmm_head_forward(&m.head, h)
}

step :: proc(m: ^Model, b: dt.Batch) -> (loss, post_nll: ^ml.Tensor) {
	nll := dt.gmm_nll(posterior_q(m, b), b.y)
	return nll, nll
}

main :: proc() {
	fmt.println("=== M2: amortized posterior, InvGamma prior on σ², DeepSets MLP → GMM ===")
	ml.warn_if_unoptimized()
	ml.setup_from_env()
	ml.seed(0)

	m := Model{
		phi1   = ml.linear(1, EMBED),
		phi2   = ml.linear(EMBED, EMBED),
		prior  = ml.linear(2, HIDDEN),
		pooled = ml.linear(EMBED, HIDDEN),
		l2     = ml.linear(HIDDEN, HIDDEN),
		l3     = ml.linear(HIDDEN, HIDDEN),
		head   = dt.gmm_head(HIDDEN, K),
	}
	params: [dynamic]^ml.Tensor
	for l in ([]ml.Linear{m.phi1, m.phi2, m.prior, m.pooled, m.l2, m.l3}) do ml.linear_params(&params, l)
	dt.gmm_head_params(&params, m.head)
	fmt.printfln("n=%d obs, K=%d, DeepSets φ %d + MLP %d×3, %d params", CFG.n_obs, K, EMBED, HIDDEN, dt.count_params(params[:]))

	dt.train(&m, params[:], CFG, step)
	dt.evaluate(&m, CFG, posterior_q, "DeepSets MLP → GMM (K=5)")
}
