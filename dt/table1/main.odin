package main

// ============================================================================
// M3 — the Distribution Transformer on the paper's conjugate problem (Table 1).
//
// Same problem, harness and eval as M2 (dt/conjugate), but the model is the
// paper's architecture (dt/transformer.odin): prior params → k tokens, one
// token per observation, 6 pre-LN decoder blocks (self-attention over the
// prior tokens, cross-attention to the data), unembedded to a k-GMM.
//
// Loss (paper, Sec. 3.3): posterior NLL + prior NLL, where the prior term
// unembeds the prior tokens directly and scores the same sampled x.
//
//   make dt-table1                      DT-5
//   odin run dt/table1 -o:speed -- 2    DT-2   (optional 2nd arg: steps)
//
// Paper, Table 1 (wide meta-prior): DT-2 0.0058, DT-5 0.0003.
// ============================================================================

import "core:fmt"
import "core:os"
import "core:strconv"
import dt ".."
import ml "../../ml"

posterior_q :: proc(m: ^dt.Dt_Model, b: dt.Batch) -> dt.Gmm {
	_, q := dt.dt_forward(m, b.prior, b.z)
	return q
}

prior_q :: proc(m: ^dt.Dt_Model, b: dt.Batch) -> dt.Gmm {
	return dt.dt_unembed(m, dt.dt_prior_tokens(m, b.prior))
}

step :: proc(m: ^dt.Dt_Model, b: dt.Batch) -> (loss, post_nll: ^ml.Tensor) {
	q_prior, q_post := dt.dt_forward(m, b.prior, b.z)
	post_nll = dt.gmm_nll(q_post, b.y)
	return ml.add(post_nll, dt.gmm_nll(q_prior, b.y)), post_nll
}

main :: proc() {
	cfg := dt.Config{n_obs = 10, batch = 1024, steps = 10000, warmup = 300, log_every = 500, n_test = 1000, lr = 1e-3}
	model_cfg := dt.DT_PAPER
	if len(os.args) > 1 do model_cfg.k = i32(strconv.parse_int(os.args[1]) or_else 5)
	if len(os.args) > 2 do cfg.steps = strconv.parse_int(os.args[2]) or_else cfg.steps
	cfg.log_every = max(1, min(cfg.log_every, cfg.steps / 10))

	fmt.printfln("=== M3: Distribution Transformer DT-%d, InvGamma prior on σ² ===", model_cfg.k)
	ml.warn_if_unoptimized()
	ml.debug_from_env()
	ml.seed(0)

	m := dt.dt_model(model_cfg)
	params := dt.dt_params(m)
	fmt.printfln("%d layers, d=%d, %d heads, MLP %d, k=%d: %d params; n=%d obs, batch %d, %d steps",
		model_cfg.layers, model_cfg.dim, model_cfg.heads, model_cfg.mlp, model_cfg.k,
		dt.count_params(params[:]), cfg.n_obs, cfg.batch, cfg.steps)

	dt.train(&m, params[:], cfg, step)
	dt.evaluate(&m, cfg, posterior_q, fmt.tprintf("DT-%d", model_cfg.k))
	dt.evaluate_prior_fit(&m, cfg, prior_q)
}
