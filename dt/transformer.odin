package dt

// ============================================================================
// The Distribution Transformer (paper, Sec. 3) for 1D priors given by params.
//
//   prior params ──MLP──► k prior tokens ─┬─► unembed ─► q(x | prior)       (prior loss)
//                                         │
//   observations ──MLP──► n obs tokens ──►│ 6 × [ self-attn over the k tokens
//                         (one per obs)   │       cross-attn to the obs tokens
//                                         │       MLP ]           (pre-LN)
//                                         └─► unembed ─► q(x | prior, data)  (posterior)
//
// Tokens are unordered: no positional encodings, so the model is permutation-
// equivariant in the components and permutation-invariant in the observations.
// Unembedding is a per-token MLP → (logit, mean, log σ); the mixture weights are
// a softmax across the k tokens (the paper's "cross-softmax", inside gmm_nll).
// ============================================================================

import ml "../ml"

Dt_Config :: struct {
	k:      i32, // mixture components = prior tokens
	dim:    i32, // latent width
	heads:  i32,
	layers: int,
	mlp:    i32, // hidden width of each block's MLP
}

// 6 layers, d=64, 8 heads, MLP 256: ~0.43M params, the paper's size for this experiment.
DT_PAPER :: Dt_Config{k = 5, dim = 64, heads = 8, layers = 6, mlp = 256}

Block :: struct {
	ln1, ln2, ln3: ml.LayerNorm,
	self_attn:     ml.Attention,
	cross_attn:    ml.Attention,
	fc1, fc2:      ml.Linear,
}

Dt_Model :: struct {
	cfg:                       Dt_Config,
	prior1, prior2:            ml.Linear, // (log α, log β) → dim/2 → k·dim
	obs1, obs2:                ml.Linear, // z → dim/2 → dim, per observation
	blocks:                    []Block,
	ln_obs, ln_out:            ml.LayerNorm,
	unembed:                   ml.Linear, // dim → dim/2, per token
	out_logit, out_mean, out_log_std: ml.Linear, // dim/2 → 1 each
}

dt_model :: proc(cfg: Dt_Config, n_prior_params: i32 = 2) -> (m: Dt_Model) {
	d, h := cfg.dim, cfg.dim / 2
	m.cfg = cfg
	m.prior1 = ml.linear(n_prior_params, h)
	m.prior2 = ml.linear(h, cfg.k * d, .Xavier)
	m.obs1 = ml.linear(1, h)
	m.obs2 = ml.linear(h, d, .Xavier)
	m.blocks = make([]Block, cfg.layers)
	for &b in m.blocks {
		b = Block{
			ln1        = ml.layer_norm_layer(d),
			ln2        = ml.layer_norm_layer(d),
			ln3        = ml.layer_norm_layer(d),
			self_attn  = ml.attention_layer(d, cfg.heads),
			cross_attn = ml.attention_layer(d, cfg.heads),
			fc1        = ml.linear(d, cfg.mlp),
			fc2        = ml.linear(cfg.mlp, d, .Xavier),
		}
	}
	m.ln_obs = ml.layer_norm_layer(d)
	m.ln_out = ml.layer_norm_layer(d)
	m.unembed = ml.linear(d, h)
	m.out_logit = ml.linear(h, 1, .Zeros)
	m.out_mean = ml.linear(h, 1, .Xavier)
	m.out_log_std = ml.linear(h, 1, .Zeros)
	return
}

dt_params :: proc(m: Dt_Model) -> (ps: [dynamic]^ml.Tensor) {
	for l in ([]ml.Linear{m.prior1, m.prior2, m.obs1, m.obs2, m.unembed, m.out_logit, m.out_mean, m.out_log_std}) {
		ml.linear_params(&ps, l)
	}
	for b in m.blocks {
		for ln in ([]ml.LayerNorm{b.ln1, b.ln2, b.ln3}) do ml.layer_norm_params(&ps, ln)
		ml.attention_params(&ps, b.self_attn)
		ml.attention_params(&ps, b.cross_attn)
		ml.linear_params(&ps, b.fc1)
		ml.linear_params(&ps, b.fc2)
	}
	ml.layer_norm_params(&ps, m.ln_obs)
	ml.layer_norm_params(&ps, m.ln_out)
	return
}

// prior params [B, P] → k tokens [B, k, dim]
dt_prior_tokens :: proc(m: ^Dt_Model, prior: ^ml.Tensor) -> ^ml.Tensor {
	h := ml.relu(ml.linear_forward(&m.prior1, prior))
	return ml.reshape(ml.linear_forward(&m.prior2, h), {prior.shape[0], m.cfg.k, m.cfg.dim})
}

// observations [B, n, 1] → tokens [B, n, dim]
dt_obs_tokens :: proc(m: ^Dt_Model, z: ^ml.Tensor) -> ^ml.Tensor {
	return ml.linear_forward(&m.obs2, ml.relu(ml.linear_forward(&m.obs1, z)))
}

// Prior tokens updated by attending to each other and to the data.
dt_transform :: proc(m: ^Dt_Model, x, obs: ^ml.Tensor) -> ^ml.Tensor {
	mem := ml.layer_norm_forward(&m.ln_obs, obs)
	x := x
	for &b in m.blocks {
		h := ml.layer_norm_forward(&b.ln1, x)
		x = ml.add(x, ml.attention_forward(&b.self_attn, h, h))
		x = ml.add(x, ml.attention_forward(&b.cross_attn, ml.layer_norm_forward(&b.ln2, x), mem))
		h = ml.layer_norm_forward(&b.ln3, x)
		x = ml.add(x, ml.linear_forward(&b.fc2, ml.relu(ml.linear_forward(&b.fc1, h))))
	}
	return x
}

// tokens [B, k, dim] → GMM with k components
dt_unembed :: proc(m: ^Dt_Model, x: ^ml.Tensor) -> Gmm {
	B, k := x.shape[0], x.shape[1]
	h := ml.relu(ml.linear_forward(&m.unembed, ml.layer_norm_forward(&m.ln_out, x)))
	return {
		ml.reshape(ml.linear_forward(&m.out_logit, h), {B, k}),
		ml.reshape(ml.linear_forward(&m.out_mean, h), {B, k}),
		ml.reshape(ml.linear_forward(&m.out_log_std, h), {B, k}),
	}
}

// Both mixtures: q(x | prior) from the prior tokens alone, q(x | prior, data).
dt_forward :: proc(m: ^Dt_Model, prior, z: ^ml.Tensor) -> (q_prior, q_post: Gmm) {
	tokens := dt_prior_tokens(m, prior)
	q_prior = dt_unembed(m, tokens)
	q_post = dt_unembed(m, dt_transform(m, tokens, dt_obs_tokens(m, z)))
	return
}
