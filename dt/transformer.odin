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
	gelu:   bool, // activation of all MLPs (ReLU otherwise)
}

// 6 layers, d=64, 8 heads, MLP 256: ~0.43M params, the paper's size for this experiment.
DT_PAPER :: Dt_Config{k = 5, dim = 64, heads = 8, layers = 6, mlp = 256}

Block :: struct {
	ln1, ln2, ln3: ml.LayerNorm,
	self_attn:     ml.Attention,
	cross_attn:    ml.Attention,
	fc1, fc2:      ml.Linear,
}

// The decoder stack shared by every DT variant: prior tokens attend to each
// other and to the observation tokens.
Trunk :: struct {
	blocks:         []Block,
	ln_obs, ln_out: ml.LayerNorm,
	gelu:           bool,
}

trunk_make :: proc(cfg: Dt_Config) -> (t: Trunk) {
	d := cfg.dim
	t.gelu = cfg.gelu
	t.blocks = make([]Block, cfg.layers)
	for &b in t.blocks {
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
	t.ln_obs = ml.layer_norm_layer(d)
	t.ln_out = ml.layer_norm_layer(d)
	return
}

trunk_params :: proc(ps: ^[dynamic]^ml.Tensor, t: Trunk) {
	for b in t.blocks {
		for ln in ([]ml.LayerNorm{b.ln1, b.ln2, b.ln3}) do ml.layer_norm_params(ps, ln)
		ml.attention_params(ps, b.self_attn)
		ml.attention_params(ps, b.cross_attn)
		ml.linear_params(ps, b.fc1)
		ml.linear_params(ps, b.fc2)
	}
	ml.layer_norm_params(ps, t.ln_obs)
	ml.layer_norm_params(ps, t.ln_out)
}

act :: proc(gelu: bool, x: ^ml.Tensor) -> ^ml.Tensor {
	return gelu ? ml.gelu(x) : ml.relu(x)
}

// Prior tokens updated by attending to each other and to the data (pre-LN).
// Unembeddings apply ln_out.
trunk_forward :: proc(t: ^Trunk, x, obs: ^ml.Tensor) -> ^ml.Tensor {
	mem := ml.layer_norm_forward(&t.ln_obs, obs)
	x := x
	for &b in t.blocks {
		h := ml.layer_norm_forward(&b.ln1, x)
		x = ml.add(x, ml.attention_forward(&b.self_attn, h, h))
		x = ml.add(x, ml.attention_forward(&b.cross_attn, ml.layer_norm_forward(&b.ln2, x), mem))
		h = ml.layer_norm_forward(&b.ln3, x)
		x = ml.add(x, ml.linear_forward(&b.fc2, act(t.gelu, ml.linear_forward(&b.fc1, h))))
	}
	return x
}

// ---- 1D priors given by parameters (M2/M3, the conjugate problem) -------------

Dt_Model :: struct {
	cfg:                       Dt_Config,
	prior1, prior2:            ml.Linear, // (log α, log β) → dim/2 → k·dim
	obs1, obs2:                ml.Linear, // z → dim/2 → dim, per observation
	trunk:                     Trunk,
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
	m.trunk = trunk_make(cfg)
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
	trunk_params(&ps, m.trunk)
	return
}

// prior params [B, P] → k tokens [B, k, dim]
dt_prior_tokens :: proc(m: ^Dt_Model, prior: ^ml.Tensor) -> ^ml.Tensor {
	h := act(m.cfg.gelu, ml.linear_forward(&m.prior1, prior))
	return ml.reshape(ml.linear_forward(&m.prior2, h), {prior.shape[0], m.cfg.k, m.cfg.dim})
}

// observations [B, n, 1] → tokens [B, n, dim]
dt_obs_tokens :: proc(m: ^Dt_Model, z: ^ml.Tensor) -> ^ml.Tensor {
	return ml.linear_forward(&m.obs2, act(m.cfg.gelu, ml.linear_forward(&m.obs1, z)))
}

dt_transform :: proc(m: ^Dt_Model, x, obs: ^ml.Tensor) -> ^ml.Tensor {
	return trunk_forward(&m.trunk, x, obs)
}

// tokens [B, k, dim] → GMM with k components
dt_unembed :: proc(m: ^Dt_Model, x: ^ml.Tensor) -> Gmm {
	B, k := x.shape[0], x.shape[1]
	h := act(m.cfg.gelu, ml.linear_forward(&m.unembed, ml.layer_norm_forward(&m.trunk.ln_out, x)))
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

// ---- GMM priors: one token per mixture component (the paper's main form) -----
//
//   prior components ──MLP──► K tokens ─┬─► unembed ─► q(x | prior)
//   sensor readings ──MLP──► obs tokens ─► trunk ─► unembed ─► q(x | prior, data)
//
// A component token is embedded from its features (log w, μ, log diag U, lower
// U; see gmm_full.odin) and unembedded back to the same parameterization, so a
// posterior can be fed in again as the next prior (filtering).

Dt_Full :: struct {
	cfg:                                      Dt_Config,
	comp1, comp2:                             ml.Linear, // N_FEAT → dim/2 → dim
	obs1, obs2:                               ml.Linear, // OBS_FEAT → dim/2 → dim
	trunk:                                    Trunk,
	unembed:                                  ml.Linear, // dim → dim/2
	out_logit, out_mean, out_log_diag, out_lower: ml.Linear,
	maps:                                     Tril_Maps,
}

dt_full_model :: proc(cfg: Dt_Config, n_obs_feat: i32) -> (m: Dt_Full) {
	d, h := cfg.dim, cfg.dim / 2
	m.cfg = cfg
	m.comp1 = ml.linear(N_FEAT, h)
	m.comp2 = ml.linear(h, d, .Xavier)
	m.obs1 = ml.linear(n_obs_feat, h)
	m.obs2 = ml.linear(h, d, .Xavier)
	m.trunk = trunk_make(cfg)
	m.unembed = ml.linear(d, h)
	m.out_logit = ml.linear(h, 1, .Zeros)
	m.out_mean = ml.linear(h, D, .Xavier)
	m.out_log_diag = ml.linear(h, D, .Zeros)
	m.out_lower = ml.linear(h, M, .Zeros)
	m.maps = tril_maps()
	return
}

dt_full_params :: proc(m: Dt_Full) -> (ps: [dynamic]^ml.Tensor) {
	for l in ([]ml.Linear{m.comp1, m.comp2, m.obs1, m.obs2, m.unembed, m.out_logit, m.out_mean, m.out_log_diag, m.out_lower}) {
		ml.linear_params(&ps, l)
	}
	trunk_params(&ps, m.trunk)
	return
}

// component features [B, K, N_FEAT] → tokens [B, K, dim]
dt_full_prior_tokens :: proc(m: ^Dt_Full, comps: ^ml.Tensor) -> ^ml.Tensor {
	return ml.linear_forward(&m.comp2, act(m.cfg.gelu, ml.linear_forward(&m.comp1, comps)))
}

// observation features [B, n, n_obs_feat] → tokens [B, n, dim]
dt_full_obs_tokens :: proc(m: ^Dt_Full, z: ^ml.Tensor) -> ^ml.Tensor {
	return ml.linear_forward(&m.obs2, act(m.cfg.gelu, ml.linear_forward(&m.obs1, z)))
}

dt_full_unembed :: proc(m: ^Dt_Full, x: ^ml.Tensor) -> Gmm_Full {
	h := act(m.cfg.gelu, ml.linear_forward(&m.unembed, ml.layer_norm_forward(&m.trunk.ln_out, x)))
	return {
		ml.linear_forward(&m.out_logit, h),
		ml.linear_forward(&m.out_mean, h),
		ml.clip(ml.linear_forward(&m.out_log_diag, h), LOG_DIAG_MIN, LOG_DIAG_MAX),
		ml.linear_forward(&m.out_lower, h),
	}
}

dt_full_forward :: proc(m: ^Dt_Full, comps, z: ^ml.Tensor) -> (q_prior, q_post: Gmm_Full) {
	tokens := dt_full_prior_tokens(m, comps)
	q_prior = dt_full_unembed(m, tokens)
	q_post = dt_full_unembed(m, trunk_forward(&m.trunk, tokens, dt_full_obs_tokens(m, z)))
	return
}
