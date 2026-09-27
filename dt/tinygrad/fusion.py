"""
M4 (sensor fusion, paper Sec. 4.3.1) in tinygrad: the same Distribution Transformer as
dt/fusion (Odin), parameter for parameter, so checkpoints load in both directions.

  uv run fusion.py check [odin.safetensors]   whole-model oracle: loss on one batch here vs Odin
  uv run fusion.py bench [steps]              kernels/step and ms/step (try BEAM=2)
  uv run fusion.py train [steps]              train, save ../../models/fusion_dt4_tinygrad.safetensors

Evaluate a tinygrad-trained model with the Odin evaluator (same file format):
  odin run dt/fusion -o:speed -- eval models/fusion_dt4_tinygrad.safetensors

TINYGRAD_PATH=/path/to/tinygrad uses a local checkout instead of the pinned one.
"""
import math, os, sys, time, pathlib
if p := os.environ.get("TINYGRAD_PATH"): sys.path.insert(0, p)
import numpy as np
from tinygrad import Tensor, TinyJit, GlobalCounters, Context, Device
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import safe_save, safe_load

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
CHECKPOINT = MODELS / "fusion_dt4_tinygrad.safetensors"

# ---- the problem (dt/tracking.odin) ------------------------------------------------
D, K, M = 4, 4, 6
N_FEAT, N_OBS, OBS_FEAT = 1 + 2 * D + M, 2, 4
RANGE_SCALE, RANGE_RATE, RANGE_MAX, RANGE_W = 0.1, 1.0, 20.0, (0.7, 0.2, 0.1)
BEARING_STD = 0.1
LOG_W_MIN, LOG_DIAG_MIN, LOG_DIAG_MAX = -14.0, -7.0, 7.0
TRIL = [(i, j) for i in range(1, D) for j in range(i)]  # (1,0) (2,0) (2,1) (3,0) ...
MU_CHOL = np.linalg.cholesky(np.eye(D) + np.ones((D, D)))

def sample_batch(rng: np.random.Generator, B: int):
  """Fresh problems: GMM prior from the conjugate meta-prior, x ~ prior, one reading per sensor."""
  w = rng.gamma(1.0 / K, 1.0, (B, K)); w /= w.sum(-1, keepdims=True)
  mu = rng.standard_normal((B, K, D)) @ MU_CHOL.T
  A = np.zeros((B, K, D, D))  # Bartlett: Λ = 4 A Aᵀ + 1e-4 I, A_ii² ~ χ²(D+1-i)
  for i in range(D):
    A[..., i, i] = np.sqrt(rng.chisquare(D + 1 - i, (B, K)))
    for j in range(i): A[..., i, j] = rng.standard_normal((B, K))
  U = np.linalg.cholesky(4 * A @ A.swapaxes(-1, -2) + 1e-4 * np.eye(D))  # precision factor
  k = (rng.random((B, 1)) > np.cumsum(w, -1)).sum(-1).clip(0, K - 1)
  b = np.arange(B)
  eps = rng.standard_normal((B, D, 1))
  x = mu[b, k] + np.linalg.solve(U[b, k].swapaxes(-1, -2), eps)[..., 0]  # μ + U⁻ᵀ ε
  r = np.hypot(x[:, 0], x[:, 2])
  u = rng.random(B)
  z_r = np.where(u < RANGE_W[0], rng.normal(r, RANGE_SCALE * (r + 1)),
                 np.where(u < RANGE_W[0] + RANGE_W[1], rng.exponential(1 / RANGE_RATE, B), rng.uniform(0, RANGE_MAX, B)))
  z_r = z_r.clip(0, RANGE_MAX)
  z_b = np.arctan2(x[:, 0], x[:, 2]) + rng.normal(0, BEARING_STD, B)
  z_b = z_b - 2 * np.pi * np.floor((z_b + np.pi) / (2 * np.pi))
  comps = np.concatenate([np.maximum(np.log(np.maximum(w, 1e-300)), LOG_W_MIN)[..., None], mu,
                          np.log(np.diagonal(U, axis1=-2, axis2=-1)), np.stack([U[..., i, j] for i, j in TRIL], -1)], -1)
  z = np.zeros((B, N_OBS, OBS_FEAT))
  z[:, 0, 0], z[:, 0, 2], z[:, 1, 1], z[:, 1, 3] = z_r, 1, z_b, 1
  return comps.astype(np.float32), z.astype(np.float32), x[:, None, :].astype(np.float32)

# ---- the model (dt/transformer.odin, ml/nn.odin) --------------------------------------
class Linear:  # W [in, out] like ml.Linear, so weights are interchangeable
  def __init__(self, n_in, n_out, init="he"):
    std = {"he": math.sqrt(2 / n_in), "xavier": math.sqrt(2 / (n_in + n_out)), "zeros": 0.0}[init]
    self.W = Tensor.randn(n_in, n_out) * std if std else Tensor.zeros(n_in, n_out)
    self.b = Tensor.zeros(n_out)
  def __call__(self, x): return x @ self.W + self.b
  def params(self): return [self.W, self.b]

class LayerNorm:
  def __init__(self, d): self.g, self.b = Tensor.ones(d), Tensor.zeros(d)
  def __call__(self, x): return x.layernorm(eps=1e-5) * self.g + self.b
  def params(self): return [self.g, self.b]

class Attention:
  def __init__(self, d, heads):
    self.q, self.k, self.v, self.o = (Linear(d, d, "xavier") for _ in range(4))
    self.heads = heads
  def __call__(self, x, mem):
    B, T, Dm = x.shape; S, H = mem.shape[1], self.heads; dh = Dm // H
    split = lambda t, L: t.reshape(B, L, H, dh).permute(0, 2, 1, 3)
    q, k, v = split(self.q(x), T), split(self.k(mem), S), split(self.v(mem), S)
    att = (q @ k.transpose(-1, -2) * (1 / math.sqrt(dh))).softmax(-1)
    return self.o((att @ v).permute(0, 2, 1, 3).reshape(B, T, Dm))
  def params(self): return [p for l in (self.q, self.k, self.v, self.o) for p in l.params()]

class Block:
  def __init__(self, d, heads, mlp):
    self.ln1, self.ln2, self.ln3 = LayerNorm(d), LayerNorm(d), LayerNorm(d)
    self.self_attn, self.cross_attn = Attention(d, heads), Attention(d, heads)
    self.fc1, self.fc2 = Linear(d, mlp), Linear(mlp, d, "xavier")
  def params(self):
    return (self.ln1.params() + self.ln2.params() + self.ln3.params() + self.self_attn.params()
            + self.cross_attn.params() + self.fc1.params() + self.fc2.params())

class DtFull:
  """FUSION_MODEL: k=4, dim=64, heads=8, layers=6, mlp=256, GELU. Param order = dt_full_params."""
  def __init__(self, dim=64, heads=8, layers=6, mlp=256):
    h = dim // 2
    self.comp1, self.comp2 = Linear(N_FEAT, h), Linear(h, dim, "xavier")
    self.obs1, self.obs2 = Linear(OBS_FEAT, h), Linear(h, dim, "xavier")
    self.blocks = [Block(dim, heads, mlp) for _ in range(layers)]
    self.ln_obs, self.ln_out = LayerNorm(dim), LayerNorm(dim)
    self.unembed = Linear(dim, h)
    self.out_logit, self.out_mean = Linear(h, 1, "zeros"), Linear(h, D, "xavier")
    self.out_log_diag, self.out_lower = Linear(h, D, "zeros"), Linear(h, M, "zeros")
    sel, put = np.zeros((D, M), np.float32), np.zeros((M, D), np.float32)
    for p, (i, j) in enumerate(TRIL): sel[i, p], put[p, j] = 1, 1
    self.sel, self.put = Tensor(sel).is_param_(False), Tensor(put).is_param_(False)

  def params(self):
    ps = []
    for l in (self.comp1, self.comp2, self.obs1, self.obs2, self.unembed,
              self.out_logit, self.out_mean, self.out_log_diag, self.out_lower): ps += l.params()
    for b in self.blocks: ps += b.params()
    return ps + self.ln_obs.params() + self.ln_out.params()

  def trunk(self, x, obs):
    mem = self.ln_obs(obs)
    for b in self.blocks:
      h = b.ln1(x)
      x = x + b.self_attn(h, h)
      x = x + b.cross_attn(b.ln2(x), mem)
      x = x + b.fc2(b.fc1(b.ln3(x)).gelu())
    return x

  def unembed_gmm(self, x):
    h = self.unembed(self.ln_out(x)).gelu()
    return self.out_logit(h), self.out_mean(h), self.out_log_diag(h).clip(LOG_DIAG_MIN, LOG_DIAG_MAX), self.out_lower(h)

  def forward(self, comps, z):
    tokens = self.comp2(self.comp1(comps).gelu())
    obs = self.obs2(self.obs1(z).gelu())
    return self.unembed_gmm(tokens), self.unembed_gmm(self.trunk(tokens, obs))

  def nll(self, g, x):  # dt/gmm_full.odin: log N = Σ log U_jj − ½‖Uᵀ(x−μ)‖² − (D/2) log 2π
    logits, means, log_diag, lower = g
    r = x - means
    y = log_diag.exp() * r + (lower * (r @ self.sel)) @ self.put
    comp = log_diag.sum(-1, keepdim=True) - (y * y).sum(-1, keepdim=True) * 0.5
    lse = (logits.log_softmax(1) + comp).logsumexp(1, keepdim=True)
    return -(lse - D * 0.5 * math.log(2 * math.pi)).mean()

  def losses(self, comps, z, x):
    q_prior, q_post = self.forward(comps, z)
    post = self.nll(q_post, x)
    return post, self.nll(q_prior, x)

def save(model, path, meta):
  safe_save({f"{i:03d}": p for i, p in enumerate(model.params())}, str(path), {k: str(v) for k, v in meta.items()})

def load(model, path):
  sd = safe_load(str(path))
  for i, p in enumerate(model.params()):
    t = sd[f"{i:03d}"]
    assert t.shape == p.shape, f"{i:03d}: file {t.shape} vs model {p.shape}"
    p.assign(t.to(p.device)).realize()

def cosine_lr(step, total, warmup, base, min_lr=0.0):  # ml/optim.odin
  if step < warmup: return base * (step + 1) / warmup
  if step >= total: return min_lr
  p = (step - warmup) / max(1, total - warmup)
  return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * p))

def batch_tensors(rng, B):
  return [Tensor(a).is_param_(False) for a in sample_batch(rng, B)]

# ---- modes ------------------------------------------------------------------------------
def check(path):
  """Losses of an Odin-trained model on a fixed batch; writes the batch for the Odin side."""
  model = DtFull()
  load(model, path)
  comps, z, x = sample_batch(np.random.default_rng(0), 256)
  out = ROOT / "build" / "fusion_check_batch.safetensors"
  out.parent.mkdir(exist_ok=True)
  safe_save({"comps": Tensor(comps), "z": Tensor(z), "x": Tensor(x)}, str(out))
  post, prior = model.losses(Tensor(comps), Tensor(z), Tensor(x))
  print(f"tinygrad  posterior nll {post.item():+.6f}   prior nll {prior.item():+.6f}   ({path})")
  print(f"batch written to {out}; Odin side:\n  odin run dt/fusion -o:speed -- check {out.relative_to(ROOT)}")

def make_train_step(model, opt):
  @TinyJit
  @Context(TRAINING=1)
  def step(comps, z, x):
    opt.zero_grad()
    post, prior = model.losses(comps, z, x)
    (post + prior).backward()
    Tensor.realize(post, prior, *opt.schedule_step())
    return post, prior
  return step

def bench(steps):
  model = DtFull()
  opt = Adam(model.params(), lr=1e-3)
  step = make_train_step(model, opt)
  rng = np.random.default_rng(0)
  t_sample, t_step, kernels = [], [], 0
  for i in range(steps):
    t0 = time.perf_counter()
    comps, z, x = batch_tensors(rng, 1024)
    t1 = time.perf_counter()
    GlobalCounters.reset()
    step(comps, z, x)[0].item()
    t2 = time.perf_counter()
    if i >= 3: t_sample.append(t1 - t0); t_step.append(t2 - t1); kernels = GlobalCounters.kernel_count
    if i < 3: print(f"  warmup step {i}: {1e3 * (t2 - t1):.0f} ms" + ("  (JIT capture / BEAM search)" if i < 2 else ""))
  med = lambda v: 1e3 * sorted(v)[len(v) // 2]
  print(f"{Device.DEFAULT} BEAM={os.environ.get('BEAM', '0')}: {kernels} kernels/step   "
        f"step {med(t_step):.2f} ms (median of {len(t_step)})   batch sampling {med(t_sample):.2f} ms")

def train(steps, batch=1024, lr=1e-3, warmup=500, log_every=500):
  Tensor.manual_seed(0)
  model = DtFull()
  opt = Adam(model.params(), lr=lr)
  step = make_train_step(model, opt)
  rng = np.random.default_rng(0)
  run_post = run_prior = 0.0
  t0 = time.perf_counter()
  print(f"training on {Device.DEFAULT} (BEAM={os.environ.get('BEAM', '0')}): batch {batch}, {steps} steps, Adam lr {lr}, cosine (warmup {warmup})")
  for i in range(steps):
    opt.lr.assign(Tensor([cosine_lr(i, steps, warmup, lr)], device=opt.lr.device, dtype=opt.lr.dtype)).realize()
    post, prior = step(*batch_tensors(rng, batch))
    run_post += post.item(); run_prior += prior.item()
    if (i + 1) % log_every == 0:
      print(f"  step {i+1:5d}  posterior nll {run_post/log_every:+.4f}   prior nll {run_prior/log_every:+.4f}   {time.perf_counter()-t0:.1f}s", flush=True)
      last = run_post / log_every
      run_post = run_prior = 0.0
  secs = time.perf_counter() - t0
  MODELS.mkdir(exist_ok=True)
  out = CHECKPOINT if steps == 60000 else CHECKPOINT.with_name(f"{CHECKPOINT.stem}_{steps}steps.safetensors")
  save(model, out, {"experiment": "sensor fusion, paper 4.3.1 (M4), tinygrad port",
                           "arch": "DT-full k=4 dim=64 heads=8 layers=6 mlp=256 gelu=true",
                           "train": f"batch={batch} steps={steps} lr={lr} warmup={warmup}",
                           "train_seconds": f"{secs:.0f} on {Device.DEFAULT} BEAM={os.environ.get('BEAM', '0')}",
                           "final_posterior_nll": f"{last:.4f}", "framework": "tinygrad"})
  print(f"saved {out}")

if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
  arg = sys.argv[2] if len(sys.argv) > 2 else None
  if mode == "check": check(arg or MODELS / "fusion_dt4.safetensors")
  elif mode == "bench": bench(int(arg or 30))
  elif mode == "train": train(int(arg or 60000))
  else: sys.exit(__doc__)
