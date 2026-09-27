# dt — Distribution Transformers, on odin-ml

Reimplementing [Distribution Transformers](https://arxiv.org/abs/2502.02463)
(Whittle et al., 2025): amortized Bayesian inference that maps a **prior + data**
to a **posterior**, both as Gaussian mixtures, in one forward pass.

This folder is the paper project; `ml/` is the library it runs on.

```
dt/                 package dt — shared pieces
  gmm.odin          GMM head (logits, means, log σ), GMM NLL loss, f64 density
  invgamma.odin     the conjugate problem: meta-prior, sampler, exact posterior
  kl.odin           KL(exact ‖ q) by quadrature, reference baselines
  train.odin        shared harness: fresh batches, Adam + cosine, evaluation
  transformer.odin  the Distribution Transformer: shared trunk; 1D-param priors (M3)
                    and GMM priors, one token per component (M4)
  gmm_full.odin     full-covariance mixtures: NLL on the graph, f64 host math
  tracking.odin     the sensor-fusion problem: dynamics, sensors, meta-prior
  checkpoint.odin   model library: save to / load from models/*.safetensors
  conjugate/        M2: DeepSets MLP → GMM on the conjugate problem
  table1/           M3: the transformer on the same problem (paper, Table 1)
  fusion/           M4: sensor fusion, the DT as the update step of a filter (Table 3)
  models/           `make models`: list the trained models and their metadata
```

Every experiment loads its trained weights from `models/` if present, otherwise trains
and saves them (models/ is not in git: retrain to reproduce) (safetensors, with config and eval results as metadata; they also load
in numpy/torch/MLX). `-- train` forces retraining.

## M2 — conjugate check (`make dt-conjugate`, ~2 min)

σ² ~ InvGamma(α, β) with (α, β) drawn per problem, 10 observations z ~ N(0, σ²).
The exact posterior is known, so the model is graded by KL(exact ‖ model) over
y = log σ². Training minimizes −log q(y_true) on fresh problems every step; the
printed gap to the exact posterior's NLL is the remaining expected KL.

| KL on 1000 held-out problems | mean | median | p95 |
|---|---|---|---|
| DeepSets MLP → GMM (K=5), 12k steps | 0.0015 | 0.00048 | 0.0033 |
| best single Gaussian (moment-matched) | 0.0101 | 0.0101 | 0.0130 |
| prior only | 0.489 | 0.355 | 1.36 |

## M3 — the Distribution Transformer (`make dt-table1`, ~14 min; `-- 2` for DT-2)

Paper architecture (Sec. 3): prior params → k tokens; one token per observation
(shared MLP); 6 pre-LN decoder blocks, d=64, 8 heads (self-attention over the prior
tokens, cross-attention to the data); per-token unembedding → k-GMM, weights by a
softmax across tokens. Loss = posterior NLL + prior NLL (the prior tokens alone
must decode to the prior). 0.42M params; 10k steps × batch 1024 = 10M samples.

| KL(exact ‖ ·), 1000 held-out problems | mean | median | p95 | paper (wide) |
|---|---|---|---|---|
| DT-5 | 0.00082 | 0.00034 | 0.0014 | 0.0003 |
| DT-2 | 0.00139 | 0.00099 | 0.0021 | 0.0058 |
| DeepSets MLP (M2, 12k steps) | 0.00150 | 0.00048 | 0.0033 | — |
| best single Gaussian | 0.0101 | 0.0101 | 0.0130 | — |

Prior reconstruction (prior tokens → GMM vs the exact prior): DT-5 0.0009, DT-2 0.0040.

Same ordering as the paper and the same order of magnitude; not number for
number, since the paper doesn't publish its meta-prior ranges or n, and trains
at batch 5000 / lr 5e-3 (we use 1024 / 1e-3). Unstated details we chose:
ReLU MLPs, pre-LN, block MLP width 256 (gives the paper's 0.43M params).

## M4 — sensor fusion: the DT as a filter (`make dt-fusion`)

Paper Sec. 4.3.1, settings from the reference implementation: a 2D target with state
(px, vx, py, vy) and linear dynamics with rank-2 process noise, seen through a
rangefinder (70% true echo with range-proportional noise, 20% exponential clutter,
10% uniform failures) and a bearing sensor (σ = 0.1 rad).

The DT learns **one Bayesian update**: 4-component full-covariance GMM prior + one
reading per sensor → GMM posterior. Each prior component is one token (log w, μ, log-diag
and lower triangle of the precision's Cholesky factor U); each reading is one token (a
shared MLP with a sensor one-hot). Training priors come from the reference's conjugate
meta-prior (Dirichlet weights, Gaussian means, Wishart precisions). As a filter, the
posterior is pushed through the dynamics exactly (GMM → GMM) and fed back as the next
prior, 100 times.

Full-covariance NLL without a triangular solve: with Λ = UUᵀ,
log N(x) = Σ log U_jj − ½‖Uᵀ(x−μ)‖² + c, and Uᵀr is a diagonal scale plus two constant
0/1 matmuls over the strict lower triangle. (The paper factors the covariance instead.)

60k steps × batch 1024 (61M samples, ~37 min on the M5 GPU), 0.41M params:

| one update, 1000 held-out problems | E[−log q] | E[KL(exact ‖ q)] |
|---|---|---|
| exact posterior (importance sampling, 20k draws) | −0.068 | — |
| DT-4 (60k steps) | +0.075 | **0.143** |
| DT-4 (20k steps) | +0.214 | 0.282 |
| best single Gaussian | +0.873 | 0.942 |
| prior (ignores readings) | +1.910 | 1.978 |

| filtering, 100 series × 100 steps | mean −log q_t(x_t) | ms / step (100 series) |
|---|---|---|
| DT-4 + exact predict (60k steps) | **−0.359 ± 0.039** | 2.5 (Metal) |
| DT-4 + exact predict (20k steps) | −0.125 ± 0.043 | 2.6 |
| particle filter, 1000 | −0.355 ± 0.051 | 0.7 (10 cores) |
| particle filter, 5000 | −0.457 ± 0.047 | 3.5 |
| particle filter, 50000 | −0.490 ± 0.035 | 37 |
| dynamics only (no readings) | +4.225 | — |
| paper: DT / PF-5000 / EKF | −0.197 / −0.244 / +95.9 | |

By time step, the DT is within 0.04–0.05 of the particle filters for the first 30 steps
and falls behind later (t = 60–99: −0.16 vs −0.35): the error builds up as the track
drifts away from the training meta-prior's range (paper App. C.5 sees the same, later).
The filter starts from 4 identical components, as in the paper, and the permutation-
equivariant DT keeps them nearly identical: on a typical track the mixture behaves like
one Gaussian.

### Same model in tinygrad (`dt/tinygrad/`)

A parameter-for-parameter port (same layout, so checkpoints load both ways). On a fixed
batch with our trained weights the losses agree to 1e-6 (whole-model oracle). Trained the
same way (60k × 1024) and scored by the Odin evaluator:

| M5 GPU, same training step | kernels / step | ms / step | 60k-step run | update E[KL] | filter NLL |
|---|---|---|---|---|---|
| odin-ml | 1432 (+ Adam on CPU) | 34 | 37 min | 0.143 | −0.359 ± 0.039 |
| tinygrad, JIT | 1295 (764 fused Adam) | 36.7 | — | — | — |
| tinygrad, JIT + BEAM=2 | 1295 (764 fused Adam) | 16.6 | 21 min | 0.151 | −0.286 ± 0.041 |

Same training loss and single-update KL; the filter score differs by seed, almost all of
it late in the sequence (t = 60–99: −0.16 vs −0.02), where the tracks leave the training
distribution. tinygrad fuses the model into about half our kernels, but that alone buys
nothing (fused Adam: 1295 → 764 kernels at the same 16.6 ms): its JIT replays a recorded
command list, and the 2.2× over its own default comes from BEAM-tuned kernels.

Differences from the paper: batch 1024 (paper 5000), block MLP 256 wide (2048), pre-LN,
precision instead of covariance factor. The particle filter is ours (bootstrap,
systematic resampling, weighted Gaussian fit + 1e-3·I as in the paper's scoring); it is
stronger than the paper's (−0.46 vs −0.24 at 5000 particles), so compare the DT against
our PF column, not across papers.

## Next

- Close the gap to the particle filter late in the sequence (training priors closer to
  what the filter sees; wider block MLP as in the paper).
