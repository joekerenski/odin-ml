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
  transformer.odin  the Distribution Transformer (prior tokens × observation tokens)
  conjugate/        M2: DeepSets MLP → GMM on the conjugate problem
  table1/           M3: the transformer on the same problem (paper, Table 1)
```

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

## Next

- M4: full-covariance mixtures + sequential filtering (posterior → next prior).
