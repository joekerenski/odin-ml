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
  conjugate/        M2: DeepSets MLP → GMM on the conjugate problem
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

Paper, same problem family with its transformer (DT-5): KL ≈ 0.0004.

## Next

- M3: the transformer — prior GMM components as tokens, observations as tokens,
  self-attention over components + cross-attention to data; reproduce Table 1.
- M4: full-covariance mixtures + sequential filtering (posterior → next prior).
