# Study 02 — cross-entropy, information, and the overlap between distributions

> From the "X·w = y" foundations thread. The MLE loss and cross-entropy are the
> same number wearing two hats (statistics and information theory). This study
> builds the information-theoretic view.

## Cross-entropy, defined

For two distributions over the same outcomes, the cross-entropy of `q`
relative to `p` is:

```
H(p, q) = −Σ_x p(x)·ln q(x)
```

- `p` = the **true/empirical** distribution
- `q` = the **model's** distribution

The Bernoulli loss summand `−[ x·ln p̂ + (1−x)·ln(1−p̂) ]` is, for a single
training example, exactly `H(empirical data, model)`: `x` is the one-hot label
(empirical distribution over {0,1}) and `p̂` is the model's predicted
distribution. **MLE and cross-entropy minimization are literally the same
objective.**

## The magic identity — the "overlap"

```
H(p, q)  =  H(p)  +  KL(p ‖ q)
```

- **`H(p)` = Shannon entropy**: `−Σ p ln p`. The intrinsic unpredictability of
  `p` — the average surprise of drawing from it.
- **`KL(p ‖ q)` = KL divergence**: `Σ p·ln(p/q)`. The *extra* surprise of
  believing `q` while the world draws from `p`. Measures how "far" q is from p —
  how little they overlap.
- **`H(p,q)` = cross-entropy**: intrinsic randomness + mismatch = *total*
  surprise of using `q` on data that actually comes from `p`.

Caveat: KL is **asymmetric** (`KL(p‖q) ≠ KL(q‖p)`), so it's not a symmetric
distance — the symmetric version is Jensen–Shannon. But the intuition holds:
full overlap → `KL = 0`; little overlap → huge KL; and `q(x)=0` where `p(x)>0`
blows up to ∞ (infinite surprise).

## The "surprise" reading

Shannon's self-information of event `x` under distribution `q`:

```
I(x) = −ln q(x)
```

Likely events are cheap, rare events are expensive. Cross-entropy is the
**expected surprise** — "average of `−ln q(x)` over x drawn from `p`." In plain
English, the loss says:

> *"On average, how surprised am I by the observed labels, given what my model
> believes?"*

Training = reduce the surprise until the model's beliefs match reality. That's
why the exit activation is non-negotiable: `−ln 0 = ∞`, `−ln 1 = 0`, so `q`
must be a proper probability — sigmoid/softmax is what keeps it one.

## Why the entropy term disappears

`H(p)` depends only on the data, not on the model parameters. During training
`p` (the labels) is fixed, so:

```
min_w H(p, q_w)  ⟺  min_w KL(p ‖ q_w)          (H(p) is a constant, dropped)
```

Minimizing cross-entropy is minimizing KL — the mismatch — because the data's
intrinsic randomness is out of your hands. Same "drop the constant" move as
`1/√(2πσ²)` in the Gaussian case, and the same as `A(θ)` being the only
θ-dependent term in the exponential family. Constants that don't depend on the
parameter don't matter.

## The noise floor

`KL ≥ 0` (Gibbs inequality), equality iff `q = p`. So:

```
H(p, q) ≥ H(p)
```

You can't beat `H(p)` — the **irreducible loss**, the intrinsic randomness of
the data. This is the "well-behaved noise" from Study 01 wearing an
information-theory hat: the Gaussian noise variance `σ²` and the data entropy
`H(p)` are the same idea — the floor below which no model can improve.

## Closing the arc

**noise assumption → likelihood → loss (matched pair) → exponential family →
`prediction − target` → and finally: the loss is "average surprise," KL is the
"overlap," and the entropy of the noise is the wall you can't pass.**
