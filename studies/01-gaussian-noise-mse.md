# Study 01 — Gaussian noise ⟹ MSE, and likelihood vs probability

> From the "X·w = y" foundations thread. Assumption: well-behaved (Gaussian)
> noise. Question: where do the loss and the exit activation actually come from?

## Setup — the data story

The assumption is that the world generates targets like this:

```
y = f(x; w) + ε,        ε ~ N(0, σ²)        (i.i.d. across data points)
```

- `f(x; w)` = the model's prediction (the "signal")
- `ε` = everything unexplained (the "noise")
- `σ²` = noise variance — how wild the noise is

So **conditioned on x and w, y is a Gaussian centered at the prediction**:

```
y | x, w  ~  N( f(x; w), σ² )
```

That single sentence is the *entire* assumption. Everything after this is just
bookkeeping.

## Step 1 — the density of one point

```
p(y | x, w) = 1/√(2πσ²) · exp( −(y − f(x;w))² / (2σ²) )
```

## Step 2 — likelihood of the whole dataset

Independent draws → multiply them together:

```
L(w) = Πᵢ p(yᵢ | xᵢ, w)
```

## Step 3 — log-likelihood (turns the product into a sum)

```
ℓ(w) = ln L(w) = Σᵢ [ −½·ln(2πσ²)  −  (yᵢ − f(xᵢ;w))² / (2σ²) ]
```

## Step 4 — maximize ℓ(w). Only one term depends on w:

```
argmax_w ℓ(w)  =  argmax_w { −(1/2σ²) · Σᵢ (yᵢ − f(xᵢ;w))² }
```

The normalization `−½ln(2πσ²)` is constant in w, and `1/2σ² > 0` is a constant
scale. Neither moves the argmax, so drop them:

```
argmax_w ℓ(w)  =  argmin_w Σᵢ (yᵢ − f(xᵢ;w))²        ← MSE × N
```

**That's it.** Maximizing the Gaussian likelihood *is* minimizing squared error.
MSE wasn't chosen; it fell out.

## The conceptual logic

The loss is *defined* as the negative log-density:

```
loss(w) = −ln p(y | x, w) = (y − f)² / (2σ²) + const
```

The Gaussian's exponent is a **quadratic form**: `−(y−f)²/(2σ²)`. Taking `−ln`
flips the sign, so the squared distance literally lands in the loss. **The loss
is always "whatever was sitting in the exponent" of your noise density.** That's
the whole recipe:

> assume a noise distribution → write its density → `−ln` it → the exponent
> becomes your cost, and the parameter it's centered on becomes your exit
> activation.

This is why losses and exit activations come in matched pairs: there is no
separate act of designing a loss. There's only a choice of noise, and the loss
is derivative.

## The free choices that *didn't* matter

- `σ²` — scales the loss, doesn't change the optimum (it does matter later:
  it's the variance of the estimator, and `1/σ²` becomes per-point *weights*
  if noise isn't constant).
- `½` vs `1` — convention, for clean derivatives.
- sum vs mean — same argmin.

## What falls out downstream — back to `X·w = y`

Set the gradient to zero:

```
∂L/∂w = Σᵢ (f(xᵢ;w) − yᵢ) · ∂f(xᵢ;w)/∂w = 0
```

For the linear model `f(x;w) = wᵀx`:

```
Σᵢ (wᵀxᵢ − yᵢ)·xᵢ = 0
⟹  (XᵀX) w = Xᵀy
⟹  w* = (XᵀX)⁻¹ Xᵀ y          ← the normal equation
```

So `X·w = y` is exactly the Gaussian MLE, closed form. And the gradient carries
`(fᵢ − yᵢ)` — the **prediction − target** structure, the same skeleton as the
`softmax − one_hot` fused gradient in `ml/autograd.odin`. It's one recurring
shape: *the gradient of a natural loss w.r.t. the natural parameter is
prediction − target.*

## Likelihood vs probability

Same function, different *axis*:

```
p(x ; θ)   ← read as a function of x (data), with θ fixed  → probability (density)
L(θ) = p(x ; θ)   ← read as a function of θ (parameter), with x fixed  → likelihood
```

- **Probability** asks: "given this θ, how spread-out is the data?"
- **Likelihood** asks: "given this one observed x, how much does each θ 'like' it?"

The pdf value at a point is a *density* (probability per unit), not a
probability — pointwise probability is 0 for continuous RVs. This is harmless:
we only ever use density *ratios*, and the infinitesimal unit cancels.

The likelihood is *not* a global feature of the distribution — it's the most
local thing there is: the joint density of your **one specific dataset**,
evaluated at exactly the observed values, read as a function of `w`.

| | probability `p(x;θ)` | likelihood `L(θ)` |
|---|---|---|
| varies over | data axis `x` | parameter axis `θ` |
| integrates to | **1** (a distribution over x) | **no guarantee** (not a distribution over θ) |
| the constant `1/√(2πσ²)` | is what *makes it sum to 1* | is dead weight — cancels in ratios |
| question it answers | "how likely is this *outcome*?" | "how plausible is this *parameter*, given what I saw?" |

The `1/√(2πσ²)` factor is the key witness: it's the normalization that lives on
the *data* axis. When you flip to the likelihood view, that constant has no
business being a normalization anymore — it doesn't depend on `w`, so it just
multiplies everything equally and you drop it. That's why Step 4 could throw
constants away.

Fisher's point: likelihood is **not** a probability, because it doesn't live on
a space that normalizes. You can't say "this θ has 30% probability" — but you
*can* say "this θ is 10× more plausible than that one." All inference with
likelihood is comparison (`argmax_w`, likelihood ratios, confidence regions) —
never "probability of θ."

In the Gaussian exercise: `p(yᵢ|xᵢ,w)` is the density of one point (Step 1,
data axis), and `L(w) = Πᵢ p(yᵢ|xᵢ,w)` is the likelihood (Step 2, parameter
axis) — **the same bell-curve formula, just letting `w` move instead of `y`.**

## Exercise to run the recipe yourself

Redo steps 1–4 assuming `y ∈ {0,1}` with `y|x,w ~ Bernoulli(p)` where
`p = σ(z)` (sigmoid, so p stays in [0,1]). You should land on binary
cross-entropy with the fused `σ(z) − y` gradient — and you'll see exactly
*where* the exit activation is forced: the density's parameter is a
probability, so the model's unbounded `z` needs sigmoid to become one.

## Why it all resolves so nicely — the exponential family and the partition function

The one-line answer: **the derivative of the log-partition function is the
mean, and the mean is the prediction.** That single theorem is the entire
"why it resolves so nicely."

### The exponential-family template

Gaussian, Bernoulli, Categorical are all members of one family. Every member
can be written as:

```
p(x; θ) = h(x) · exp( θ·T(x) ) / Z(θ)
```

- **θ (natural parameter)** — what the model's linear output `z` *is*. For
  Bernoulli, θ = logit.
- **T(x) (sufficient statistic)** — the part of the data that talks to θ. For
  Bernoulli, T(x) = x.
- **Z(θ) (partition function)** — the constant that makes the whole thing
  integrate/sum to 1. The normalizer.
- **h(x) (base measure)** — leftovers that don't depend on θ (e.g. Gaussian's
  `e^{−y²/2σ²}`).

Take `−ln` of both sides and the skeleton appears:

```
−ln p(x;θ) = ln Z(θ) − θ·T(x) − ln h(x)
   loss(θ) =  A(θ)   − θ·T(x) − const
              ↑           ↑
           normalizer    score
```

The loss has *exactly two* θ-dependent terms: the log-partition
`A(θ) = ln Z(θ)`, and a term that's **linear in θ**. Linear terms are trivially
differentiable, so the only interesting object is `A(θ)`.

### Why the normalizer isn't dead weight after all

We "dropped" the constant `1/√(2πσ²)` earlier because it didn't depend on `w`.
That was true but misleading — the *logarithmic* normalizer `A(θ)` depends on θ
and does the real work.

Here's the theorem. Since `p` must integrate to 1:

```
Z(θ) = ∫ h(x)·e^{θ·T(x)} dx
dZ/dθ = ∫ h(x)·T(x)·e^{θ·T(x)} dx = Z(θ) · E[T(x)]
```

Divide by Z and use `A = ln Z`:

```
dA/dθ = (dZ/dθ)/Z = E[T(x)]  =  μ(θ)    ← the mean parameter
```

**The gradient of the log-partition is the expected sufficient statistic.**
Now differentiate the loss:

```
d/dθ [ loss ] = dA/dθ − T(x) = μ(θ) − T(x) = prediction − target
```

Not a clever trick, not a coincidence — a **theorem about normalized
densities**. The "prediction − target" gradient is automatic whenever your
model outputs the natural parameter, because *the mean is literally the
derivative of the normalizer*.

### The three cases, all the same machine

| family | θ | T(x) | Z(θ) | A(θ) = ln Z | dA/dθ = E[T] | loss = A − θ·T |
|---|---|---|---|---|---|---|
| Bernoulli | logit `z` | `x` | `1+e^z` | `ln(1+e^z)` (softplus) | `σ(z) = p` | `ln(1+e^z) − xz` |
| Gaussian (fixed σ²) | `μ/σ²` | `y` | `√(2πσ²)·e^{μ²/2σ²}` | `μ²/2σ²` | `μ` | `(μ−y)²/2σ²` |
| Categorical | `z` vector | one-hot `x` | `Σ_j e^{z_j}` | `ln Σ e^{z_j}` (LSE) | `softmax(z) = p` | `ln Σe^z − x·z` |

The loss is always `A(θ) − θ·T(x)`. For Categorical that's `ln Σ_j e^{z_j} − x·z`
— exactly the logits+CE that `ml/ops.odin:520` computes. The **score term**
`θ·T(x)` is the "agreement between data and parameter"; it's linear in θ, so
its gradient is just `T(x)` = the target.

### The full resolution

1. The likelihood forces a **normalized** density, so the θ-dependence can only
   appear in two places: a linear score `θ·T(x)` and the normalizer `Z(θ)`.
2. Taking `−ln` turns the normalizer into `A(θ)` — and by the normalization
   identity, `dA/dθ = mean`.
3. Your model outputs θ (the natural parameter), the mean IS the prediction, so
   `gradient = prediction − target`. Every matched pair — MSE, BCE, CE, Poisson —
   is this same machine with a different `Z(θ)`.

That's why softmax+CE can be fused into `softmax − one_hot` in one line of
backward code (`ml/autograd.odin:213`): the fused gradient isn't an
optimization, it's the identity `dA/dθ = μ` applied at the natural parameter.
