Objective
Small-model training lib in Odin (tinygrad as inspiration + oracle): Tensor → UOp IR → kernels.
North star: reimplement Distribution Transformers (arXiv 2502.02463) — amortised Bayesian
inference, prior GMM + observations → posterior GMM, ~0.43M params. Refine the lib on the way.
UI (odin-ui-v2) comes after the strategy fits.

Decisions
- One flow: Tensor (lazy) → UOp DAG → fuse → kernels. No second IR, no sketches.
- MatMul stays a primitive. No mul+sum → GEMM recovery pass.
- Performance comes from hand-written kernels per primitive / fused group
  (Accelerate on CPU now, handwritten Metal once the rest settles), not from a
  codegen pipeline.
- ReLU/Sigmoid/softmax/LayerNorm etc. are UOp compositions, not primitives.

Layout
  ml/                 the library
  examples/tour/      the API at every level (01 tensor … 05 nn), always runnable
  examples/           regression, mnist (MLP), mnist_cnn
  tests/              tensor_ops (98 checks), metal (GPU smoke test)
  bench/              loop bench, matmul benches, tinygrad/numpy compare
  docs/, studies/     notes
  Run from repo root via make: `make test`, `make mnist`, `make cnn` (see Makefile)

The UOp model (ml/uop.odin)
- `Tensor :: UOp`. One node type: op, src, arg, shape, data, requires_grad, grad.
- Buffers are always dense row-major. No strides; Reshape is a view (same buffer),
  Permute realizes a copy or folds into GEMM (sgemm transpose flags).
- ops.odin builds nodes; compositions (relu, sigmoid, mean) are just ops of ops.
- autograd.odin: grad rules emit UOps; backward realizes loss + all leaf grads
  in ONE schedule, so backward is fused like forward.
- realize.odin is the only executor: topo → fuse same-shape ewise chains
  (single consumer) → fold transposes → run fused kernels / primitive kernels.
- IR ops: Input Const | Add Sub Mul Div Max CmpLt Neg Exp Expand | Sum |
  Reshape Permute | MatMul Conv2d MaxPool2d CrossEntropy (+ backward prims)

Numbers (M5, after unification; before in parens)
- MNIST MLP 20 epochs 4.1 s (14.0 s), 98.0%. MNIST CNN 3 epochs 27.8 s (49.4 s), 98.2%.
- bench/loop mlp_step 0.15 ms (0.87 ms). 11/11 match tinygrad.

Known debt
- Multi-consumer nodes break fusion (x feeding relu AND its grad = own kernel).
- Over-budget fused groups (>16 ops / >8 inputs) fall back to op-by-op.
- SGD updates raw buffers outside the graph (fine for now; lazy optim later).
- CrossEntropy is a primitive with labels in arg; becomes a composition
  (logsumexp − gather) once Log / reduce-max exist.
- Metal: metal_add prototype only, not wired into the scheduler.

Milestones
0. Cleanup ✓  One UOp model: one executor, autograd emits UOps ✓
1. Ops: Log, Sqrt, reduce Max; softmax/logsumexp/LayerNorm as compositions;
   batched matmul; Adam; cosine LR + warmup. Each checked vs tinygrad.
2. Conjugate toy with an MLP: (prior params, data) → posterior params,
   diagonal-cov GMM NLL loss, synthetic meta-prior sampler. Check vs closed form.
3. Attention + 6-layer decoder (d=64, 8 heads, MLP 2048). Reproduce Table 1 (KL ≈ 4e-4).
4. Full-covariance GMM (Cholesky param) + sequential sensor fusion. First UI hook.
5. Handwritten Metal kernels for the hot primitives (MatMul, fused ewise, reduce).

References
- Paper: https://arxiv.org/html/2502.02463v3
- Reference impl (PyTorch, MIT): https://github.com/GWhittle110/distribution-transformers
- docs/tiny-inspo.md
