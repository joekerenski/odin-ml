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
  tests/              tensor_ops (118 checks), metal (GPU smoke test)
  dt/                 the paper project (Distribution Transformers), see dt/README.md
  bench/              loop bench, matmul benches, tinygrad/numpy compare
  docs/, studies/     notes
  Run from repo root via make: `make test`, `make mnist`, `make cnn` (see Makefile)

The UOp model (ml/uop.odin)
- `Tensor :: UOp`. One node type: op, src, arg, shape, data, requires_grad, grad.
- Buffers are always dense row-major. No strides; Reshape is a view (same buffer),
  Permute realizes a copy or folds into GEMM (sgemm transpose flags).
- ops.odin builds nodes; compositions (relu, sigmoid, softmax family,
  layer_norm, cross_entropy) are just ops of ops. detach() stops grads.
- Axes as in numpy: negative counts from the end. sum(x) / mean(x) / max_all(x)
  reduce everything; sum(x, axis) keeps the reduced dim as 1.
- MatMul is batched ([..., M, K] @ [..., K, N], batch dims broadcast); a 2D
  weight folds the batch into one GEMM; mT (last-two swap) folds into sgemm.
- autograd.odin: grad rules emit UOps; backward realizes loss + all leaf grads
  in ONE schedule, so backward is fused like forward.
- realize.odin is the only executor: topo → fuse same-shape ewise chains
  (single consumer) → fold transposes → run fused kernels / primitive kernels.
- IR ops: Input Const | Add Sub Mul Div Max CmpLt Neg Exp Log Sqrt Expand |
  Sum ReduceMax | Reshape Permute | MatMul Conv2d MaxPool2d (+ conv/pool bwd)
- optim.odin: SGD, Adam/AdamW (tinygrad semantics), cosine_lr with warmup;
  Optimizer union drives Trainer. nn: Linear, Conv2d, LayerNorm.

Numbers (M5, after unification; before in parens)
- MNIST MLP 20 epochs 3.6 s (14.0 s), 98.0%. MNIST CNN 3 epochs 17.3 s (49.4 s), 98.2%.
- bench/loop mlp_step 0.15 ms (0.87 ms). 33/33 match tinygrad (ops, grads,
  attention fwd+bwd, Adam/AdamW trajectories).
- CPU backend: fused ewise kernel runs per 256-element chunk; parallel_for on
  a persistent worker pool (fused, permute, reduce, batched GEMM); kernel
  outputs allocated non-zeroed; dt arenas use 64 MB warm blocks.
- Metal feasibility (bench/metal_dispatch): 1000 dependent kernels in ONE
  command buffer cost ~1 µs dispatch each; 327k-element ewise 0.017 ms vs CPU
  0.084 ms (5×). Commit+wait per kernel: 0.2–0.3 ms (slower than CPU).

Known debt
- Multi-consumer nodes break fusion (x feeding relu AND its grad = own kernel).
- Over-budget fused groups (>16 ops / >8 inputs) fall back to op-by-op.
- Optimizers update raw buffers outside the graph (fine for now; lazy optim later).
- Exp/Log run the scalar path of the fused kernel (no SIMD exp/log yet).
- Metal: metal_add prototype only, not wired into the scheduler.
- A DT step is ~1200 kernels; multi-consumer fusion would cut that a lot.

Plan for today (user)
1. Work through the milestones  2. Implement the paper (M2–M4)
3. Clean up ml/ so it imports as a module (Odin collection)
4. Visualization app on odin-ui-v2 in its own folder

Milestones
0. Cleanup ✓  One UOp model: one executor, autograd emits UOps ✓
1. Ops ✓  Log, Sqrt, reduce Max; softmax/logsumexp/LayerNorm/CE as compositions;
   batched matmul; Adam/AdamW; cosine LR + warmup. All checked vs tinygrad.
2. Conjugate toy ✓  dt/conjugate: DeepSets MLP → 5-GMM over log σ² (InvGamma
   prior, 10 obs). KL(exact‖q) mean 0.0015, median 0.00048 (paper DT-5 ≈ 0.0004);
   best single Gaussian 0.0101. ~2 min on M5.
3. Transformer ✓  dt/table1: the paper's DT (0.42M params). KL mean/median:
   DT-5 0.00082/0.00034 (paper 0.0003), DT-2 0.00139/0.00099 (paper 0.0058).
   ~14 min per run on CPU (84 ms/step).
4. Full-covariance GMM (Cholesky param) + sequential sensor fusion. First UI hook.
5. Metal backend (next, before M4): device-resident buffers, whole step in one
   command buffer, MSL codegen for fused groups, reduce/permute/batched GEMM.

References
- Paper: https://arxiv.org/html/2502.02463v3
- Reference impl (PyTorch, MIT): https://github.com/GWhittle110/distribution-transformers
- docs/tiny-inspo.md
