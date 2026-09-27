Objective
Small-model training lib in Odin (tinygrad as inspiration + oracle): Tensor → UOp IR → kernels.
North star: reimplement Distribution Transformers (arXiv 2502.02463) — amortised Bayesian
inference, prior GMM + observations → posterior GMM, ~0.43M params. Refine the lib on the way.
UI (odin-ui-v2) comes after the strategy fits.

Decisions
- One flow: Tensor (lazy) → UOp DAG → fuse → kernels. No second IR, no sketches.
- MatMul stays a primitive. No mul+sum → GEMM recovery pass.
- Performance comes from hand-written kernels per primitive / fused group
  (Accelerate + all cores on CPU, generated/handwritten Metal on GPU), not from a
  general optimizing compiler (fused groups are rendered to MSL directly).
- ReLU/Sigmoid/softmax/LayerNorm etc. are UOp compositions, not primitives.

Layout
  ml/                 the library
  examples/tour/      the API at every level (01 tensor … 05 nn), always runnable
  examples/           regression, mnist (MLP), mnist_cnn
  tests/              tensor_ops (118 checks, any device), metal (Metal vs CPU parity)
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
- Metal backend (ml/backend_metal_darwin.odin): ML_DEVICE=metal, same code.
  Unified memory: ml.arena_init arenas are zero-copy; other memory is staged.
  A realize encodes into command buffers committed every 128 kernels (GPU runs
  while the CPU encodes), synced once. Fused groups → generated MSL, cached by
  shape hash. GEMM on simdgroup_matrix 8×8 units fed from threadgroup memory,
  split-K for long K, a tiny kernel for attention heads. Conv/pool: CPU fallback.
  DT step 86 ms (CPU) → 34 ms (Metal).
- Checks: tests on both devices (118), Metal-vs-CPU parity over every kernel
  path (29), tinygrad + MLX oracles on both devices (33 each). MLX's GPU fp32
  matmul is reduced precision on the M5 (~1e-3), so its oracle runs on CPU.

Known debt
- Multi-consumer nodes break fusion (x feeding relu AND its grad = own kernel).
- Over-budget fused groups (>16 ops / >8 inputs) fall back to op-by-op.
- Optimizers update raw buffers outside the graph (fine for now; lazy optim later).
- Exp/Log run the scalar path of the fused kernel (no SIMD exp/log yet).
- Metal: conv/pool run on the CPU (sync + fallback); ~1300 dispatches/step is
  now the limit — fewer kernels (multi-consumer fusion, permutes folded into
  GEMM strides) is the next lever for both backends. CUDA: same Backend
  interface with managed memory (cudaMallocManaged) + NVRTC for fused groups.
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
5. Metal backend ✓  (done before M4; see above). CUDA later, same interface.

References
- Paper: https://arxiv.org/html/2502.02463v3
- Reference impl (PyTorch, MIT): https://github.com/GWhittle110/distribution-transformers
- docs/tiny-inspo.md
