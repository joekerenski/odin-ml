Objective
Small-model training lib in Odin (tinygrad as inspiration + oracle): Tensor → UOp IR → kernels.
North star: reimplement Distribution Transformers (arXiv 2502.02463) — amortised Bayesian
inference, prior GMM + observations → posterior GMM, ~0.43M params. Refine the lib on the way.
UI (odin-ui-v2) comes after the strategy fits.

Decisions
- One flow: Tensor (lazy) → UOp DAG → fuse → kernels. No second graph IR. Below
  the scheduler: one kernel IR (performance plan stage 2) that all backends render.
- MatMul stays a primitive. No mul+sum → GEMM recovery pass.
- Performance: a lean kernel compiler. Explicit fusion rules, a small knob space
  searched on the device and cached; GEMM stays a hand-written, parameterized
  template. No general rewrite engine (tinygrad has one; we take ~20% of it).
- ReLU/Sigmoid/softmax/LayerNorm etc. are UOp compositions, not primitives.

Layout
  ml/                 the library
  examples/tour/      the API at every level (01 tensor … 05 nn), always runnable
  examples/           regression, mnist (MLP), mnist_cnn
  tests/              tensor_ops (127 checks, any device), metal (Metal vs CPU parity)
  dt/                 the paper project (Distribution Transformers), see dt/README.md
  bench/              loop bench, matmul benches, tinygrad/numpy compare
  docs/, studies/     notes
  Run from repo root via make: `make test`, `make mnist`, `make cnn` (see Makefile)

The UOp model (ml/uop.odin)
- `Tensor :: UOp`. One node type: op, src, arg, shape, data, requires_grad, grad.
- Buffers are always dense row-major. Permute and Reshape are views: strides
  over a base buffer, read in place by elementwise kernels, reductions and GEMMs
  (strided operands); a GEMM whose output is only permuted stores straight into
  the permuted layout. What can't be addressed is copied (ML_VIEWS=0: always).
- ops.odin builds nodes; compositions (relu, sigmoid, softmax family,
  layer_norm, cross_entropy) are just ops of ops. detach() stops grads.
- Axes as in numpy: negative counts from the end. sum(x) / mean(x) / max_all(x)
  reduce everything; sum(x, axis) keeps the reduced dim as 1.
- MatMul is batched ([..., M, K] @ [..., K, N], batch dims broadcast); a 2D
  weight folds the batch into one GEMM. GEMM operands are strided (≤ 2 batch
  dims), so mT and attention head permutes cost nothing.
- autograd.odin: grad rules emit UOps; backward realizes loss + all leaf grads
  in ONE schedule, so backward is fused like forward.
- realize.odin is the only executor: topo → plan views → fuse (elementwise
  groups, reductions with prologue / epilogue) → run kernels.
- IR ops: Input Const | Add Sub Mul Div Max CmpLt Neg Exp Log Sqrt Expand |
  Sum ReduceMax | Reshape Permute | MatMul Conv2d MaxPool2d (+ conv/pool bwd)
- checkpoint.odin: save/load params as safetensors (+ string metadata); names default
  to param order. Interop-checked by loading in MLX.
- optim.odin: SGD, Adam/AdamW (tinygrad semantics), cosine_lr with warmup;
  Optimizer union drives Trainer. nn: Linear, Conv2d, LayerNorm.

Numbers (M5, after unification; before in parens)
- MNIST MLP 20 epochs 3.6 s (14.0 s), 98.0%. MNIST CNN 3 epochs 17.3 s (49.4 s), 98.2%.
- bench/loop mlp_step 0.15 ms (0.87 ms). 41/41 match tinygrad and MLX (ops, grads,
  attention fwd+bwd, Adam/AdamW trajectories).
- CPU backend: fused ewise kernel runs per 256-element chunk; parallel_for on
  a persistent worker pool (fused, permute, reduce, batched GEMM, optimizers,
  conv/pool loops); kernel outputs allocated non-zeroed; dt arenas use 64 MB
  warm blocks. Constants are immediates in fused groups (not input slots).
  Within a realize, dead backward buffers are reused (cache-hot) for later
  outputs (realize.odin; ML_REUSE=0 turns it off).
- x86-64 (i5-13500, 20 threads, Linux): AVX2+FMA GEMM picked at runtime by
  CPUID (kernel_gemm_amd64.odin: packed 6×16 micro-kernel, tiles on all cores,
  split-K for long K, batched tiny-GEMM path) — a plain build is fast, no
  -microarch flag. The pool spins briefly between jobs (5 µs/job at 20
  threads vs 13 µs waking sleepers). Subnormals flushed (FTZ/DAZ).
  DT fusion step 80 ms (M5 86), table1 86 ms (M5 84), MNIST MLP 3.2 s
  (M5 3.6), CNN 8.9 s (M5 17.3); mlp_step 0.26–0.35 ms (M5 0.15: latency-bound,
  two 25 MFLOP GEMMs at ~50 µs each). Before: 5700 ms DT step (the generic
  x86 build lowered every fused_mul_add lane to a libm fmaf call).
  ML_THREADS=n overrides the thread count. Most kernels are now DRAM-bound
  here (~50 GB/s vs the M5's unified memory).
- Metal backend (ml/backend_metal_darwin.odin): ML_DEVICE=metal, same code.
  Unified memory: ml.arena_init arenas are zero-copy; other memory is staged.
  A realize encodes into command buffers committed every 128 kernels (GPU runs
  while the CPU encodes), synced once. Fused groups → generated MSL, cached by
  shape hash. GEMM on simdgroup_matrix 8×8 units, generated per tile and
  picked per shape by timing on the device (backend_metal_gemm_darwin.odin),
  split-K for long K, a tiny kernel for attention heads. Conv/pool: CPU fallback.
  DT step 86 ms (CPU) → 34 ms (Metal).
- CUDA backend (ml/backend_cuda_linux.odin): ML_DEVICE=cuda, same code. libcuda,
  NVRTC, cuBLAS loaded at runtime (a build without them still runs). Managed
  memory; on CUDA the arena sends allocations ≥ 4 KB out of band to a caching
  allocator, so kernel outputs stay on GPU pages and host-written graph nodes /
  batches on host pages (no page ping-pong); CPU fallbacks and the optimizer
  bulk-prefetch what they touch (Backend.to_host). Heap tensors are staged via
  pinned memory. Fused groups → CUDA C → NVRTC cubin, cached by program hash
  (~1 s of compiles on the first DT step). GEMM: cuBLAS fp32 (no TF32); tiny
  batched (attention heads) on a one-thread-per-output kernel.
  RTX 4090: DT fusion 13 ms/step (GPU 7 ms of it), table1 15.6 ms/step;
  tinygrad on the same GPU 41.6 ms/step (BEAM=0, 1300 kernels).
  MNIST MLP 8.1 s and CNN 16.7 s (conv/pool on the CPU) — slower than the CPU
  backend at these sizes.
- Checks: tests on CPU, Metal, CUDA (138), GPU-vs-CPU parity over every kernel
  path, heap and arena memory, CPU vs the reference path (views copied,
  nothing recomputed) (tests/parity: 78 on Metal), tinygrad oracle
  41/41 with odin on CPU and CUDA vs tinygrad on CUDA; whole-model DT oracle
  agrees to 1e-6 on both (dt/tinygrad/fusion.py check). MLX's GPU fp32
  matmul is reduced precision on the M5 (~1e-3), so its oracle runs on CPU.

Known debt
- Fusion places a group at its last member's position; a group-level cycle
  check (then a topo sort of groups) would also merge e.g. bias-grad sums into
  the kernel producing the grad (~50 standalone column sums per M4 step).
- Optimizers update raw buffers outside the graph (fine for now; lazy optim later).
- Buffer reuse covers backward-built nodes only: forward intermediates may be
  read by the caller after backward, so they keep their (cold) buffers.
- Exp/Log run the scalar path of the fused kernel (no SIMD exp/log yet).
- Metal and CUDA: conv/pool run on the CPU (sync + fallback); ~1300
  dispatches/step is now the limit — fewer kernels (multi-consumer fusion,
  permutes folded into GEMM strides) is the next lever for every backend.
- CUDA: ~4.5 ms/step of launch (encode) time overlaps 7 ms of GPU time; CUDA
  Graphs (record a step, replay it) and Adam on the GPU are the next levers.
  NVRTC programs aren't cached on disk (first step compiles ~1 s). The caching
  allocator never returns blocks to the driver.
- A DT step is ~1200 kernels; multi-consumer fusion would cut that a lot.

Performance plan (locked 2026-10-06): proper fusion, then fast kernels by search
Oracle and bar: tinygrad on the same step (dt/tinygrad), default and BEAM=2.
Workload: one M4 training step, ~9.3 GFLOP, ~400 MB of activation traffic.
- M5 (10-core GPU): measured 126 GB/s (spec 153); our fp32 GEMM 2.0 TFLOPS
  (fp32 shader peak est. ~4–4.5). Floor ~6–8 ms/step.
  Was 28.5 ms; now ~15.4 ms (PR 9). Target ≤ 15 ms (tinygrad BEAM 16.6);
  stretch ~8 ms.
- RTX 4090: 82.6 TFLOPS fp32, 1008 GB/s. Floor ~1.5–2.5 ms/step, set by kernel
  count × launch cost. Now 13 ms. Target < 6 ms; stretch ~2 ms.
- Kernels per step 1484 → ~600.
One branch + PR per stage, measured with `make perf` on the M5; CUDA checked on
the 4090 before merge:
1. Measurement: make perf (fixed workloads, per-kernel-type profile with
   bandwidth/FLOP rates, JSON log), tinygrad comparison  [PR 1]
   M5 baselines in bench/perf/baseline-*.json (Metal: M4 27.5 ms, 1302 kernels;
   45% fused elementwise, 31% GEMM, 15% reduce, 9% permute)
2. Kernel IR: one representation for every fused kernel (index space, loads,
   expression DAG, reduce accumulators, stores); all backends render from it  [PR 2]
   ml/kernel_ir.odin, kernel_render.odin (one renderer for Metal + CUDA),
   kernel_cpu.odin. Same kernels and plans as before; make check-cuda-render
   parses every generated CUDA source with clang where there is no GPU.
3. Multi-consumer fusion with a cycle check  [PR 3]
   A merge must keep "run at the last member" valid: every outside user of a
   member comes after the group. Budgets checked at merge time (no op-by-op
   fallback). M4: 1302 → 1089 kernels, Metal 27.5 → 24.6 ms. Left: [B,T,1]
   per-row values after reductions (PR 4), bias/residual adds around GEMMs
   (PR 6), 16-op budget hits (PR 6).
4. Reduction fusion: elementwise into reductions, row kernels
   (softmax/LayerNorm/logsumexp in one)  [PR 4]
   Prologue (input-shaped producers, values others need stored from inside the
   loop) and epilogue (output-shaped consumers). New GPU plans: Reduce_Simd (a
   32-lane group per row, hardware reduce) and the split for fused column
   reductions (pass A prologue + partial, pass B combine + epilogue). CPU runs
   fused reductions as prologue pass → reduce_block → epilogue pass.
   M4: 1089 → 802 kernels, Metal 24.6 → 23.7 ms; LayerNorm fwd −34%.
5. Views: per-input strides; Permute/Expand become index transforms  [PR 5]
   Permute / Reshape compose into strides over a base buffer (plan_views);
   read in place when every consumer can address them (elementwise: any;
   reductions: merges into [outer, r, inner]; GEMM: strided operands, ≤ 2
   batch dims on every backend). A GEMM whose output is only rearranged
   stores into the target layout (attention's head merge). Expand was already
   an index transform (stride 0) in fused kernels.
   M4: 802 → 682 kernels, no copies left; Metal 23.7 → 22.1 ms, CPU 42.8 → 40.5 ms.
6. No op/input budgets; rematerialization  [PR 6]
   Budgets 16 ops / 12 inputs → 64 / 24, plus the real limit: ≤ 28 bound
   buffers (inputs + stores; Metal binds 31). Rematerialization (remat.odin):
   a value other kernels read is recomputed in each when its expression reads
   ≤ 1 full-size buffer (the cone is cloned into the reading group) — e.g.
   GELU's intermediates are recomputed in the backward pass from the
   pre-activation instead of 8 stores + reads. On by default on GPUs only
   (the CPU interpreter's scalar exp/log make recompute dearer than the
   traffic); ML_REMAT=0|1 forces it.
   M4: 682 → 672 kernels; Metal 22.1 → ~20 ms (fused elementwise 2.7 → 1.3 ms
   GPU time); CPU 40.5 → 42.3 ms (bigger groups). Now ~16 ms of GPU work
   (GEMM 9, reductions 5.5, elementwise 1.3) and ~4 ms of host scheduling
   outside the GPU window per step. GEMM epilogues moved to stage 7, where
   GEMMs become generated kernels (upside now < 1 ms).
7. Lowering knobs (workgroup, upcast, unroll, reduce strategy, GEMM tiles) +
   kernel search on device + disk cache of choices and binaries; GEMM
   epilogues (bias + activation)  [PR 7: GEMM part]
   Metal GEMMs are generated per tile (32/64 × 32/64), each staged through
   threadgroup memory or loading 8×8 fragments straight from device memory
   (no barriers; wins most of our K = 64 shapes), plus the tiny kernel and
   split-K. Per problem the candidates are timed on the device (rotating
   operand copies so reads are cold, like in the step) and the choice is
   cached in ~/.cache/odin-ml/<device>.txt (search.odin; ML_SEARCH=0: fixed
   heuristic). First run of a model ~3.5 s of search on the M5. Parity runs
   every variant (ml.gemm_force).
   M4: GEMM GPU time 9.2 → 7.8 ms; step ~20 → 19.6 ms. The step moves less
   than the kernels: ~4 ms of host scheduling runs before the GPU starts, and
   854 dispatches on a serial encoder each wait for the previous one (stage 8).
   Not done: reduce / elementwise knobs, CUDA (cuBLAS stays), on-disk
   binaries. GEMM epilogues: stage 8 / PR 9.
8. Record and replay (Metal indirect command buffers, CUDA Graphs) + optimizer
   as UOps on the device  [PR 8: replay of scheduling, concurrent dispatch]
   Schedule cache (schedule_cache.odin): a step's graph has the same structure
   every time, so the scheduler's decisions (views, groups, clones, dead
   members, consumer counts, run order) are recorded by topo position and
   replayed onto the next graph with the same structure key (compared in
   full). Host scheduling 3.4 → ~1 ms per M4 step. ML_SCHED_CACHE=0 off.
   Metal: a concurrent encoder with memory barriers only where a kernel reads
   what an earlier one (since the last barrier) wrote, or writes what it read
   or wrote; kernels run in level order (longest kernel chain from the inputs)
   so neighbours are independent: 854 dispatches, 523 barriers.
   ML_METAL_SERIAL=1 for the serial encoder.
   M4: Metal 19.6 → 16.65 ms (≈ 12.9 ms GPU window + ~1 ms scheduling +
   ~2.5 ms sampling, graph building, encoding of the first batch, Adam).
   Then [PR 9]: GEMM epilogues on Metal — a MatMul whose result only one
   elementwise group reads (directly or through dense views) runs with the
   group in its store loop (Backend.matmul_epi; the result is never stored):
   the q/k/v projection bias adds, 672 → 634 kernels. Replay without maps
   (UOp.epoch / pos), the first command buffer committed after 16 kernels.
   M4: Metal 16.65 → ~15.4 ms; M3 22.4 → 20.5 ms.
   Not done: kernel-list replay (ICB / CUDA Graphs), Adam on the device (0.16
   ms on the M5's unified memory; matters on CUDA), epilogues on CUDA
   (cuBLAS) and after split-K / tiny GEMMs.

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
4. Sensor fusion ✓  dt/fusion: GMM-token DT (full covariance, precision-Cholesky),
   the paper's tracking problem, used as the update step of a filter; exact GMM predict.
   Eval: single-update E[KL] vs importance-sampled exact posterior; 100×100 filtering vs
   bootstrap PFs (1k/5k/50k), with NLL by time step. Model library: ml.save/ml.load
   (safetensors), all experiments load from models/ or train+save. Lib: gelu, tanh,
   clip, minimum; sigmoid no longer NaNs in the backward pass for x < −88.
5. Metal backend ✓  (done before M4; see above). CUDA ✓ (Linux, RTX 4090), same interface.

References
- Paper: https://arxiv.org/html/2502.02463v3
- Reference impl (PyTorch, MIT): https://github.com/GWhittle110/distribution-transformers
- docs/tiny-inspo.md
- docs/research-directions.md — later: continual learning, latent world models, scene memory
