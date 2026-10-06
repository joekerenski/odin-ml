# odin-ml benches

## make perf — the performance plan's yardstick

```bash
ML_DEVICE=metal make perf            # or cuda / cpu; ARGS=dt_fusion runs matching workloads
ML_DEVICE=metal make perf-baseline   # (re)write bench/perf/baseline-<machine>.json — commit it
make perf-tiny                       # the bar: the same M4 step in tinygrad, default and BEAM=2
```

Workloads:
- training steps: M4 and M3 DT steps (sample + forward + backward + Adam), and
  MNIST-shaped MLP and CNN steps on synthetic data;
- micros: stream bandwidth, 2048³ GEMM, row softmax, row LayerNorm.

Each workload reports:
- step time (median / p10 / p90) and kernels per step, with deltas vs the
  committed baseline for this machine type (`<device>-<os>-<arch>`);
- one profiled step broken down by kernel type: count, device ms, achieved GB/s
  and GFLOP/s.

Every run is appended to `build/perf.jsonl`. Profiled kernels run one at a time,
so the breakdown sums to more than the step median; the median is the real number.
The same profile is available in code: `ml.profile_begin()` … `ml.profile_end()`.

## Microbenches vs NumPy

Compare our kernels to **NumPy f32** on this machine (macOS → Accelerate BLAS).

## Setup

```bash
cd bench
uv sync          # installs numpy + tinygrad into .venv
```

To compare against a local tinygrad checkout instead: `TINYGRAD_PATH=/path/to/tinygrad uv run tinygrad_compare.py …`.

Two oracles read the same dump (`make oracle` from the repo root runs both):

- `tinygrad_compare.py` — tinygrad
- `mlx_compare.py` — MLX, using its own kernels where it has them (layer_norm,
  cross_entropy, scaled_dot_product_attention, Adam/AdamW). Runs MLX on its CPU
  by default: MLX's GPU float32 matmul on the M5 is reduced precision (~1e-3
  relative); `--gpu` checks against it with rtol 3e-3.

`ML_DEVICE=metal` (or `cuda`) on the odin side checks that GPU backend against the same references.

On Linux (no MLX), `make oracle-tiny` runs the tinygrad oracle from a local checkout
(`TINYGRAD_PATH`, default `~/code/repos/tinygrad`). tinygrad's default device there may be `NV`;
`DEV=CUDA make oracle-tiny` runs it on the GPU through CUDA.

## Run

```bash
# odin-ml vs tinygrad (8 core cases + matmul/add timings)
odin run odin_vs_tiny -o:speed > odin_ref.txt
uv run tinygrad_compare.py odin_ref.txt

# Matmul/elementwise (large)
uv run numpy_bench.py
odin run odin_matmul -o:speed -no-bounds-check -disable-assert
uv run compare.py

# Broadcast patterns (scalar, row, col)
uv run numpy_bench_bcast.py
odin run odin_matmul_bcast -o:speed -no-bounds-check -disable-assert
uv run compare_bcast.py

# Odin native matrix type vs our #simd kernels + asm/IR study
odin run odin_matrix -o:speed -no-bounds-check -disable-assert
cd odin_matrix && sh build_asm.sh   # writes study.s + study_ir/, greps study procs
```

## What we measure

| Op | Our path | Notes |
|----|----------|-------|
| `matmul` large | **Accelerate** `cblas_sgemm` (default on Darwin) | Same BLAS NumPy uses |
| `matmul` | Pure tiled `#simd` (4×8 microkernel) | Educational; ~3–8% of Accel |
| `add`/`mul` same-shape | Fused ewise kernel, single-op SIMD path | allocation-bound |
| `add`/`mul` broadcast | Fused ewise kernel, row / col / block loads | allocation-bound |

## Why broadcast looks slow

The Odin broadcast bench **runs `ml.add(a, b)` per iteration**, which includes:

- `new(UOp)` + shape/src slices for the graph node
- `make([]f32, numel)` for the output buffer
- Then runs the fused SIMD kernel

For a 128 MB output (the 3D case), this allocator round-trip costs more than the kernel itself. NumPy uses pre-allocated scratch buffers + memory pools; it doesn't re-allocate per op.

This is the **eager-vs-lazy** issue you flagged in your lessons. A lazy tensor that allocates all buffers once at graph-build time, then reuses them at execution time, would close this gap.

## Files

- `numpy_bench.py` / `numpy_bench_bcast.py` — NumPy timings
- `odin_matmul/` — matmul kernels + elementwise via `ml.add`/`ml.mul` + realize
- `odin_matmul_bcast/` — broadcast adds (scalar/row/col) through the graph
- `odin_matrix/` — **native `matrix[M,N]T` vs plain loops vs explicit #simd**, plus an
  asm / LLVM-IR study (`build_asm.sh`) showing what the compiler emits
- `metal_dispatch/` — Metal dispatch overhead: per-kernel commit vs one command buffer, vs CPU
- `oracle.py` — shared by both oracles: dump parser, deterministic inputs, report
- `compare.py` — matmul gate
- `compare_bcast.py` — broadcast gate

## What `odin_matrix` tells you

- Odin's native `matrix[M, N]T` is **capped at 64 elements** (`matrix[8,8]f32`
  is the largest square). It is for small linear algebra, not tensors.
- `a + b` on a native matrix compiles to fully unrolled NEON (`fadd.4s`).
- A **plain scalar `for` loop also auto-vectorizes** to NEON (see
  `_study_plain_add` in `study.s`: `ldp`/`fadd.4s`/`stp`, 8 floats per iter).
  At streaming size our explicit `#simd` kernels and plain loops both run
  ~110–140 GB/s (memory-bound) — within a few % of each other.
- Takeaway: for simple contiguous elementwise ops, LLVM gives you the SIMD;
  hand-written `#simd` matters where the auto-vectorizer gives up (strided
  access, reductions, microkernels like `kernel_matmul.odin`).

