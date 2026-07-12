# odin-ml microbenches

Compare our kernels to **NumPy f32** on this machine (macOS → Accelerate BLAS).

## Setup

```bash
cd bench
uv sync          # installs numpy into .venv
```

## Run

```bash
# Matmul/elementwise (large)
uv run numpy_bench.py
odin run odin_matmul -o:speed -no-bounds-check -disable-assert
uv run compare.py

# Broadcast patterns (scalar, row, col)
uv run numpy_bench_bcast.py
odin run odin_matmul_bcast -o:speed -no-bounds-check -disable-assert
uv run compare_bcast.py
```

## What we measure

| Op | Our path | Notes |
|----|----------|-------|
| `matmul` large | **Accelerate** `cblas_sgemm` (default on Darwin) | Same BLAS NumPy uses |
| `matmul` | Pure tiled `#simd` (4×8 microkernel) | Educational; ~3–8% of Accel |
| `add`/`mul` same-shape | Pure portable SIMD | ~40–50% of NumPy (allocation-bound) |
| `add`/`mul` broadcast | Scalar / row / col SIMD kernels | ~2–45% of NumPy |

## Why broadcast looks slow

The Odin broadcast bench **runs `ml.add(a, b)` per iteration**, which includes:

- `make([]f32, numel)` for the output tensor
- `new(Tensor)` for the autograd context
- `make([dynamic]^Tensor)` for parents list
- Then runs the SIMD kernel

For a 128 MB output (the 3D case), this allocator round-trip costs more than the kernel itself. NumPy uses pre-allocated scratch buffers + memory pools; it doesn't re-allocate per op.

This is the **eager-vs-lazy** issue you flagged in your lessons. A lazy tensor that allocates all buffers once at graph-build time, then reuses them at execution time, would close this gap.

## Files

- `numpy_bench.py` / `numpy_bench_bcast.py` — NumPy timings
- `odin_matmul/` — matmul + elementwise kernels
- `odin_matmul_bcast/` — broadcast SIMD kernels (scalar/row/col)
- `compare.py` — matmul gate
- `compare_bcast.py` — broadcast gate
