# odin-ml microbenches

Compare our kernels to **NumPy f32** on this machine (macOS → Accelerate BLAS).

## Setup

```bash
cd bench
uv sync          # installs numpy into .venv
```

## Run

```bash
# NumPy alone
uv run numpy_bench.py

# Odin alone (Accel + pure + elementwise)
odin run odin_matmul -o:speed -no-bounds-check -disable-assert

# Side-by-side gate (target ≥70% of NumPy median on ≥0.5 ms ops)
uv run compare.py
```

## What we measure

| Op | Our path | Notes |
|----|----------|-------|
| `matmul` large | **Accelerate** `cblas_sgemm` (default on Darwin) | Same BLAS NumPy uses |
| `matmul` | Pure tiled `#simd` (4×8 microkernel) | Educational; ~3–8% of Accel for now |
| `add` / `mul` | Pure portable SIMD (`f32x4` → NEON) | Memory-bound, ~90–100% of NumPy |

## Perf rule of thumb

Production path should stay **≥70% of NumPy** on this system for ops that take ≥0.5 ms
(so we don't let sub-millisecond timer noise decide).

## Files

- `numpy_bench.py` — NumPy timings
- `odin_matmul/` — Odin harness calling `ml.matmul_f32` / elementwise kernels
- `compare.py` — parse both + report %
