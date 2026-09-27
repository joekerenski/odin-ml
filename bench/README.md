# odin-ml microbenches

Compare our kernels to **NumPy f32** on this machine (macOS → Accelerate BLAS).

## Setup

```bash
cd bench
uv sync          # installs numpy into .venv
```

Local tinygrad: `/Users/joe/code/repos/tinygrad` (on `sys.path` in the compare script).

## Run

```bash
# odin-ml vs tinygrad (8 core cases + matmul/add timings)
odin run odin_vs_tiny -o:speed > /tmp/odin_ref.txt
uv run tinygrad_compare.py /tmp/odin_ref.txt

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

