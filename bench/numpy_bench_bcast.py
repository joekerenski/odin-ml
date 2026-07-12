"""Benchmark NumPy broadcast patterns we SIMD-optimized in odin-ml.

   uv run numpy_bench_bcast.py

Outputs plain-text lines that compare.py can grep for:

   <op> <shape>  best=<ms>  med=<ms>  <gflops-or-gbps>

We deliberately include the SAME patterns we tested in Odin:
  - same-shape elementwise add   ([N]+[N])
  - scalar broadcast              ([N]+[1], [M,N]+[1,1])
  - row broadcast                 ([M,N]+[N])
  - col broadcast                 ([M,N]+[M,1])
"""

from __future__ import annotations

import time
import json
import numpy as np


def bench_op(label: str, body, repeats: int = 30, warmup: int = 8, batch_target_ms: float = 8.0) -> dict:
    """Run body() `batch` times per sample so each sample is long enough to time.
    Returns median ms per call and bytes-per-sec for streaming ops.
    """
    # Estimate batch size: do one call, measure, scale up
    t0 = time.perf_counter()
    body()
    one_ms = (time.perf_counter() - t0) * 1e3
    batch = max(1, int(batch_target_ms / max(one_ms, 1e-3)))
    if batch > 256:
        batch = 256

    for _ in range(warmup):
        for _ in range(batch):
            body()
    sink = 0.0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(batch):
            body()
        t1 = time.perf_counter()
        times.append((t1 - t0) / batch * 1e3)  # ms per call
        sink += 1.0
    times.sort()
    best = times[0]
    med = times[len(times) // 2]
    return {"label": label, "best_ms": best, "med_ms": med, "batch": batch, "sink": sink}


def main() -> None:
    print(f"numpy {np.__version__}  dtype=f32")
    print("broadcast benchmarks — same patterns as odin-ml SIMD tests")
    print()

    results = []
    # Match the sizes from odin-ml benchmark
    N = 1 << 20  # 1M elements
    M = 1024
    K = 1024

    # --- Same-shape add [N]+[N] ---
    a = np.random.default_rng(0).standard_normal(N, dtype=np.float32)
    b = np.random.default_rng(1).standard_normal(N, dtype=np.float32)
    r = bench_op(f"add [{N}]+[{N}]", lambda: a + b)
    results.append({**r, "op": "add", "shape": f"[{N}]+[{N}]", "gbps": (3.0 * N * 4) / r["med_ms"] / 1e6})
    print(f"  {r['label']:<40}  best={r['best_ms']:8.3f} ms  med={r['med_ms']:8.3f} ms  "
          f"{(3.0 * N * 4) / r['med_ms'] / 1e6:6.1f} GB/s")

    # --- Scalar broadcast [N]+[1] ---
    c = np.random.default_rng(2).standard_normal(1, dtype=np.float32)
    r = bench_op(f"add [{N}]+[1]", lambda: a + c)
    results.append({**r, "op": "add", "shape": f"[{N}]+[1]", "gbps": (3.0 * N * 4) / r["med_ms"] / 1e6})
    print(f"  {r['label']:<40}  best={r['best_ms']:8.3f} ms  med={r['med_ms']:8.3f} ms  "
          f"{(3.0 * N * 4) / r['med_ms'] / 1e6:6.1f} GB/s")

    # --- Row broadcast [M,N]+[N] ---
    big = np.random.default_rng(3).standard_normal((M, K), dtype=np.float32)
    row = np.random.default_rng(4).standard_normal(K, dtype=np.float32)
    r = bench_op(f"add [{M},{K}]+[{K}]", lambda: big + row)
    results.append({**r, "op": "add", "shape": f"[{M},{K}]+[{K}]", "gbps": (3.0 * M * K * 4) / r["med_ms"] / 1e6})
    print(f"  {r['label']:<40}  best={r['best_ms']:8.3f} ms  med={r['med_ms']:8.3f} ms  "
          f"{(3.0 * M * K * 4) / r['med_ms'] / 1e6:6.1f} GB/s")

    # --- Col broadcast [M,N]+[M,1] ---
    col = np.random.default_rng(5).standard_normal((M, 1), dtype=np.float32)
    r = bench_op(f"add [{M},{K}]+[{M},1]", lambda: big + col)
    results.append({**r, "op": "add", "shape": f"[{M},{K}]+[{M},1]", "gbps": (3.0 * M * K * 4) / r["med_ms"] / 1e6})
    print(f"  {r['label']:<40}  best={r['best_ms']:8.3f} ms  med={r['med_ms']:8.3f} ms  "
          f"{(3.0 * M * K * 4) / r['med_ms'] / 1e6:6.1f} GB/s")

    # --- General broadcast [M,N]+[1,1] (just scalar in disguise, should be SIMD-able) ---
    scalar_2d = np.random.default_rng(6).standard_normal((1, 1), dtype=np.float32)
    r = bench_op(f"add [{M},{K}]+[1,1]", lambda: big + scalar_2d)
    results.append({**r, "op": "add", "shape": f"[{M},{K}]+[1,1]", "gbps": (3.0 * M * K * 4) / r["med_ms"] / 1e6})
    print(f"  {r['label']:<40}  best={r['best_ms']:8.3f} ms  med={r['med_ms']:8.3f} ms  "
          f"{(3.0 * M * K * 4) / r['med_ms'] / 1e6:6.1f} GB/s")

    # --- General 3D broadcast [B,M,N]+[1,N] (still row-broadcast, tests SIMD path) ---
    B = 32
    big3 = np.random.default_rng(7).standard_normal((B, M, K), dtype=np.float32)
    row3 = np.random.default_rng(8).standard_normal((1, K), dtype=np.float32)
    r = bench_op(f"add [{B},{M},{K}]+[1,{K}]", lambda: big3 + row3)
    n_elems = B * M * K
    results.append({**r, "op": "add", "shape": f"[{B},{M},{K}]+[1,{K}]", "gbps": (3.0 * n_elems * 4) / r["med_ms"] / 1e6})
    print(f"  {r['label']:<40}  best={r['best_ms']:8.3f} ms  med={r['med_ms']:8.3f} ms  "
          f"{(3.0 * n_elems * 4) / r['med_ms'] / 1e6:6.1f} GB/s")

    print()
    print("---json---")
    print(json.dumps(results, indent=2))


def print_results_text(results: list) -> None:
    """Print results in the same format as odin_matmul_bcast for comparison."""
    for r in results:
        print(f"  {r['label']:<40}  best={r['best_ms']:8.3f} ms  med={r['med_ms']:8.3f} ms  {r['gbps']:6.1f} GB/s")


if __name__ == "__main__":
    main()