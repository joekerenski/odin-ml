"""Benchmark NumPy f32 matmul / elementwise ops. Run: uv run numpy_bench.py"""

from __future__ import annotations

import time
import numpy as np


def bench_matmul(m: int, k: int, n: int, repeats: int = 20, warmup: int = 8) -> dict:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)

    # batch small problems so each sample is several ms
    flops = 2.0 * m * k * n
    batch = 1
    while flops * batch / 1e9 < 8.0 and batch < 64:
        batch *= 2

    for _ in range(warmup):
        for _ in range(batch):
            c = a @ b
    sink = float(c[0, 0])

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(batch):
            c = a @ b
        t1 = time.perf_counter()
        times.append((t1 - t0) / batch)
        sink += float(c[0, 0])

    times.sort()
    best = times[0]
    med = times[len(times) // 2]
    gflops = (2.0 * m * k * n) / med / 1e9  # median-based
    return {
        "op": "matmul",
        "shape": f"{m}x{k}@{k}x{n}",
        "best_ms": best * 1e3,
        "med_ms": med * 1e3,
        "gflops": gflops,
        "sink": sink,
        "batch": batch,
    }


def bench_elem(n: int, op: str, repeats: int = 50, warmup: int = 10) -> dict:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(n, dtype=np.float32)
    b = rng.standard_normal(n, dtype=np.float32)

    def run():
        if op == "add":
            return a + b
        if op == "mul":
            return a * b
        raise ValueError(op)

    for _ in range(warmup):
        c = run()
    sink = float(c[0])

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        c = run()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        sink += float(c[0])

    times.sort()
    best = times[0]
    med = times[len(times) // 2]
    # stream: 3 * n * 4 bytes (read a, read b, write c); median-based
    gbps = (3.0 * n * 4) / med / 1e9
    return {
        "op": op,
        "shape": f"{n}",
        "best_ms": best * 1e3,
        "med_ms": med * 1e3,
        "gbps": gbps,
        "sink": sink,
    }


def main() -> None:
    print(f"numpy {np.__version__}  dtype=f32")
    print(f"blas info: {np.show_config.__name__}")
    try:
        # short config dump
        np.__config__
        dll = np._core._multiarray_umath.__file__  # type: ignore[attr-defined]
        print(f"numpy path: {dll}")
    except Exception:
        pass

    results = []
    for m, k, n in [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (64, 1024, 64),   # skinny
        (1024, 64, 1024), # fat K small
    ]:
        r = bench_matmul(m, k, n)
        results.append(r)
        print(
            f"matmul {r['shape']:>22}  best={r['best_ms']:8.3f} ms  "
            f"med={r['med_ms']:8.3f} ms  {r['gflops']:7.1f} GFLOP/s  (batch={r['batch']})"
        )

    for n in [1 << 20, 1 << 22, 1 << 24]:
        for op in ("add", "mul"):
            r = bench_elem(n, op)
            results.append(r)
            print(
                f"{op:6} n={n:>10}  best={r['best_ms']:8.3f} ms  "
                f"med={r['med_ms']:8.3f} ms  {r['gbps']:6.1f} GB/s"
            )

    # machine-readable summary for our Odin compare script
    print("---json---")
    import json

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
