#!/usr/bin/env python3
"""Compare odin-ml vs local tinygrad (correctness + a few timings).

  cd bench
  odin run odin_vs_tiny -o:speed > /tmp/odin_ref.txt
  uv run tinygrad_compare.py /tmp/odin_ref.txt
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

TINY = Path("/Users/joe/code/repos/tinygrad")
sys.path.insert(0, str(TINY))
from tinygrad import Tensor  # noqa: E402


def parse_odin(path: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    results, times = {}, {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("RESULT "):
            name = line.split(" ", 1)[1]
            i += 1
            vals = np.array([float(x) for x in lines[i].split()], dtype=np.float32)
            results[name] = vals
        elif line.startswith("TIME "):
            _, name, ms = line.split()
            times[name] = float(ms)
        i += 1
    return results, times


def close(a: np.ndarray, b: np.ndarray, rtol=1e-4, atol=1e-5) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol)


def tiny_results() -> dict[str, np.ndarray]:
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[10.0, 20.0], [30.0, 40.0]])
    out = {
        "add_same": (a + b).numpy().reshape(-1),
        "mul_same": (a * b).numpy().reshape(-1),
    }

    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row = Tensor([10.0, 100.0, 1000.0])
    out["mul_row"] = (A * row).numpy().reshape(-1)

    col = Tensor([[1.0], [2.0]])
    out["add_col"] = (A + col).numpy().reshape(-1)

    M = Tensor([[1.0, 2.0], [3.0, 4.0]])
    N = Tensor([[5.0, 6.0], [7.0, 8.0]])
    out["matmul_2x2"] = (M @ N).numpy().reshape(-1)

    X = Tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]])
    w = Tensor([[1.0], [-1.0]])
    bias = Tensor([[0.5]])
    out["relu_xw_b"] = (X @ w + bias).relu().numpy().reshape(-1)

    x = Tensor([0.0, 1.0, 2.0, 3.0])
    y = Tensor([1.0, 3.5, 6.0, 8.5])
    ww, bb = Tensor([2.5]), Tensor([1.0])
    pred = x * ww + bb
    out["mse_mean"] = ((pred - y).square().mean()).numpy().reshape(-1)

    # L = mean((x*w)^2); dL/dw
    xw = Tensor([1.0, 2.0, 3.0, 4.0])
    pw = Tensor([0.5])
    pw.requires_grad = True
    loss = (xw * pw).square().mean()
    loss.backward()
    out["grad_w"] = pw.grad.numpy().reshape(-1)
    return out


def bench_tiny() -> dict[str, float]:
    times = {}
    rng = np.random.default_rng(0)
    for n in (256, 512):
        A = Tensor(rng.standard_normal((n, n), dtype=np.float32))
        B = Tensor(rng.standard_normal((n, n), dtype=np.float32))
        (A @ B).realize()
        best = math.inf
        for _ in range(5):
            C = A @ B
            t0 = time.perf_counter()
            C.realize()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        times[f"matmul_{n}"] = best

    n = 1 << 20
    A = Tensor(rng.standard_normal(n, dtype=np.float32))
    B = Tensor(rng.standard_normal(n, dtype=np.float32))
    (A + B).realize()
    best = math.inf
    for _ in range(10):
        C = A + B
        t0 = time.perf_counter()
        C.realize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    times[f"add_{n}"] = best
    return times


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    odin_r, odin_t = parse_odin(sys.argv[1])
    tiny_r = tiny_results()
    tiny_t = bench_tiny()

    print("=== correctness (odin vs tinygrad) ===")
    failed = 0
    for name, want in tiny_r.items():
        got = odin_r.get(name)
        if got is None:
            print(f"  MISS  {name}")
            failed += 1
            continue
        ok = close(got, want)
        if ok:
            print(f"  ok    {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}")
            print(f"        odin  {got}")
            print(f"        tiny  {want}")

    print("\n=== speed (best ms; % of tinygrad) ===")
    for name in sorted(set(odin_t) | set(tiny_t)):
        o, t = odin_t.get(name), tiny_t.get(name)
        if o is None or t is None:
            print(f"  {name:16s}  odin={o}  tiny={t}")
            continue
        pct = 100.0 * t / o if o > 0 else float("nan")
        print(f"  {name:16s}  odin={o:8.3f} ms  tiny={t:8.3f} ms  ({pct:5.1f}% of tiny)")

    print(f"\n=== {len(tiny_r) - failed}/{len(tiny_r)} correct ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
