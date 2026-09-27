"""Shared by the oracles (tinygrad_compare.py, mlx_compare.py): read odin-ml's
RESULT/TIME dump, the deterministic inputs both sides use, and the report."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_odin(path: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    results, times = {}, {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("RESULT "):
            name = line.split(" ", 1)[1]
            i += 1
            results[name] = np.array([float(x) for x in lines[i].split()], dtype=np.float32)
        elif line.startswith("TIME "):
            _, name, ms = line.split()
            times[name] = float(ms)
        i += 1
    return results, times


def seq_np(shape, k) -> np.ndarray:
    """sin(0.7*i + k) — the same inputs odin_vs_tiny's seq() builds."""
    n = int(np.prod(shape))
    return np.sin(np.arange(n) * 0.7 + k).astype(np.float32).reshape(shape)


def close(a: np.ndarray, b: np.ndarray, rtol=1e-4, atol=1e-5) -> bool:
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol)


def report(oracle: str, odin_r: dict[str, np.ndarray], ref: dict[str, np.ndarray], rtol=1e-4, atol=1e-5) -> int:
    print(f"=== correctness (odin vs {oracle}) ===")
    failed = 0
    for name, want in ref.items():
        got = odin_r.get(name)
        if got is None:
            print(f"  MISS  {name}")
            failed += 1
        elif close(got, want.reshape(-1), rtol, atol):
            print(f"  ok    {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}\n        odin  {got}\n        {oracle:5s} {want.reshape(-1)}")
    print(f"\n=== {len(ref) - failed}/{len(ref)} correct ===")
    return failed
