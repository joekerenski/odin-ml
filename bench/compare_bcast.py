"""Side-by-side comparison: NumPy broadcast vs Odin-ML broadcast.

   uv run compare_bcast.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = 70.0  # % of numpy


def run_numpy() -> dict[str, dict]:
    """Run numpy_bench_bcast.py and parse its lines."""
    out = subprocess.check_output(
        ["uv", "run", "numpy_bench_bcast.py"], cwd=ROOT, text=True
    )
    results: dict[str, dict] = {}
    for line in out.splitlines():
        # format: "  add [1048576]+[1048576]                   best=   0.098 ms  med=   0.101 ms   124.6 GB/s"
        m = re.match(
            r"\s*add \[(\d+(?:,\d+)*)\]\+\[(\d+(?:,\d+)*)\]\s+best=\s*([0-9.]+)\s+ms\s+med=\s*([0-9.]+)\s+ms\s+([0-9.]+)\s+GB/s",
            line,
        )
        if m:
            label = f"add [{m.group(1)}]+[{m.group(2)}]"
            results[label] = {
                "best_ms": float(m.group(3)),
                "med_ms": float(m.group(4)),
                "gbps": float(m.group(5)),
            }
    return results


def run_odin() -> dict[str, dict]:
    """Run odin_matmul_bcast and parse."""
    out = subprocess.check_output(
        [
            "odin",
            "run",
            "odin_matmul_bcast",
            "-o:speed",
            "-no-bounds-check",
            "-disable-assert",
        ],
        cwd=ROOT,
        text=True,
    )
    results: dict[str, dict] = {}
    for line in out.splitlines():
        # format: "  add [1048576]+[1048576]                   best=0.337 ms  med=0.415 ms  30.3 GB/s"
        m = re.match(
            r"\s*add \[(\d+(?:,\d+)*)\]\+\[(\d+(?:,\d+)*)\]\s+best=([0-9.]+)\s+ms\s+med=([0-9.]+)\s+ms\s+([0-9.]+)\s+GB/s",
            line,
        )
        if m:
            label = f"add [{m.group(1)}]+[{m.group(2)}]"
            results[label] = {
                "best_ms": float(m.group(3)),
                "med_ms": float(m.group(4)),
                "gbps": float(m.group(5)),
            }
    return results


def main() -> int:
    print("Running NumPy broadcast bench...")
    np_r = run_numpy()
    print("Running Odin broadcast bench...")
    od_r = run_odin()

    print()
    print(f"{'pattern':<35} {'numpy med':>12} {'odin med':>12} {'% of np':>8} {'% of np (best)':>14}  pass?")
    print("-" * 95)

    passes = 0
    fails = 0
    skipped = 0

    for label, npv in sorted(np_r.items()):
        odv = od_r.get(label)
        if odv is None:
            print(f"{label:<35} MISSING in odin")
            fails += 1
            continue
        # Compare GB/s (higher is better — memory throughput)
        ratio_med = 100.0 * odv["gbps"] / npv["gbps"] if npv["gbps"] else 0
        ratio_best = 100.0 * (npv["best_ms"] / odv["best_ms"]) * odv["gbps"] / npv["gbps"] if npv["gbps"] else 0
        # ratio_best is basically same — skip
        ok = ratio_med >= TARGET
        passes += int(ok)
        fails += int(not ok)
        flag = "OK" if ok else "LOW"
        print(
            f"{label:<35} {npv['gbps']:8.1f} GB/s {odv['gbps']:8.1f} GB/s {ratio_med:6.1f}%                {flag}"
        )

    print("-" * 95)
    print(f"gate: ≥{TARGET:.0f}% of NumPy median GB/s  |  OK={passes} LOW={fails} skip={skipped}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())