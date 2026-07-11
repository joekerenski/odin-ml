"""Compare NumPy vs Odin microbenches.

Usage (from bench/):
  uv run compare.py

Gate: production paths must reach ≥70% of NumPy for cases that take ≥0.5 ms
(median), so sub-millisecond timing noise doesn't dominate. Pure matmul is
reported as info only — we do not expect it to beat Accelerate.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = 70.0
MIN_MS = 0.2  # ignore ultra-short runs (timer noise); batching keeps most ≥ this


def run_numpy() -> dict[str, dict]:
    out = subprocess.check_output(
        ["uv", "run", "numpy_bench.py"], cwd=ROOT, text=True
    )
    results: dict[str, dict] = {}
    for line in out.splitlines():
        m = re.match(
            r"matmul\s+(\S+)\s+best=\s*([0-9.]+)\s+ms\s+med=\s*([0-9.]+)\s+ms\s+([0-9.]+)\s+GFLOP/s",
            line,
        )
        if m:
            results[f"matmul:{m.group(1)}"] = {
                "best_ms": float(m.group(2)),
                "med_ms": float(m.group(3)),
                "gflops_best": float(m.group(4)),
            }
            # gflops in the line is already median-based from the new numpy_bench
            results[f"matmul:{m.group(1)}"]["gflops"] = float(m.group(4))
            continue
        m = re.match(
            r"(add|mul)\s+n=\s*(\d+)\s+best=\s*([0-9.]+)\s+ms\s+med=\s*([0-9.]+)\s+ms\s+([0-9.]+)\s+GB/s",
            line,
        )
        if m:
            n = int(m.group(2))
            med_s = float(m.group(4)) / 1e3
            gbps_med = (3.0 * n * 4) / med_s / 1e9
            results[f"{m.group(1)}:{m.group(2)}"] = {
                "best_ms": float(m.group(3)),
                "med_ms": float(m.group(4)),
                "gbps": gbps_med,
            }
    return results


def run_odin() -> dict[str, dict]:
    out = subprocess.check_output(
        [
            "odin",
            "run",
            "odin_matmul",
            "-o:speed",
            "-no-bounds-check",
            "-disable-assert",
        ],
        cwd=ROOT,
        text=True,
    )
    results: dict[str, dict] = {}
    backend = "unknown"
    for line in out.splitlines():
        bm = re.match(r"--- backend: (\w+)", line)
        if bm:
            backend = bm.group(1)
            continue
        if line.startswith("--- elementwise"):
            backend = "Pure"
            continue
        m = re.match(
            r"matmul\s+(\d+)x(\d+)@(\d+)x(\d+)\s+best=([0-9.]+)\s+ms\s+med=([0-9.]+)\s+ms\s+([0-9.]+)\s+GFLOP/s",
            line,
        )
        if m:
            M, K, N = int(m.group(1)), int(m.group(2)), int(m.group(4))
            shape = f"{m.group(1)}x{m.group(2)}@{m.group(3)}x{m.group(4)}"
            med_s = float(m.group(6)) / 1e3
            gflops_med = (2.0 * M * K * N) / med_s / 1e9
            key = f"matmul[{backend}]:{shape}"
            results[key] = {
                "best_ms": float(m.group(5)),
                "med_ms": float(m.group(6)),
                "gflops": gflops_med,
            }
            continue
        m = re.match(
            r"(add|mul)\s+n=(\d+)\s+best=([0-9.]+)\s+ms\s+med=([0-9.]+)\s+ms\s+([0-9.]+)\s+GB/s",
            line,
        )
        if m:
            n = int(m.group(2))
            med_s = float(m.group(4)) / 1e3
            results[f"{m.group(1)}:{m.group(2)}"] = {
                "best_ms": float(m.group(3)),
                "med_ms": float(m.group(4)),
                "gbps": (3.0 * n * 4) / med_s / 1e9,
            }
    return results


def main() -> int:
    print("Running NumPy...")
    np_r = run_numpy()
    print("Running Odin...")
    od_r = run_odin()

    print()
    print(
        f"{'key':<42} {'np med':>10} {'od med':>10} {'% of np':>8}  note"
    )
    print("-" * 90)

    fails = 0
    passes = 0
    skipped = 0

    for k, npv in sorted(np_r.items()):
        if k.startswith("matmul:"):
            shape = k.split(":", 1)[1]
            for backend, gated in (("Accelerate", True), ("Pure", False)):
                od_key = f"matmul[{backend}]:{shape}"
                if od_key not in od_r:
                    if gated:
                        print(f"{od_key:<42} MISSING")
                        fails += 1
                    continue
                odv = od_r[od_key]
                # skip ultra-short (timer / turbo noise dominated)
                if max(npv["med_ms"], odv["med_ms"]) < MIN_MS:
                    print(
                        f"{od_key:<42} {npv['gflops']:10.1f} {odv['gflops']:10.1f} "
                        f"{'n/a':>7}  skip (<{MIN_MS}ms)"
                    )
                    skipped += 1
                    continue
                ratio = 100.0 * odv["gflops"] / npv["gflops"] if npv["gflops"] else 0
                if gated:
                    ok = ratio >= TARGET
                    passes += int(ok)
                    fails += int(not ok)
                    note = "OK" if ok else "LOW"
                else:
                    note = "info"
                print(
                    f"{od_key:<42} {npv['gflops']:10.1f} {odv['gflops']:10.1f} "
                    f"{ratio:7.1f}%  {note}"
                )

        else:
            if k not in od_r:
                print(f"{k:<42} MISSING")
                fails += 1
                continue
            odv = od_r[k]
            # gate only streaming-size elementwise (≈ 16M) strictly; smaller is noisy
            name, _, nstr = k.partition(":")
            n = int(nstr) if nstr.isdigit() else 0
            ratio = 100.0 * odv["gbps"] / npv["gbps"] if npv["gbps"] else 0
            if n < 4_000_000:
                print(
                    f"{k:<42} {npv['gbps']:10.1f} {odv['gbps']:10.1f} "
                    f"{ratio:7.1f}%  info (small stream)"
                )
                skipped += 1
                continue
            ok = ratio >= TARGET
            passes += int(ok)
            fails += int(not ok)
            note = "OK" if ok else "LOW"
            print(
                f"{k:<42} {npv['gbps']:10.1f} {odv['gbps']:10.1f} "
                f"{ratio:7.1f}%  {note}"
            )

    print("-" * 90)
    print(
        f"gate: ≥{TARGET:.0f}% of NumPy (median), ops ≥{MIN_MS} ms  |  "
        f"OK={passes} LOW={fails} skip={skipped}"
    )
    print("production path = Accelerate matmul (Darwin) + pure SIMD elementwise")
    print("pure matmul is educational; not expected to hit the gate")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
