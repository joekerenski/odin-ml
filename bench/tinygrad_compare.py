#!/usr/bin/env python3
"""Compare odin-ml vs local tinygrad (correctness + a few timings).

  cd bench
  odin run odin_vs_tiny -o:speed > odin_ref.txt
  uv run tinygrad_compare.py odin_ref.txt
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# tinygrad from the environment (bench deps), or a local checkout via TINYGRAD_PATH
if os.environ.get("TINYGRAD_PATH"):
    sys.path.insert(0, os.environ["TINYGRAD_PATH"])
from tinygrad import Tensor  # noqa: E402

from oracle import close, parse_odin, seq_np  # noqa: E402


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

    # conv2d NCHW
    cx = Tensor([[[ [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0] ]]])
    cw = Tensor([[[ [1.0, 1.0], [1.0, 1.0] ]]])
    out["conv2d_k2"] = cx.conv2d(cw, stride=1, padding=0).numpy().reshape(-1)

    px = Tensor([[[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]]])
    out["maxpool2d"] = px.max_pool2d(kernel_size=2).numpy().reshape(-1)

    gx = Tensor([[[ [1.0, 2.0], [3.0, 4.0] ]]])
    gw = Tensor([[[ [0.5, 0.5], [0.5, 0.5] ]]])
    gx.requires_grad = True
    gw.requires_grad = True
    gy = gx.conv2d(gw, stride=1, padding=0).sum()
    gy.backward()
    out["conv_dW"] = gw.grad.numpy().reshape(-1)
    return out


def seq(shape, k, requires_grad=False) -> Tensor:
    t = Tensor(seq_np(shape, k))
    if requires_grad:
        t.requires_grad = True
    return t


def g(t: Tensor) -> np.ndarray:
    return t.grad.numpy().reshape(-1)


def milestone1() -> dict[str, np.ndarray]:
    out = {}
    x = seq((4,), 0) * 0.5 + 1
    xl = Tensor(x.numpy())
    xl.requires_grad = True
    (xl.log() * xl.sqrt()).sum().backward()
    out["log_sqrt_dx"] = g(xl)

    mx = Tensor([[1.0, 5.0, 5.0], [2.0, 0.0, -1.0]])
    mx.requires_grad = True
    mm = mx.max(axis=1, keepdim=True)
    out["max_axis"] = mm.numpy().reshape(-1)
    (mm * Tensor([[1.0], [2.0]])).sum().backward()
    out["max_axis_dx"] = g(mx)

    s = seq((3, 4), 1, True)
    out["softmax"] = s.softmax(1).numpy().reshape(-1)
    out["log_softmax"] = s.log_softmax(1).numpy().reshape(-1)
    out["logsumexp"] = s.logsumexp(1, keepdim=True).numpy().reshape(-1)
    (s.softmax(1) * seq((3, 4), 2)).sum().backward()
    out["softmax_dx"] = g(s)

    ln = seq((2, 5), 3, True)
    out["layernorm"] = ln.layernorm().numpy().reshape(-1)
    (ln.layernorm() * seq((2, 5), 4)).sum().backward()
    out["layernorm_dx"] = g(ln)

    lg = seq((3, 4), 5, True)
    ce = lg.sparse_categorical_crossentropy(Tensor([2, 0, 3]))
    out["cross_entropy"] = ce.numpy().reshape(-1)
    ce.backward()
    out["cross_entropy_dx"] = g(lg)

    ba, bb = seq((2, 2, 3), 6, True), seq((2, 3, 2), 7, True)
    bw, bc = seq((3, 2), 8, True), seq((1, 3, 2), 9, True)
    out["bmm"] = (ba @ bb).numpy().reshape(-1)
    ((ba @ bb).sum() + ((ba @ bw) * 2).sum() + ((ba @ bc) * 3).sum()).backward()
    out["bmm_da"], out["bmm_db"], out["bmm_dw"], out["bmm_dc"] = g(ba), g(bb), g(bw), g(bc)

    q, k, v = seq((2, 3, 4), 10, True), seq((2, 3, 4), 11, True), seq((2, 3, 4), 12, True)
    o = ((q @ k.transpose(-1, -2)) * 0.5).softmax(2) @ v
    out["attention"] = o.numpy().reshape(-1)
    (o * seq((2, 3, 4), 13)).sum().backward()
    out["attention_dq"], out["attention_dk"], out["attention_dv"] = g(q), g(k), g(v)

    from tinygrad.helpers import Context
    from tinygrad.nn.optim import Adam, AdamW
    for name, make in (("adam_w", lambda p: Adam(p, lr=0.1)), ("adamw_w", lambda p: AdamW(p, lr=0.1, weight_decay=0.1))):
        w = seq((5,), 14, True)
        t = seq((5,), 15)
        opt = make([w])
        with Context(TRAINING=1):
            for _ in range(5):
                opt.zero_grad()
                (w - t).square().sum().backward()
                opt.step()
        out[name] = w.numpy().reshape(-1)
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
    tiny_r = tiny_results() | milestone1()
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
