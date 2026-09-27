#!/usr/bin/env python3
"""Compare odin-ml vs MLX (Apple's array framework, on the same Metal GPU).

Same RESULT names as tinygrad_compare.py. Where MLX has its own kernel the
check uses it, not our composition rebuilt in MLX: fast.layer_norm,
losses.cross_entropy, fast.scaled_dot_product_attention, optimizers.Adam(W).

Runs MLX on its CPU by default: on the M5, MLX's GPU float32 matmul is reduced
precision (~1e-3 relative vs float64; its CPU path ~1e-7), which would fail
our fp32 tolerance on every matmul. --gpu runs it on the GPU with rtol 3e-3.

  cd bench
  odin run odin_vs_tiny -o:speed > odin_ref.txt       # or ML_DEVICE=metal
  uv run mlx_compare.py odin_ref.txt [--gpu]
"""
from __future__ import annotations

import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from oracle import parse_odin, report, seq_np


def seq(shape, k):
    return mx.array(seq_np(shape, k))


def a(x) -> np.ndarray:
    return np.array(x).reshape(-1)


def mlx_results() -> dict[str, np.ndarray]:
    out = {}
    A = mx.array([[1.0, 2.0], [3.0, 4.0]])
    B = mx.array([[10.0, 20.0], [30.0, 40.0]])
    out["add_same"], out["mul_same"] = a(A + B), a(A * B)
    M = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out["mul_row"] = a(M * mx.array([10.0, 100.0, 1000.0]))
    out["add_col"] = a(M + mx.array([[1.0], [2.0]]))
    out["matmul_2x2"] = a(mx.array([[1.0, 2.0], [3.0, 4.0]]) @ mx.array([[5.0, 6.0], [7.0, 8.0]]))
    X = mx.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]])
    out["relu_xw_b"] = a(nn.relu(X @ mx.array([[1.0], [-1.0]]) + mx.array([[0.5]])))
    x, y = mx.array([0.0, 1.0, 2.0, 3.0]), mx.array([1.0, 3.5, 6.0, 8.5])
    out["mse_mean"] = a(mx.mean(mx.square(x * 2.5 + 1.0 - y)))
    xw = mx.array([1.0, 2.0, 3.0, 4.0])
    out["grad_w"] = a(mx.grad(lambda w: mx.mean(mx.square(xw * w)))(mx.array([0.5])))

    # MLX convs / pools are NHWC with weights [O, kH, kW, I]
    img = mx.array(np.arange(1, 10, dtype=np.float32).reshape(1, 3, 3, 1))
    out["conv2d_k2"] = a(mx.conv2d(img, mx.ones((1, 2, 2, 1))))
    px = mx.array(np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4, 1))
    out["maxpool2d"] = a(nn.MaxPool2d(2, 2)(px))
    gx = mx.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 2, 2, 1))
    dw = mx.grad(lambda w: mx.sum(mx.conv2d(gx, w)))(mx.full((1, 2, 2, 1), 0.5))
    out["conv_dW"] = a(dw)

    # ---- milestone 1 ----
    x0 = seq((4,), 0) * 0.5 + 1
    out["log_sqrt_dx"] = a(mx.grad(lambda t: mx.sum(mx.log(t) * mx.sqrt(t)))(x0))

    mx_in = mx.array([[1.0, 5.0, 5.0], [2.0, 0.0, -1.0]])
    out["max_axis"] = a(mx.max(mx_in, axis=1, keepdims=True))
    out["max_axis_dx"] = a(mx.grad(lambda t: mx.sum(mx.max(t, axis=1, keepdims=True) * mx.array([[1.0], [2.0]])))(mx_in))

    s = seq((3, 4), 1)
    out["softmax"] = a(mx.softmax(s, axis=1))
    out["log_softmax"] = a(s - mx.logsumexp(s, axis=1, keepdims=True))
    out["logsumexp"] = a(mx.logsumexp(s, axis=1, keepdims=True))
    out["softmax_dx"] = a(mx.grad(lambda t: mx.sum(mx.softmax(t, axis=1) * seq((3, 4), 2)))(s))

    ln = seq((2, 5), 3)
    out["layernorm"] = a(mx.fast.layer_norm(ln, None, None, 1e-5))
    out["layernorm_dx"] = a(mx.grad(lambda t: mx.sum(mx.fast.layer_norm(t, None, None, 1e-5) * seq((2, 5), 4)))(ln))

    lg = seq((3, 4), 5)
    ce = lambda t: nn.losses.cross_entropy(t, mx.array([2, 0, 3]), reduction="mean")
    out["cross_entropy"] = a(ce(lg))
    out["cross_entropy_dx"] = a(mx.grad(ce)(lg))

    ba, bb, bw, bc = seq((2, 2, 3), 6), seq((2, 3, 2), 7), seq((3, 2), 8), seq((1, 3, 2), 9)
    out["bmm"] = a(ba @ bb)
    bmm_loss = lambda p, q, w, c: mx.sum(p @ q) + mx.sum((p @ w) * 2) + mx.sum((p @ c) * 3)
    for name, g in zip(("bmm_da", "bmm_db", "bmm_dw", "bmm_dc"), mx.grad(bmm_loss, argnums=(0, 1, 2, 3))(ba, bb, bw, bc)):
        out[name] = a(g)

    # one head: [B, T, D] → [B, 1, T, D]; scale 0.5 as on the odin side
    def attention(q, k, v):
        r = lambda t: t.reshape(2, 1, 3, 4)
        return mx.fast.scaled_dot_product_attention(r(q), r(k), r(v), scale=0.5).reshape(2, 3, 4)
    q, k, v = seq((2, 3, 4), 10), seq((2, 3, 4), 11), seq((2, 3, 4), 12)
    out["attention"] = a(attention(q, k, v))
    dq, dk, dv = mx.grad(lambda q, k, v: mx.sum(attention(q, k, v) * seq((2, 3, 4), 13)), argnums=(0, 1, 2))(q, k, v)
    out["attention_dq"], out["attention_dk"], out["attention_dv"] = a(dq), a(dk), a(dv)

    # MLX Adam defaults to bias_correction=False; ours (and tinygrad's) corrects
    for name, opt in (("adam_w", optim.Adam(learning_rate=0.1, bias_correction=True)),
                      ("adamw_w", optim.AdamW(learning_rate=0.1, weight_decay=0.1, bias_correction=True))):
        params = {"w": seq((5,), 14)}
        t = seq((5,), 15)
        grad_fn = mx.grad(lambda p: mx.sum(mx.square(p["w"] - t)))
        for _ in range(5):
            opt.update(params, grad_fn(params))
        out[name] = a(params["w"])
    w7 = seq((12,), 7)
    for name, scale, f in [("gelu", 4, nn.gelu_approx), ("tanh", 3, mx.tanh),
                           ("sigmoid_tails", 120, mx.sigmoid), ("clip", 2, lambda t: mx.clip(t, -1, 0.5))]:
        xs = seq((12,), 6) * scale
        out[name] = a(f(xs))
        out[name + "_dx"] = a(mx.grad(lambda t: mx.sum(f(t) * w7))(xs))
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    odin_r, _ = parse_odin(sys.argv[1])
    gpu = "--gpu" in sys.argv[2:]
    mx.set_default_device(mx.gpu if gpu else mx.cpu)
    print(f"mlx {mx.__version__} on {mx.default_device()}")
    rtol = 3e-3 if gpu else 1e-4
    return 1 if report("mlx", odin_r, mlx_results(), rtol=rtol, atol=1e-4 if gpu else 1e-5) else 0


if __name__ == "__main__":
    raise SystemExit(main())
