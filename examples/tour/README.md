# tour — the API at every level

Runnable files. Each is a self-contained `package main`:

```
odin run examples/tour/01_tensor.odin   -file
odin run examples/tour/02_uop.odin      -file
odin run examples/tour/03_schedule.odin -file
odin run examples/tour/04_autograd.odin -file
odin run examples/tour/05_nn.odin       -file
```

Read them in order. They are the API as it exists now, not a wish list.

```
you write this                  library does this
─────────────────               ─────────────────
01  Tensor ops (lazy)    →      build UOp nodes (a Tensor IS a UOp)
02  the graph            →      op / src / arg / shape; compositions visible
03  realize              →      fuse ewise, fold transposes into GEMM, run
04  backward             →      grad rules emit UOps; fwd+bwd in one schedule
05  nn + SGD             →      Linear / Conv2d / one training step
```

What exists today

- Tensor ops: `add sub mul div maximum cmplt neg exp relu sigmoid sum sum_axes mean reshape permute transpose T expand flatten matmul conv2d max_pool2d cross_entropy`
- UOps: `Input Const | Add Sub Mul Div Max CmpLt Neg Exp Expand | Sum | Reshape Permute | MatMul Conv2d MaxPool2d CrossEntropy` (+ their backward primitives)
- Compositions (not primitives): `relu = max(x,0)`, `sigmoid = 1/(1+exp(-x))`, `mean = sum * 1/n`
- MatMul is a primitive (Accelerate GEMM). Conv is im2col + GEMM.
- Fusion: same-shape ewise ops with one consumer become one kernel.

Full training examples live elsewhere: `examples/regression`, `examples/mnist`, `examples/mnist_cnn`.
