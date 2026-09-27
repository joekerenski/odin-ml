# odin-ml

A small ML library in [Odin](https://odin-lang.org), inspired by
[tinygrad](https://github.com/tinygrad/tinygrad) (which also serves as the test oracle).

- **One graph:** a `Tensor` is a UOp node. Ops are lazy; `realize()` schedules and runs them.
- **Fusion:** chains of elementwise ops become one SIMD loop; transposes fold into GEMM.
- **Autograd on the IR:** grad rules emit UOps, so forward and backward are fused together.
- **CPU first:** Accelerate GEMM on macOS, portable SIMD elsewhere. Metal later.

Goal: grow it by reimplementing [Distribution Transformers](https://arxiv.org/abs/2502.02463)
(~0.4M params) — that project lives in [dt/](dt/README.md). Plan and status: [STATUS.md](STATUS.md).

## Run

```sh
make test     # op + grad checks
make data     # download MNIST into data/mnist
make mnist    # MLP, ~98% in a few seconds
make cnn      # small conv net
make tour     # the API at every level: examples/tour/
make dt-conjugate  # paper project, M2: amortized posterior vs exact (~2 min)
```

Builds are optimized (`-o:speed`); plain `odin run` without it is ~10x slower.
Benchmarks and the tinygrad comparison live in [bench/](bench/README.md).

MIT licensed.
