# odin-ml

A small ML library in [Odin](https://odin-lang.org), inspired by
[tinygrad](https://github.com/tinygrad/tinygrad) (which also serves as the test oracle).

- **One graph:** a `Tensor` is a UOp node. Ops are lazy; `realize()` schedules and runs them.
- **Fusion:** chains of elementwise ops become one SIMD loop; transposes fold into GEMM.
- **Autograd on the IR:** grad rules emit UOps, so forward and backward are fused together.
- **Checkpoints:** `ml.save` / `ml.load` write safetensors (loads in numpy, torch, MLX, tinygrad).
- **Devices:** CPU (multi-core, Accelerate GEMM on macOS) and Metal (Apple GPU, unified
  memory, a whole step in one command buffer). One switch, same code: `ML_DEVICE=metal`.

Goal: grow it by reimplementing [Distribution Transformers](https://arxiv.org/abs/2502.02463)
(~0.4M params) — that project lives in [dt/](dt/README.md). Plan and status: [STATUS.md](STATUS.md).

## Run

```sh
make test     # op + grad checks          (ML_DEVICE=metal make test: same on the GPU)
make test-metal  # Metal vs CPU parity, every kernel path
make oracle   # vs tinygrad and MLX
make data     # download MNIST into data/mnist
make mnist    # MLP, ~98% in a few seconds
make cnn      # small conv net
make tour     # the API at every level: examples/tour/
make dt-conjugate  # paper project, M2: DeepSets MLP posterior vs exact
make dt-table1     # paper project, M3: the Distribution Transformer (Table 1)
make dt-fusion     # paper project, M4: sensor fusion, the DT as a filter (Table 3)
make models        # the model library: trained weights in models/, with their metadata
```

The dt experiments load their weights from `models/` (not in git) when present and train
(then save) otherwise; add `-- train` to retrain (e.g. `odin run dt/fusion -o:speed -- train`).

Builds are optimized (`-o:speed`); plain `odin run` without it is ~10x slower.
Any program picks its device from `ML_DEVICE=cpu|metal` (default cpu).
Benchmarks and the tinygrad comparison live in [bench/](bench/README.md).

MIT licensed.
