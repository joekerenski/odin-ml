# M4 in tinygrad

The sensor-fusion Distribution Transformer (`dt/fusion`) ported to tinygrad, parameter for
parameter: same layers, same parameter order and shapes (Linear weights are `[in, out]`),
same batch sampler (numpy), same loss. Checkpoints load in both directions.

It serves two purposes:
- **Whole-model oracle:** our trained weights give the same loss here on the same batch.
- **Speed reference:** tinygrad's fusion and BEAM-searched kernels on the same training
  step, to set targets for odin-ml's scheduler.

```sh
uv run fusion.py check        # loss of models/fusion_dt4 on a fixed batch; then run the Odin side:
odin run dt/fusion -o:speed -- check build/fusion_check_batch.safetensors
uv run fusion.py bench 30     # kernels/step, ms/step; BEAM=2 uv run ... for tuned kernels
uv run fusion.py train        # 60k steps → models/fusion_dt4_tinygrad.safetensors
odin run dt/fusion -o:speed -- eval models/fusion_dt4_tinygrad.safetensors   # Odin's evaluation
```

tinygrad is pinned to a master commit in `pyproject.toml`. To use a local checkout instead:
`TINYGRAD_PATH=~/code/repos/tinygrad uv run --no-project --with numpy python fusion.py ...`
