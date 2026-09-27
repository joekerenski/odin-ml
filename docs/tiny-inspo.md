# tinygrad Abstraction Layers — Inspiration for odin-ml

Reference notes on how tinygrad accelerates ML, for designing an Odin counterpart.
Source: https://github.com/tinygrad/tinygrad

---

## Mental model

**One IR all the way down.** A single UOp DAG from Tensor programs to command buffers.

> *"tinygrad: a single dialect from Tensor programs to Command Buffers"*

Four framework pieces:

| Piece | Role |
|-------|------|
| **Frontend** | PyTorch-like `Tensor` API; sugar over UOps |
| **Scheduler** | Big UOp graph → kernels (`CALL`s in a `LINEAR`) |
| **Lowering** | AST → optimized UOps → source → binary (`PROGRAM`) |
| **Execution** | Device runtimes launch kernels / copies / graphs |

Directory order = processing order:

```
schedule/  →  codegen/opt/  →  codegen/  →  renderer/  →  engine/  →  runtime/
```

---

## Stack (top → bottom)

```
0. nn / optim / datasets          user libraries
1. Tensor API                     lazy frontend
2. UOp IR                         single dialect
3. Callify                        graph → function + buffers
4. Schedule                       kernel split + mem plan
5. Codegen                        OptOps / BEAM / lower
6. Renderer + Compiler            source / ISA → binary
7. Engine                         realize / JIT / graphs
8. Device + Runtime               alloc, launch, HCQ
9. Hardware                       GPU / CPU / Metal / …
```

---

## Layer details

### 0 — User libraries (`tinygrad/nn/`)

Linear, Conv, Adam/SGD, state save/load, datasets, LLM helpers.
No `nn.Module` — plain classes + `get_parameters`. Functional style.
Builds lazy graphs; acceleration lives below.

### 1 — Tensor frontend (`tensor.py`, `mixin/*`, `function.py`)

- `Tensor` holds `uop: UOp` (+ `grad`, `is_param`).
- **Fully lazy**: `a+b` only builds graph. Run on `.realize()`, `.numpy()`, `.item()`, or JIT.
- Mixins shared with UOp; ops mostly `_apply_uop`.
- `@function` traces Python fn → reusable `Ops.FUNCTION` fragment.
- `TinyJit` captures schedule and replays without Python rebuild.
- Multi-device via `Tensor.shard`.

**Acceleration:** laziness = fusion opportunity; AD on IR; JIT; shard.

### 2 — UOp IR (`uop/`)

UOp = `(op, src, arg, tag)` with derived dtype/shape/device/addrspace.

Categories: Source, Movement, Reduce, Elementwise, Load/Store, Ordering
(`RANGE`/`LINEAR`/`SINK`), Call, Codegen-only (`WMMA`, `PROGRAM`, …).

Whole compiler is **pattern-matching rewrites** (`UPat` / `graph_rewrite`).

**Acceleration:** symbolic simplify, algebraic opts, CSE via UOp cache,
high-level ops as compositions (GEMM = reshape+mul+sum) → natural fusion.

### 3 — Callify (`callify.py`)

Lazy tensor graph → stateless function with explicit buffers.
Insert `BUFFER`/`STORE` only where needed; elide free movement to views/slices.

### 4 — Schedule (`schedule/`)

```
SINK → rangeify → split kernels → CALL list → toposort → LINEAR
     → memory_plan (TLSF lifetime packing)
```

One `CALL` = one GPU kernel.

**Acceleration:**
- Kernel fusion (elementwise + reduce stay together until forced split)
- Reduce splitting for large reductions
- Multi-device collectives as UOp compositions
- **Memory planning** — non-overlapping buffer reuse (very high ROI)
- Schedule cache

Dominates training “model speed.”

### 5 — Codegen (`codegen/`)

Per-kernel: apply OptOps (BEAM or heuristics) → expand → decomp → linearize.

| OptOp | Effect |
|-------|--------|
| `TC` | Tensor-core WMMA |
| `UPCAST` | Register vectorization |
| `UNROLL` | Unroll reduce loops |
| `LOCAL` / `GROUP` | Shared memory / workgroup |
| `THREAD` | CPU thread parallel |

**Acceleration:** BEAM search, tensor cores, local mem, coalescing,
index simplify, transcendental approx. Dominates kernel speed.

### 6 — Renderer + Compiler (`renderer/`, `runtime/support/compiler_*`)

~25 low-level ops → device needs Renderer + Compiler + launch.
C-style (Metal/CUDA/OpenCL/HIP), PTX, LLVM, WGSL, ISA paths.

### 7 — Engine (`engine/realize.py`, `engine/jit.py`)

`run_linear`: compile each CALL → exec kernel/copy/graph/HCQ.
TinyJit: capture → PARAM-ize → memory plan → replay (optionally as graphs).

**Acceleration:** kernel graph batching, HCQ prebuilt queues, program cache.

### 8 — Device + Runtime (`device.py`, `runtime/ops_*`)

Buffer / MultiBuffer / Allocator / LRUAllocator / Program.
HCQ userspace queues (NV/AMD) bypass CUDA/HIP runtime overhead.
LRU buffer cache, SDMA, multi-device.

---

## End-to-end flow

```
User:  loss = model(x).crossentropy(y).backward()
       Tensor.realize(...)   # or TinyJit

① Tensor methods → UOp DAG
② callify: CONTIGUOUS → BUFFER+STORE; views → SLICE
③ schedule: rangeify → split → LINEAR + mem plan
④ each CALL: codegen (OptOps/BEAM) → PROGRAM (source+binary)
⑤ run_linear: Device runtime launch / COPY / graph / HCQ
⑥ Tensor.uop becomes realized BUFFER
```

---

## Acceleration techniques by layer

| Layer | Techniques |
|-------|------------|
| Tensor | Laziness, fusion-friendly graph, AD on IR, JIT, shard |
| UOp | Single dialect, graph_rewrite, symbolic, CSE, composition-as-fusion |
| Callify | Buffer only where needed; view elision |
| Schedule | Kernel fusion/split, reduce split, TLSF mem plan, collectives |
| Codegen | BEAM, TC/WMMA, local mem, upcast/unroll, coalesce |
| Renderer | Device-native emit; ISA path |
| Engine | TinyJit replay, command graphs, program cache |
| Runtime | HCQ userspace, LRU alloc, SDMA |

---

## Compact spaces diagram

```
API ── Tensor space (shapes, nn, autograd)
         │
         ▼
       UOp tensor graph (lazy DAG)
         │ callify
         ▼
       UOp function + buffers
         │ rangeify + split
         ▼
       Kernel ASTs (per-CALL, RANGE axes)
         │ OptOps / BEAM
         ▼
       Loop-scheduled AST (GLOBAL/LOCAL/UPCAST/…)
         │ expand + decomp + linearize
         ▼
       Linear UOps / source / binary
         │ realize / JIT / HCQ
         ▼
       Device queues + memory
```

---

## Implications for odin-ml

1. **One IR** beats many IRs for hackability and end-to-end fusion.
2. **Laziness is the fusion mechanism** — not a separate pass over eager ops.
3. **Movement ops are free** until materialization (views / reshape / permute).
4. **Schedule ≠ codegen schedule**: first splits *kernels*; second transforms *loop nests*.
5. **Pattern-matching rewrites** are the compiler API.
6. **~25 ALU ops** + load/store/range is enough for many backends.
7. **Memory planning** (lifetimes → arenas) is separate from fusion and high ROI.
8. **JIT + command graphs** matter as much as kernel quality for step time.
9. **Multi-device as IR** scales cleaner than a runtime bolt-on.
10. **Escape hatches** at every level (custom kernel → foreign source → asm).

### Practical staging for Odin

| Phase | Build |
|-------|-------|
| Now | Eager/lazy tensors + arena-per-step + CPU ops |
| Next | Single IR (UOp-like) + realize + simple fusion |
| Then | Schedule (kernel split + buffer lifetime plan) |
| Later | Metal/codegen + JIT capture/replay |
| Eventually | Opt search, multi-device, HCQ-style graphs |

### Key files in this repo

```
tinygrad/tensor.py, callify.py, device.py
tinygrad/uop/          # IR + rewrite
tinygrad/schedule/     # kernels + memory plan
tinygrad/codegen/      # opt + lower
tinygrad/renderer/     # emit
tinygrad/engine/       # realize + jit
tinygrad/runtime/      # devices
docs/developer/{developer,layout,speed}.md
docs/abstractions{3,4}.py
spec/tinyspec.tex
```
