# Odin Allocators & Efficient Memory — Notes for odin-ml

Research from Odin core (`2026-07a`), [overview docs](https://odin-lang.org/docs/overview/),
and current odin-ml usage.

---

## Core model

Odin has **no GC**. Memory is explicit via `Allocator` on `context`.

```odin
Allocator :: struct {
    procedure: Allocator_Proc,
    data:      rawptr,
}

// context carries two:
context.allocator       // general / subsystem default
context.temp_allocator  // short-lived scratch
```

Defaults: heap for `allocator`, growing arena (~4 MiB) for `temp_allocator`.

`context` is **implicit and inherited**. Switch it for a scope:

```odin
old := context.allocator
context.allocator = mem.dynamic_arena_allocator(&arena)
// all new/make in this call tree use the arena
context.allocator = old
```

**Important:** `[dynamic]` arrays and maps **store their allocator at creation**.
Later `append` does not re-read `context.allocator`.

| API | Meaning |
|-----|---------|
| `new` / `free` | Single object |
| `make` / `delete` | Groups (slice, dynamic, map, …) |
| `alloc` / `resize` | Raw pointers |
| `*_non_zeroed` | Skip zero-fill (faster for large buffers) |
| `free_all` | Bulk reclaim (arena-style) |

`DEFAULT_ALIGNMENT` = 16 on 64-bit.

---

## Built-in allocators — when to use

| Allocator | Free model | Use for |
|-----------|------------|---------|
| **heap** (`heap_allocator`) | Individual free | Long-lived: weights, dataset, opt state |
| **`mem.Arena`** | Free_All / nested temp | Fixed buffer, known max size |
| **`mem.Dynamic_Arena`** | reset / free_all (no individual free) | Per-step graphs (odin-ml pattern) |
| **`core:mem/virtual`.Arena** | free_all; Growing/Static/Buffer | Large models; prefer for growing arenas |
| **`context.temp_allocator`** | `free_all` once per frame/step | Strings, tiny temps, fmt |
| **`mem.Scratch`** | Ring; wrap invalidates old ptrs | Short scratch only — careful |
| **`mem.Tracking_Allocator`** | Wraps another | Debug leaks only |
| **tlsf / Buddy / Stack** | Individual free variants | Specialized heaps |
| **Mutex_Allocator** | Wraps another | Share allocator across threads |

### Dynamic_Arena details (hot path)

```odin
mem.dynamic_arena_init(&arena,
    block_size    = 4 * mem.Megabyte,   // default 64 KiB — raise for ML
    out_band_size = 2 * mem.Megabyte,   // default ~6.5 KiB
)
// loop:
mem.dynamic_arena_reset(&arena)   // recycle blocks — prefer in training loop
// teardown:
mem.dynamic_arena_destroy(&arena)
```

- Small allocs bump inside blocks; large (≥ `out_band_size`) go out-of-band to heap.
- **`reset`** keeps warm blocks. **`free_all`** returns blocks to OS/heap — wasteful every step.
- No individual `free`.

### virtual.Arena (recommended for big workloads)

```odin
import vmem "core:mem/virtual"
vmem.arena_init_growing(&arena, reserved = ...)
context.allocator = vmem.arena_allocator(&arena)
```

Runtime docs: prefer `core:mem/virtual` for your own growing arena needs.

---

## Best practices for ML / numerics

### Golden rules

1. **Never malloc per op** on the hot path.
2. **Split lifetimes:**
   - Persistent → heap / long arena (weights, opt, data)
   - Ephemeral → step arena (activations, graph nodes, grads)
3. Prefer **`reset` / `free_all` once** over tracking individual frees.
4. Use **non-zeroed** allocs for large tensors you immediately fill.
5. Align numeric buffers for SIMD (16/32/64).
6. **Reuse capacity** — warm arena blocks via `reset`, not free+realloc every step.
7. Keep tensors contiguous; avoid tiny heap metadata allocs if you can use fixed `[MAX_DIMS]`.

### Arena-per-training-step (canonical)

```odin
// persistent (heap)
params := init_model()
opt    := init_optimizer(params)
data   := load_dataset()

arena: mem.Dynamic_Arena
mem.dynamic_arena_init(&arena,
    block_size = 4 * mem.Megabyte,
    out_band_size = 2 * mem.Megabyte,
)
defer mem.dynamic_arena_destroy(&arena)

for batch in data {
    old := context.allocator
    context.allocator = mem.dynamic_arena_allocator(&arena)

    loss := step_forward_backward(batch, params)
    loss_val := loss.data[0]
    opt_step(opt, params)

    context.allocator = old
    mem.dynamic_arena_reset(&arena)  // NOT free_all in the loop
    clear_param_grads(params)        // nil dangling .grad into freed arena
}
free_all(context.temp_allocator)
```

### Other patterns

| Pattern | How |
|---------|-----|
| Tiny temps | `context.temp_allocator` + `free_all` per epoch/frame |
| Nested savepoint | `begin_arena_temp_memory` / `vmem.arena_temp_begin` |
| Static shapes / inference | Preallocated buffer pools, reuse every step |
| Debug leaks | `Tracking_Allocator` on **persistent** allocator only |
| Workspace (matmul) | Step-arena or thread-local scratch, not heap per call |
| GPU | Host arena for graph meta; device buffer pool with same cadence |

---

## Common pitfalls

1. `free` on arena memory → `Mode_Not_Implemented`. Use reset/free_all.
2. Holding pointers after reset → UAF. Always `clear_grads` after reclaim.
3. Creating `[dynamic]` under arena, using after reset — dead.
4. `Scratch` wrap-around silently invalidates older pointers.
5. Zeroing multi-MB tensors — real cost; use non-zeroed + fill.
6. Wrong allocator on `delete` — dynamic arrays use allocator stored at `make`.
7. **`dynamic_arena_free_all` every step** — reallocates blocks; use **`reset`**.
8. Default `out_band_size` (~6.5 KiB) sends big tensors out-of-band — raise it or `block_size`.
9. Loading data while step arena is `context.allocator` — pollutes/frees data.
10. Tracking every step-arena alloc — noise + cost; track persistent only.

---

## Implications for odin-ml

### Already good

- Persistent params vs per-step `Dynamic_Arena` graph
- `context.allocator` switch around forward/backward
- `clear_grads` after reclaim
- Eval uses its own arena

### Concrete upgrades

1. Switch training loops from `dynamic_arena_free_all` → **`dynamic_arena_reset`**.
2. Size arena for batch: `block_size = 4..8 MiB`, raise `out_band_size`.
3. Consider `virtual.Arena` growing for large models.
4. Fixed `[MAX_DIMS]i32` shape/strides to cut per-tensor tiny allocs.
5. Optional: persistent grad buffers + `zero_grad` when shapes are stable.
6. GPU path: separate device buffer pool with free_all cadence matching host step.
7. Debug: `Tracking_Allocator` on heap/persistent only.

---

## API map (Odin install)

| Concern | Path under `odin root` |
|---------|------------------------|
| Allocator + new/make/free | `core/mem/alloc.odin` |
| Arena, Scratch, Dynamic_Arena, Buddy | `core/mem/allocators.odin` |
| Tracking | `core/mem/tracking_allocator.odin` |
| Virtual arenas | `core/mem/virtual/arena.odin` |
| Heap | `base/runtime/heap_allocator.odin` |
| Default temp | `base/runtime/default_temporary_allocator.odin` |
| Context | `base/runtime/core.odin` |

---

## Bottom line

Treat each training step like a game frame: bump-allocate the whole graph on a
`Dynamic_Arena` (or `virtual.Arena`), read scalars / apply grads, then **`reset`
once**. Keep parameters and datasets on the heap. No per-op malloc. Prefer
non-zeroed large buffers and warm blocks via `reset`.
