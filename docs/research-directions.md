# Research directions (later)

Notes from discussion, 2026-10. Not scheduled; odin-ml needs proper fusion and fast
kernels first (small-model loops are exactly the overhead-bound regime).

## Premise

"In the limit, memory = capacity, as long as memory is fluid": a small fixed core plus
a large writable memory. Split capacity in two:

- **Knowledge** (facts, associations): mostly storage, so memory can stand in for
  parameters (kNN-LM, RETRO, memory layers, product-key memory).
- **Computation** (multi-step composition): bounded by the core's per-step
  compute unless the core can iterate over memory (recurrence, test-time compute).

Hypothesis: a small core plus fluid memory wins on memory-bound tasks and hits a wall
on compute-bound ones unless it can loop. Finding that wall is the experiment.

## Scenes, not facts; planning in latent space

- Don't commit facts. Store whole scenes (latent sequences) under an index. A partial
  cue (the trigger) completes the pattern and replays the scene; facts are read off
  replays. Background: hippocampal indexing theory.
- Reuse elements across pathways while keeping sequences unique:
  - clone-structured cognitive graphs: the same observation gets context-dependent
    clones;
  - the Tolman–Eichenbaum machine: structure factorized from content, so
    structure transfers.
- Plan in a compact latent world model (JEPA-style; MuZero, Dreamer and TD-MPC
  already plan in latents), not in token or observation space. Replanning there is
  cheap.
- Probe, then replan: when the belief is uncertain, take the action with the highest
  expected information gain (belief-space planning, active inference), then replan.
- The DT is already a small belief-state model. Its state is a posterior (not tokens),
  its weights are fixed, and a learned update rewrites the posterior per observation.
  M4's late-sequence drift is its distribution-shift / forgetting failure.

Candidate architecture:
- an encoder into a compact latent;
- a learned update (DT) and learned dynamics;
- an episodic store of latent sequences keyed by context indices;
- retrieval by partial cue, then replay;
- a latent planner that reuses replays and inserts probes when uncertainty is high;
- slow consolidation of scenes into the world model, so generalization doesn't rely
  on lookup alone.

## Continual-learning variables

- **What gets updated:** slow weights (skills) vs fast state (memory, facts).
- **What stays fixed:** the key space. If the encoder drifts, stored keys go stale
  (key drift).
- **Write, merge, evict policy:** merging mixture components is an explicit,
  principled form of forgetting.
- **Consolidation:** distill memory into slow weights periodically ("sleep").

## Experiment ladder

1. **Active sensing in M4:**
   - Give the tracker a steerable sensor and a goal.
   - Plan in belief space with the DT as the model: probe where expected
     information gain is highest, then replan.
2. **Scene memory:**
   - Store latent belief trajectories of past tracks, indexed by their opening
     segment.
   - A matching opening triggers replay, used as prior and rollout.
   - Compare against simulating forward from scratch.
3. **Learn the latent:** a JEPA-style embedding, with anti-collapse regularization,
   in place of the mixture state. Keep the GMM model as the interpretable baseline.
4. **Memory vs core size:**
   - Sweep core params (10k → 1M) × memory slots (0 → 1M) at equal compute.
   - Run it on memory-bound tasks (associative recall) and compute-bound ones
     (multi-hop, composition).
   - Baselines: fine-tune, EWC, replay buffer, frozen core + key–value memory,
     test-time fast weights (TTT / Titans), DT-style learned Bayesian update.
   - Metrics: forgetting, forward and backward transfer, memory size, compute per
     update.
5. **Key drift:** keep training the core while memory persists. Compare a frozen
   encoder, re-encoding, and a learned stable key space.
6. **Consolidation:** distill memory into the core every N tasks; check that
   forgetting stays bounded with a small memory.
7. **Partially observed navigation across environments with shared structure:**
   scene reuse and structure transfer, tested against simple baselines.

## What odin-ml needs for this

- gather / scatter (indexing) and top-k: memory reads and writes
- persistent state outside autograd, updated in place
- inner-loop gradients (a backward inside the forward) for test-time-training layers
- fast small-model loops: fusion, kernel search, record and replay
