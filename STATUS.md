Objective
Build a small CPU-first ML library in Odin (tinygrad-style Tensor + autograd), then harden tensors, SIMD, Metal backends, benches vs NumPy, and move toward lazy graph execution.
Important Details
Design: Tensor holds data/shape/strides/requires_grad/grad/ctx/device; Context = LazyOp (Op + parents + meta); reverse-mode autograd via topo sort.
Memory: prefer arenas/context.allocator; params persistent; after arena free_all, clear_grads nils dangling grads.
macOS M5: NumPy matmul ≈ Accelerate; library matmul defaults to Accelerate; pure tiled SIMD educational (~few % of BLAS).
Device: .CPU / .Metal; Metal via vendor:darwin/Metal (_darwin file suffix); not MLX C; cross-platform goal (Linux later).
Lazy path: ops should only build graph; realize(sink) topo-sorts sources→sink then runs kernels; backward should realize first.
User wants to learn; UOps later; next priority is lazy + realize, not beam search yet.
Broadcast SIMD worth it for scalar/row/col patterns; general strided stays scalar; eager per-op alloc made broadcast benches far below NumPy.
Work State
Completed
Package ml/: tensor, ops, autograd, optim (SGD), f32 SIMD kernels, matmul (pure + Accelerate), Metal add path, device enum.
Demos/tests: regression.odin, tensor_ops/, metal_probe/, metal_test/ (GPU add worked).
Benches: bench/ with uv+NumPy, matmul/elem compare (~100% NumPy on Accelerate matmul/large elem), broadcast compare (Odin low % due to alloc).
Broadcast SIMD: scalar/row/col paths; 41 tensor_ops tests passed at last full run.
Started lazy: ml/realize.odin (new_tensor_lazy, realize, item, forward_op); ops header/docs lean lazy; regression edited toward lazy + item(loss).
Active
Inspectability in: debug_level / ML_DEBUG, Counters, print_graph, per-op fwd/bwd timing in realize/backward.
Blocked
(none)
Next Move
MNIST green with traces. Then ops MNIST needs; fusion later.
Relevant Files
/Users/joe/code/sketches/odin-ml/ml/tensor.odin — Tensor core, device, allclose/realize hooks
/Users/joe/code/sketches/odin-ml/ml/ops.odin — graph builders (should be lazy)
/Users/joe/code/sketches/odin-ml/ml/realize.odin — realize + forward_op
/Users/joe/code/sketches/odin-ml/ml/autograd.odin — backward / topo_sort
/Users/joe/code/sketches/odin-ml/ml/kernel_f32.odin — contiguous + broadcast SIMD
/Users/joe/code/sketches/odin-ml/ml/kernel_matmul.odin / kernel_accelerate.odin — GEMM backends
/Users/joe/code/sketches/odin-ml/ml/backend_metal_darwin.odin — Metal pipeline
/Users/joe/code/sketches/odin-ml/ml/device.odin — Device enum
/Users/joe/code/sketches/odin-ml/regression.odin — training loop
/Users/joe/code/sketches/odin-ml/tensor_ops/ — correctness + broadcast microbench
/Users/joe/code/sketches/odin-ml/bench/ — NumPy/Odin compare scripts
/Users/joe/code/sketches/odin-ml/notes.txt — architecture notes
