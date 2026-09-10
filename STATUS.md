Objective
CPU-first ML library in Odin (tinygrad-inspired): Tensor + UOp IR + autograd.
Local accelerator: CPU now, Metal/CUDA later. Keep the IR small; compose new ops.

Stack (have)
- Tensor: shape/strides, requires_grad, grad, ctx=LazyOp, device
- Realize: Tensor DAG → UOp → fuse same-shape ewise → kernels
- ReLU = max(x,0), sigmoid = 1/(1+exp(-x)); MatMul/Conv/CE stay primitives
- Reverse-mode autograd on Tensor (not UOp)
- Kernels: SIMD ewise/bcast + fused ewise loop; matmul Accelerate; Conv = im2col+GEMM
- nn: Linear, Conv2d, collect_params; SGD+mom; Trainer (arena reset per step)
- data: MNIST IDX, minibatch, eval_accuracy
- debug: ML_DEBUG, Counters (kernels/fused_ops/alloc), print_graph
- Tests: tensor_ops 88; tinygrad compare 11; regression; MNIST MLP/CNN ~98%
- Bench: `odin run bench/loop -o:speed -no-bounds-check -disable-assert`

Missing
- Embedding, LayerNorm/BN, Dropout, RNN/GRU, Adam
- softmax, gather, cat/pad, GELU (as UOp compositions)
- Fused backward; Metal as fused-kernel renderer (add prototype only)

Examples ladder
1. regression ✓  2. MNIST MLP ✓  3. MNIST CNN ✓
4. char RNN / seq  5. digit transformer  6. tiny GPT later

Next Move
Fuse backward ewise, or Metal renderer of fused groups — not more forward_op cases.
Breadth (RNN/transformer) once those ops are compositions on the existing IR.

Relevant Files
ml/{tensor,ops,uop,fuse,realize,autograd,nn,optim,data,debug,device,kernel_*}.odin
regression.odin, mnist/, tensor_ops/, bench/loop/, docs/tiny-inspo.md
