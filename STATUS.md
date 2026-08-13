Objective
CPU-first ML library in Odin (tinygrad-inspired): Tensor + lazy graph + autograd.
Breadth next (nn API, examples CNN→RNN→Transformer); deep opts (fusion/UOps) when limits bite.

Stack (have)
- Tensor: shape/strides, requires_grad, grad, ctx=LazyOp, device
- Lazy ops: +−×÷, MatMul, Sum/mean, Reshape/T, ReLU/Sigmoid, CE, Conv2d, MaxPool2d, flatten
- realize + item; backward skips nil-grad nodes; reverse-mode autograd
- Kernels: SIMD ewise/bcast; matmul Accelerate; Conv = im2col+GEMM; NCHW bias fast path
- nn: Linear, Conv2d, collect_params / linear_params / conv_params; SGD+mom
- data: MNIST IDX, minibatch, eval_accuracy
- debug: ML_DEBUG, Counters, print_graph
- Tests: tensor_ops 51; tinygrad compare 11; regression; MNIST MLP~98%; MNIST CNN~98%

Missing (breadth)
- Embedding, LayerNorm/BN, Dropout, RNN/GRU, Adam
- softmax, gather, cat/pad, free views, GELU
- Metal beyond add; fusion/UOps

Examples ladder
1. regression ✓  2. MNIST MLP ✓  3. MNIST CNN ✓
4. char RNN / seq  5. digit transformer  6. tiny GPT later

Hygiene
- Div bwd broadcast-safe; accum_grad same-shape fast path; topo_sort map
- classify_binary; CE labels on ctx; minibatch remainder; Metal stub
- **Views**: reshape/transpose share storage; contig_data/ensure_contig densify for kernels
- **ml.seed**; **Trainer** (trainer_epoch_ce) — context switch must be same-proc (Odin by-value)
- Tests: 68 unit + 11 tinygrad; MNIST MLP/CNN ~98%

Next Move
Char RNN or digit transformer breadth. Optimize only if an example is too slow.

Relevant Files
ml/{tensor,ops,realize,autograd,nn,optim,data,debug,device,kernel_*}.odin
regression.odin, mnist/, tensor_ops/, bench/, docs/tiny-inspo.md
