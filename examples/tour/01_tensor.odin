package main

// High-level Tensor API. Ops are lazy: they only build a graph.
// Nothing runs until realize() / item() / backward().
//
//   odin run examples/tour/01_tensor.odin -file

import "core:fmt"
import ml "../../ml"

show :: proc(name: string, t: ^ml.Tensor) {
	ml.realize(t)
	fmt.printfln("%-16s shape=%v  data=%v", name, t.shape, t.data)
}

main :: proc() {
	fmt.println("=== 01 Tensor ops (lazy frontend) ===\n")

	// Leaves (.Input nodes): already have data.
	a := ml.from_data_copy({1, 2, 3, 4}, {2, 2})
	b := ml.from_data_copy({10, 20, 30, 40}, {2, 2})
	show("a", a)
	show("b", b)

	// Same-shape elementwise
	show("a+b", ml.add(a, b))
	show("a*b", ml.mul(a, b))
	show("a-b", ml.sub(a, b))
	show("neg(a)", ml.neg(a))

	// Broadcast: right-aligned NumPy rules. [2,2] + [2] → row; [2,2]+[2,1] → col
	row := ml.from_data_copy({1, 10}, {2})
	col := ml.from_data_copy({1, 2}, {2, 1})
	show("[2,2]+[2]", ml.add(a, row))
	show("[2,2]+[2,1]", ml.add(a, col))

	// Activations
	x := ml.from_data_copy({-1, 0, 2, -3}, {4})
	show("relu(x)", ml.relu(x))
	show("sigmoid(0)", ml.sigmoid(ml.from_data_copy({0}, {1})))

	// Matmul is 2D only. [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
	A := ml.from_data_copy({1, 2, 3, 4}, {2, 2})
	B := ml.from_data_copy({5, 6, 7, 8}, {2, 2})
	show("A@B", ml.matmul(A, B))

	// Movement: reshape is a view (same buffer); T realizes a dense copy
	// (or, feeding a matmul, folds into the GEMM — see 03).
	t := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {2, 3})
	r := ml.reshape(t, {3, 2})
	ml.realize(r)
	fmt.printfln("%-16s shape=%v  shares_t=%v", "reshape(2,3→3,2)", r.shape, raw_data(r.data) == raw_data(t.data))
	show("T(2x3)", ml.T(t))

	// Reductions keep the reduced dim as 1. No axis = everything → shape [1];
	// negative axes count from the end (-1 = last), as in numpy.
	show("sum all", ml.sum(a))
	show("mean", ml.mean(a))
	show("sum axis=1", ml.sum(a, 1)) // keeps dim as 1: [2,1]
	show("sum axis=-2", ml.sum(a, -2)) // first axis of a 2D tensor: [1,2]

	// log / sqrt / max over an axis (keepdim)
	show("log(a)", ml.log(a))
	show("sqrt(a)", ml.sqrt(a))
	show("max axis=1", ml.max_axis(a, 1))

	// Compositions of the above: softmax family, layer norm
	z := ml.from_data_copy({1, 2, 3, 1, 1, 1}, {2, 3})
	show("softmax axis=1", ml.softmax(z, 1))
	show("logsumexp ax=1", ml.logsumexp(z, 1))
	show("layer_norm", ml.layer_norm(z)) // last axis, mean 0 / var 1

	// Batched matmul: [B,M,K] @ [B,K,N]; mT swaps the last two axes
	q := ml.from_data_copy({1, 0, 0, 1, 1, 1, 0, 2}, {2, 2, 2})
	show("q @ q^T (B=2)", ml.matmul(q, ml.mT(q)))

	// Spatial, NCHW. conv: x[N,Ci,H,W]  w[Co,Ci,kH,kW]
	img := ml.from_data_copy({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3})
	k := ml.from_data_copy({1, 1, 1, 1}, {1, 1, 2, 2})
	show("conv2d k=2", ml.conv2d(img, k, stride = 1, padding = 0))
	show("maxpool 2x2", ml.max_pool2d(img, kernel_size = 2, stride = 1))
	show("flatten", ml.flatten(img)) // [1, 9]

	// Lazy: the graph exists before any kernel runs
	fmt.println("\n-- lazy graph of relu(A@B + 1), before realize --")
	bias := ml.from_data_copy({1}, {1})
	y := ml.relu(ml.add(ml.matmul(A, B), bias))
	fmt.printfln("  y.data is nil? %v  (unrealized)", y.data == nil)
	ml.print_graph(y, "relu(A@B+1)")
	ml.realize(y)
	fmt.printfln("  after realize: %v", y.data)
}
