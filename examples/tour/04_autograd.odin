package main

// Reverse-mode autograd on the UOp graph, emitting UOps.
// backward(loss) applies each op's grad rule (chain rule) to build the grad
// graph, puts it on leaves' .grad, then realizes loss + all grads in ONE
// schedule — the backward pass is fused exactly like the forward.
//
//   odin run examples/tour/04_autograd.odin -file

import "core:fmt"
import ml "../../ml"

main :: proc() {
	fmt.println("=== 04 autograd (UOp graph) ===\n")

	// Scalar-ish: y = relu(x*w + b), L = mean(y)
	x := ml.from_data_copy({1, -1, 2}, {3})
	w := ml.from_data_copy({0.5}, {1}, requires_grad = true)
	b := ml.from_data_copy({0.1}, {1}, requires_grad = true)

	h := ml.relu(ml.add(ml.mul(x, w), b)) // [0.6, 0, 1.1]
	loss := ml.mean(h)

	fmt.println("forward graph:")
	ml.print_graph(loss, "loss")

	ml.counters_reset()
	ml.backward(loss)
	fmt.println("\nw.grad is itself a UOp graph (now realized):")
	ml.print_graph(w.grad, "dL/dw")
	ml.counters_print("fwd+bwd, one schedule")

	fmt.printfln("\n  h    = %v", h.data)
	fmt.printfln("  loss = %.4f", ml.item(loss))
	fmt.printfln("  dL/dw = %v", w.grad.data)
	fmt.printfln("  dL/db = %v", b.grad.data)
	fmt.println("  x has no .grad (requires_grad=false)")

	// Fused relu(a+b): Add's forward buffer is elided, but grads still flow.
	fmt.println("\n-- fused relu(a+b), both need grad --")
	a := ml.from_data_copy({1, -2, 3, -4}, {2, 2}, requires_grad = true)
	bb := ml.from_data_copy({10, 20, 30, 40}, {2, 2}, requires_grad = true)
	y := ml.relu(ml.add(a, bb))
	s := ml.sum(y, -1)
	ml.backward(s)
	fmt.printfln("  y = %v  (all > 0 → dA=dB=1)", y.data)
	fmt.printfln("  dA = %v", a.grad.data)
	fmt.printfln("  dB = %v", bb.grad.data)

	fmt.println("\n  rule of thumb:")
	fmt.println("  - leaves you want to train: requires_grad = true")
	fmt.println("  - backward(loss) seeds dL/dloss = 1 and builds grad UOps per op")
	fmt.println("  - forward + backward are scheduled together, so both get fused")
	fmt.println("  - relu grad = g * (0 < x): a CmpLt node, fused with the Mul")
}
