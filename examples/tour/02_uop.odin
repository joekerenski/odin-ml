package main

// A Tensor IS a UOp node: `ml.Tensor :: ml.UOp`. There is one graph.
// Ops build nodes (op, src, arg, shape); realize() fills .data.
//
// Compositions are visible in the graph — relu is not a node type,
// it is Max(x, Const 0). MatMul/Conv/Pool/CE are primitives.
//
//   odin run examples/tour/02_uop.odin -file

import "core:fmt"
import ml "../../ml"

dump :: proc(u: ^ml.UOp, indent := 0) {
	for _ in 0 ..< indent do fmt.print("  ")
	arg := u.arg != nil ? fmt.tprintf("  arg=%v", u.arg) : ""
	fmt.printfln("%v shape=%v%s%s", u.op, u.shape, arg, u.data != nil ? "  (realized)" : "")
	for s in u.src do dump(s, indent + 1)
}

main :: proc() {
	fmt.println("=== 02 the UOp graph ===")

	x := ml.from_data_copy({-1, 2, -3, 4}, {4})

	fmt.println("\n-- relu(x) = max(x, 0) --")
	dump(ml.relu(x))

	fmt.println("\n-- sigmoid(x) = 1 / (1 + exp(-x)) --")
	dump(ml.sigmoid(x))

	fmt.println("\n-- mean(x) = sum(x) * (1/n) --")
	dump(ml.mean(x))

	fmt.println("\n-- movement and reduce carry their arg: permute order, sum axes --")
	m := ml.from_data_copy({1, 2, 3, 4, 5, 6}, {2, 3})
	dump(ml.T(m))
	dump(ml.sum(m, 1))

	fmt.println("\n-- lazy until realize --")
	y := ml.matmul(m, ml.T(m))
	fmt.printfln("  y.data == nil: %v", y.data == nil)
	ml.realize(y)
	fmt.printfln("  realized: %v  (m @ m^T = [[14,32],[32,77]])", y.data)
}
