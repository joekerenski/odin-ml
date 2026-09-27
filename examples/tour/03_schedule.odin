package main

// What realize() does with the graph:
//   1. topo-sort the unrealized nodes,
//   2. fuse chains of same-shape elementwise ops into ONE loop
//      (intermediates are never stored),
//   3. fold a transpose feeding a matmul into the GEMM call,
//   4. run: fused kernels, views, primitive kernels (GEMM, conv, …).
//
//   odin run examples/tour/03_schedule.odin -file
//   ML_DEBUG=3 odin run examples/tour/03_schedule.odin -file   (show schedules)

import "core:fmt"
import ml "../../ml"

main :: proc() {
	fmt.println("=== 03 schedule: fuse, fold, run ===\n")
	ml.debug_from_env()
	if ml.debug_level < 2 do ml.debug_level = 2 // one line per kernel

	X := ml.from_data_copy({1, 0, 0, 1, -1, 2}, {3, 2})
	W := ml.from_data_copy({1, -1}, {2, 1})
	b := ml.from_data_copy({0.5}, {1})

	fmt.println("1. relu(X@W + b): MatMul + one fused [Add, Max] kernel")
	ml.counters_reset()
	y := ml.relu(ml.add(ml.matmul(X, W), b))
	ml.realize(y)
	ml.counters_print("relu(X@W+b)")
	fmt.printfln("  y = %v   (expect [1.5, 0, 0])\n", y.data)

	fmt.println("2. relu(a+c)*c: all elementwise → 1 kernel, 1 buffer")
	a := ml.from_data_copy({1, -2, 3}, {3})
	c := ml.from_data_copy({10, 10, 10}, {3})
	ml.counters_reset()
	z := ml.mul(ml.relu(ml.add(a, c)), c)
	ml.realize(z)
	ml.counters_print("relu(a+c)*c")
	fmt.printfln("  z = %v  (Add, Max, Mul fused; only z stored)\n", z.data)

	fmt.println("3. X^T @ X: the Permute folds into GEMM (no transpose kernel)")
	ml.counters_reset()
	g := ml.matmul(ml.T(X), X)
	ml.realize(g)
	ml.counters_print("X^T@X")
	fmt.printfln("  g = %v\n", g.data)

	fmt.println("4. sigmoid over a batch with a row bias: broadcast loads stay fused")
	h := ml.from_data_copy({0, 1, 2, 3, 4, 5}, {2, 3})
	bias := ml.from_data_copy({-1, 0, 1}, {3})
	ml.counters_reset()
	s := ml.sigmoid(ml.add(h, bias))
	ml.realize(s)
	ml.counters_print("sigmoid(h+bias)")
	ml.debug_level = 0
}
