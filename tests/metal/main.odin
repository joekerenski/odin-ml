package main

// ============================================================================
// Metal GPU smoke test — the raw kernel path works end-to-end.
//
//   odin run tests/metal -o:speed
//
// metal_add is a standalone prototype kernel (not yet wired into the UOp
// scheduler): compile MSL at runtime, dispatch on unified memory, compare
// against the same add realized on the CPU through the graph.
// ============================================================================

import "core:fmt"
import ml "../../ml"

main :: proc() {
	fmt.println("=== Metal GPU add test ===")

	if !ml.metal_init() {
		fmt.println("FAIL: no Metal device")
		return
	}

	for n in ([]int{8, 1024, 1_000_003}) {
		a := make([]f32, n)
		b := make([]f32, n)
		for i in 0 ..< n {
			a[i] = f32(i % 1000)
			b[i] = f32(i % 7) * 10
		}
		gpu := make([]f32, n)
		ml.metal_add(gpu, a, b)

		cpu := ml.add(ml.from_data(a, {i32(n)}), ml.from_data(b, {i32(n)}))
		ml.realize(cpu)

		ok := true
		for i in 0 ..< n do if gpu[i] != cpu.data[i] do ok = false
		fmt.printfln("%s: GPU add (n=%d) matches CPU graph", ok ? "PASS" : "FAIL", n)
	}
	fmt.println("done.")
}
