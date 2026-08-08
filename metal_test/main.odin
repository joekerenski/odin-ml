package main

// ============================================================================
// Metal GPU smoke test — prove the device dispatch works end-to-end.
//
//   odin run metal_test -o:speed
//
// Creates tensors on CPU, switches to Metal device, does add on GPU, verifies
// the result matches CPU computation. This is the first GPU op in odin-ml.
// ============================================================================

import "core:fmt"
import ml "../ml"

main :: proc() {
	fmt.println("=== Metal GPU add test ===")

	// Init Metal
	if !ml.metal_init() {
		fmt.println("FAIL: no Metal device")
		return
	}

	// Create two tensors on the Metal device
	a := ml.from_data_copy({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, {8}, false, .Metal)
	b := ml.from_data_copy({10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0}, {8}, false, .Metal)

	fmt.printfln("a (Metal): %v", a.data)
	fmt.printfln("b (Metal): %v", b.data)

	// GPU add — lazy graph; realize runs Metal kernel
	c := ml.add(a, b)
	ml.realize(c)

	fmt.printfln("c = a + b (GPU): %v", c.data)
	fmt.printfln("c.device = %v", c.device)

	// Verify against expected
	expected := []f32{11, 22, 33, 44, 55, 66, 77, 88}
	ok := true
	for i in 0..<len(expected) {
		if c.data[i] != expected[i] do ok = false
	}

	if ok {
		fmt.println("PASS: GPU add result matches expected")
	} else {
		fmt.println("FAIL: GPU add result mismatch")
	}

	// Larger test — 1024 elements
	n := 1024
	a2_data := make([]f32, n)
	b2_data := make([]f32, n)
	for i in 0..<n {
		a2_data[i] = f32(i)
		b2_data[i] = f32(i) * 2
	}
	a2 := ml.from_data_copy(a2_data, {i32(n)}, false, .Metal)
	b2 := ml.from_data_copy(b2_data, {i32(n)}, false, .Metal)

	c2 := ml.add(a2, b2)
	ml.realize(c2)

	// Verify
	ok2 := true
	for i in 0..<n {
		expected := f32(i) + f32(i) * 2
		if c2.data[i] != expected do ok2 = false
	}
	if ok2 {
		fmt.printfln("PASS: GPU add (n=%d) matches expected", n)
	} else {
		fmt.printfln("FAIL: GPU add (n=%d) mismatch", n)
	}

	fmt.println("done.")
}