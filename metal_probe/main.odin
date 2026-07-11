package main

// Proof of concept: can Odin access the Apple GPU via Metal?
//   odin run metal_probe -o:speed

import "core:fmt"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

main :: proc() {
	fmt.println("=== Metal GPU probe ===")

	// Get the system's default Metal device (the GPU)
	device := MTL.CreateSystemDefaultDevice()
	if device == nil {
		fmt.println("FAIL: no Metal device available")
		return
	}

	// Device name
	name := MTL.Device_name(device)
	fmt.printfln("GPU device: %s", NS.String_UTF8String(name))

	// Create a command queue
	queue := MTL.Device_newCommandQueue(device)
	fmt.printfln("command queue created: %v", queue != nil)

	// Create a buffer — this is unified memory on Apple Silicon!
	// CPU and GPU access the SAME physical RAM. Zero copy.
	test_data := []f32{1.0, 2.0, 3.0, 4.0}
	buf := MTL.Device_newBufferWithSlice(device, test_data, {})
	fmt.printfln("buffer length: %d bytes", MTL.Buffer_length(buf))

	// Read back the buffer contents (zero-copy on Apple Silicon unified memory!)
	contents := MTL.Buffer_contentsAsSlice(buf, []f32)
	fmt.printfln("buffer contents (shared CPU/GPU mem): %v", contents)

	// Max threadgroup memory (tells us the GPU's workgroup limit)
	maxTg := MTL.Device_maxThreadgroupMemoryLength(device)
	fmt.printfln("max threadgroup memory: %d bytes", maxTg)

	// Buffer alignment
	align := MTL.Device_minimumLinearTextureAlignmentForPixelFormat(device, .R32Float)
	fmt.printfln("min linear texture alignment (R32Float): %d", align)

	fmt.println()
	fmt.println("=> Odin CAN access the Apple GPU via Metal.")
	fmt.println("=> Unified memory means zero-copy between CPU and GPU buffers.")
	fmt.println()
	fmt.println("Three paths to GPU compute from here:")
	fmt.println("  1. Metal compute shaders (MSL kernel strings compiled at runtime)")
	fmt.println("  2. MLX C API (Apple's ML framework, pure C, needs build+link)")
	fmt.println("  3. MPS via ObjC runtime (Metal Performance Shaders — no bindings yet)")
	fmt.println("done.")
}