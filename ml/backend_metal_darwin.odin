package ml

// ============================================================================
// Metal backend — Apple Silicon GPU compute.
//
// Darwin-only file (suffix _darwin). On Linux this file is simply not
// compiled. backend_metal_stub.odin provides metal_add → CPU fallback.
//
// Architecture:
//   - metal_ctx holds the device, command queue, and a shader cache.
//   - Shaders are Metal Shading Language (MSL) strings compiled at runtime.
//   - Buffers use .Shared storage → unified memory. The CPU []f32 and the GPU
//     buffer point at the SAME physical RAM. Zero copy in both directions.
//   - Each op: encode → set pipeline → bind buffers → dispatch → commit → wait.
//
// This is the same pipeline PyTorch MPS uses. The difference is we own it.
// ============================================================================

import "core:fmt"
import "core:mem"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

Metal_Context :: struct {
	device:      ^MTL.Device,
	queue:       ^MTL.CommandQueue,
	shaders:     map[string]^MTL.ComputePipelineState,
	initialized: bool,
}

metal_ctx: Metal_Context

metal_init :: proc() -> bool {
	if metal_ctx.initialized do return true
	metal_ctx.device = MTL.CreateSystemDefaultDevice()
	if metal_ctx.device == nil do return false
	metal_ctx.queue = MTL.Device_newCommandQueue(metal_ctx.device)
	metal_ctx.shaders = make(map[string]^MTL.ComputePipelineState)
	metal_ctx.initialized = true
	fmt.printfln("Metal: device = %s", MTL.Device_name(metal_ctx.device)->UTF8String())
	return true
}

// Compile an MSL shader string and cache the pipeline state by function name.
metal_get_kernel :: proc(source: string, fn_name: string) -> ^MTL.ComputePipelineState {
	if fn_name in metal_ctx.shaders {
		return metal_ctx.shaders[fn_name]
	}

	// Create NSString from source
	ns_source := NS.String_alloc()
	ns_source = NS.String_initWithOdinString(ns_source, source)
	defer ns_source->release()

	// Compile library from source
	library, err := MTL.Device_newLibraryWithSource(metal_ctx.device, ns_source, nil)
	if err != nil {
		desc := NS.Error_localizedDescription(err)
		fmt.printfln("Metal: library compile error: %s", desc->UTF8String())
		return nil
	}

	// Get function from library
	ns_fn := NS.String_alloc()
	ns_fn = NS.String_initWithOdinString(ns_fn, fn_name)
	fn := MTL.Library_newFunctionWithName(library, ns_fn)
	ns_fn->release()
	library->release()

	if fn == nil {
		fmt.printfln("Metal: function '%s' not found in shader", fn_name)
		return nil
	}

	// Create compute pipeline state
	pipeline, err2 := MTL.Device_newComputePipelineStateWithFunction(metal_ctx.device, fn)
	fn->release()
	if err2 != nil {
		desc := NS.Error_localizedDescription(err2)
		fmt.printfln("Metal: pipeline error: %s", desc->UTF8String())
		return nil
	}

	metal_ctx.shaders[fn_name] = pipeline
	return pipeline
}

// Create a Metal buffer from a []f32 (copies data into GPU-accessible memory).
// On Apple Silicon unified memory, this is the same RAM — near-zero cost.
metal_new_buffer :: proc(data: []f32) -> ^MTL.Buffer {
	return MTL.Device_newBufferWithSlice(metal_ctx.device, data, {})
}

// Create an empty Metal buffer of given byte length.
metal_new_buffer_empty :: proc(byte_len: int) -> ^MTL.Buffer {
	return MTL.Device_newBufferWithLength(metal_ctx.device, NS.UInteger(byte_len), {})
}

// Run a 1D compute kernel over `n` elements.
// Binds buffers at indices 0..len(buffers)-1, dispatches, commits, waits.
metal_run_kernel :: proc(
	pipeline: ^MTL.ComputePipelineState,
	buffers: []^MTL.Buffer,
	n: int,
) {
	cmd_buffer := MTL.CommandQueue_commandBuffer(metal_ctx.queue)
	encoder := MTL.CommandBuffer_computeCommandEncoder(cmd_buffer)

	MTL.ComputeCommandEncoder_setComputePipelineState(encoder, pipeline)

	for i in 0..<len(buffers) {
		MTL.ComputeCommandEncoder_setBuffer(encoder, buffers[i], 0, NS.UInteger(i))
	}

	// Determine threadgroup size
	max_threads := MTL.ComputePipelineState_maxTotalThreadsPerThreadgroup(pipeline)
	threads_per_tg := min(int(max_threads), n)
	if threads_per_tg == 0 do threads_per_tg = 1

	threadgroups := (n + threads_per_tg - 1) / threads_per_tg

	MTL.ComputeCommandEncoder_dispatchThreadgroups(
		encoder,
		MTL.Size{width = NS.Integer(threadgroups), height = 1, depth = 1},
		MTL.Size{width = NS.Integer(threads_per_tg), height = 1, depth = 1},
	)

	MTL.CommandEncoder_endEncoding(encoder)
	MTL.CommandBuffer_commit(cmd_buffer)
	MTL.CommandBuffer_waitUntilCompleted(cmd_buffer)
}

// Read a Metal buffer's contents as []f32 (zero-copy on unified memory).
metal_buffer_to_f32 :: proc(buf: ^MTL.Buffer) -> []f32 {
	return MTL.Buffer_contentsAsSlice(buf, []f32)
}

// ---- elementwise add shader -----------------------------------------------
// Dispatch exactly n threads; no n-buffer needed (avoids the old f32→uint bit-hack).

metal_add_source :: `
#include <metal_stdlib>
using namespace metal;

kernel void elem_add(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    uint                tid  [[thread_position_in_grid]]
) {
    out[tid] = a[tid] + b[tid];
}
`

// GPU elementwise add: out = a + b (same shape, contiguous). Prototype only.
metal_add :: proc(out, a, b: []f32) {
	if !metal_init() {
		add_f32_contiguous(out, a, b)
		return
	}

	pipeline := metal_get_kernel(metal_add_source, "elem_add")
	if pipeline == nil {
		add_f32_contiguous(out, a, b)
		return
	}

	n := len(a)
	buf_a := metal_new_buffer(a)
	buf_b := metal_new_buffer(b)
	buf_out := metal_new_buffer_empty(n * size_of(f32))
	defer buf_a->release()
	defer buf_b->release()
	defer buf_out->release()

	metal_run_kernel(pipeline, {buf_a, buf_b, buf_out}, n)

	result := metal_buffer_to_f32(buf_out)
	copy(out, result)
}