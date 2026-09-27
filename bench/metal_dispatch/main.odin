package main

// Is a Metal backend worth it for our workload? A DT training step is ~1200
// small kernels (0.3–1.3M elements each), so what matters is per-kernel
// dispatch overhead, not peak FLOPs. Measures:
//   A. one kernel per command buffer (commit + wait each)   — naive dispatch
//   B. 1000 dependent kernels in ONE command buffer          — real backend
//   C. B at a tiny size                                      — pure overhead
// and the CPU path (fused kernel, all cores) on the same op and size.
//
//   odin run bench/metal_dispatch -o:speed

import "core:fmt"
import "core:time"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"
import ml "../../ml"

SRC :: `
#include <metal_stdlib>
using namespace metal;
kernel void step(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                 device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] * 0.999f + b[i];
}`

encode :: proc(enc: ^MTL.ComputeCommandEncoder, pso: ^MTL.ComputePipelineState, a, b, out: ^MTL.Buffer, n: u32) {
	n := n
	MTL.ComputeCommandEncoder_setComputePipelineState(enc, pso)
	MTL.ComputeCommandEncoder_setBuffer(enc, a, 0, 0)
	MTL.ComputeCommandEncoder_setBuffer(enc, b, 0, 1)
	MTL.ComputeCommandEncoder_setBuffer(enc, out, 0, 2)
	MTL.ComputeCommandEncoder_setBytes(enc, ([^]byte)(&n)[:4], 3)
	MTL.ComputeCommandEncoder_dispatchThreads(enc, {NS.Integer(n), 1, 1}, {256, 1, 1})
}

// K dependent kernels (ping-pong x ← x*0.999 + b) in one command buffer.
chain :: proc(pso: ^MTL.ComputePipelineState, bufs: [3]^MTL.Buffer, n: u32, K: int) -> (wall_ms, gpu_ms: f64) {
	t0 := time.tick_now()
	cb := MTL.CommandQueue_commandBuffer(ml.metal_ctx.queue)
	enc := MTL.CommandBuffer_computeCommandEncoder(cb) // serial: in-order, dependent
	for k in 0 ..< K {
		src, dst := bufs[k % 2], bufs[(k + 1) % 2]
		encode(enc, pso, src, bufs[2], dst, n)
	}
	MTL.CommandEncoder_endEncoding(enc)
	MTL.CommandBuffer_commit(cb)
	MTL.CommandBuffer_waitUntilCompleted(cb)
	wall_ms = time.duration_milliseconds(time.tick_since(t0))
	gpu_ms = f64(MTL.CommandBuffer_GPUEndTime(cb) - MTL.CommandBuffer_GPUStartTime(cb)) * 1000
	return
}

main :: proc() {
	if !ml.metal_init() do return
	pso := ml.metal_get_kernel(SRC, "step")
	K :: 1000

	for n in ([]u32{327_680, 1_310_720, 4096}) {
		bytes := int(n) * 4
		bufs: [3]^MTL.Buffer
		for &b in bufs do b = ml.metal_new_buffer_empty(bytes)

		// A: commit + wait per kernel
		t0 := time.tick_now()
		for k in 0 ..< 200 {
			cb := MTL.CommandQueue_commandBuffer(ml.metal_ctx.queue)
			enc := MTL.CommandBuffer_computeCommandEncoder(cb)
			encode(enc, pso, bufs[k % 2], bufs[2], bufs[(k + 1) % 2], n)
			MTL.CommandEncoder_endEncoding(enc)
			MTL.CommandBuffer_commit(cb)
			MTL.CommandBuffer_waitUntilCompleted(cb)
		}
		a_ms := time.duration_milliseconds(time.tick_since(t0)) / 200

		// B: one command buffer (warm up once, then best of 3)
		chain(pso, bufs, n, K)
		wall, gpu: f64 = 1e9, 1e9
		for _ in 0 ..< 3 {
			w, g := chain(pso, bufs, n, K)
			wall, gpu = min(wall, w), min(gpu, g)
		}

		// CPU: the same op through the graph (fused kernel, all cores)
		x := ml.randn({i32(n)}, 0, 1)
		b := ml.randn({i32(n)}, 0, 1)
		ml.realize(ml.add(ml.mul(x, ml.scalar(0.999)), b))
		t1 := time.tick_now()
		for _ in 0 ..< 200 do ml.realize(ml.add(ml.mul(x, ml.scalar(0.999)), b))
		cpu_ms := time.duration_milliseconds(time.tick_since(t1)) / 200

		fmt.printfln("n=%8d (%5.2f MB/buf)  A: %.3f ms/kernel   B: %.4f ms/kernel wall, %.4f GPU   CPU: %.4f ms/kernel",
			n, f64(bytes) / 1e6, a_ms, wall / K, gpu / K, cpu_ms)
	}
}
