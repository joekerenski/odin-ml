package ml

// ============================================================================
// Metal backend (Apple GPU). Darwin only; backend_metal_stub.odin elsewhere.
//
// Memory. Tensor data stays a host []f32. On Apple Silicon a shared MTLBuffer
// IS host memory, so metal_allocator hands out buffer contents and keeps a
// registry base → MTLBuffer: any slice inside a registered block maps to
// (buffer, offset) with no copy. ml.arena_init puts per-step arenas on it.
// Data from other allocators (heap params, test tensors) is staged into
// scratch buffers on use; outputs landing there are copied back at sync.
//
// Execution. Every kernel of a realize() is encoded into ONE command buffer
// (serial encoder: in order, each sees the previous results); sync() commits
// and waits once. That's what makes ~1200 small kernels per step cheap
// (~1 µs dispatch each, bench/metal_dispatch).
//
// Kernels. Every scheduler kernel (kernel_ir.odin) is rendered to MSL by
// kernel_render.odin and compiled once per structure (cached by kernel_hash);
// shapes, strides and constants are parameters. GEMMs are generated per tile
// shape, picked per problem by timing (backend_metal_gemm_darwin.odin).
// Fast-math is off so results match the CPU path closely.
// ============================================================================

import "core:fmt"
import "core:mem"
import "core:strings"
import "core:time"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

Dev_Ref :: struct {
	buf: ^MTL.Buffer,
	off: int, // bytes
}

@(private = "file")
Dev_Block :: struct {
	base: uintptr,
	size: int,
	buf:  ^MTL.Buffer,
}

@(private = "file")
Copy_Back :: struct {
	host: []f32,
	ref:  Dev_Ref,
}

Metal_Context :: struct {
	device:      ^MTL.Device,
	queue:       ^MTL.CommandQueue,
	options:     ^MTL.CompileOptions,
	shaders:     map[string]^MTL.ComputePipelineState, // fixed kernels, by name
	programs:    map[u64]^MTL.ComputePipelineState, // fused groups, by shape hash
	initialized: bool,
	// memory
	blocks:      [dynamic]Dev_Block,
	scratch:     [dynamic]^MTL.Buffer,
	scratch_i:   int,
	scratch_off: int,
	staged:      map[uintptr]Dev_Ref, // host slice → where it lives in this batch
	copy_backs:  [dynamic]Copy_Back,
	// the open batch
	pool:        ^NS.AutoreleasePool,
	cmd:         ^MTL.CommandBuffer,
	enc:         ^MTL.ComputeCommandEncoder,
	opened:      time.Tick, // for ML_DEBUG timing
	n_dispatch:  int,
	in_flight:   [dynamic]^MTL.CommandBuffer, // committed, not yet waited on
}

// Commit every this many kernels without waiting, so the GPU runs while the
// CPU keeps encoding. Same queue + tracked buffers keep them in order.
@(private = "file")
COMMIT_EVERY :: 128

metal_ctx: Metal_Context

metal_init :: proc() -> bool {
	if metal_ctx.initialized do return true
	metal_ctx.device = MTL.CreateSystemDefaultDevice()
	if metal_ctx.device == nil do return false
	metal_ctx.queue = MTL.Device_newCommandQueue(metal_ctx.device)
	choices_open(fmt.tprintf("metal-%s", MTL.Device_name(metal_ctx.device)->odinString()))
	metal_ctx.options = MTL.CompileOptions_alloc()->init()
	MTL.CompileOptions_setFastMathEnabled(metal_ctx.options, false)
	metal_ctx.shaders = make(map[string]^MTL.ComputePipelineState, scratch())
	metal_ctx.programs = make(map[u64]^MTL.ComputePipelineState, scratch())
	metal_ctx.blocks = make([dynamic]Dev_Block, scratch())
	metal_ctx.scratch = make([dynamic]^MTL.Buffer, scratch())
	metal_ctx.staged = make(map[uintptr]Dev_Ref, scratch())
	metal_ctx.copy_backs = make([dynamic]Copy_Back, scratch())
	metal_ctx.in_flight = make([dynamic]^MTL.CommandBuffer, scratch())
	metal_ctx.initialized = true
	return true
}

metal_backend :: proc() -> (Backend, bool) {
	if !metal_init() do return {}, false
	return Backend {
		device    = .Metal,
		allocator = metal_allocator,
		kernel    = metal_kernel,
		matmul    = metal_matmul,
		sync      = metal_sync,
	}, true
}

// ---- compile --------------------------------------------------------------

@(private = "file")
metal_compile :: proc(source, fn_name: string) -> ^MTL.ComputePipelineState {
	ns_source := NS.String_alloc()->initWithOdinString(source)
	defer ns_source->release()
	library, err := MTL.Device_newLibraryWithSource(metal_ctx.device, ns_source, metal_ctx.options)
	if err != nil {
		fmt.panicf("Metal: compile error: %s\n%s", NS.Error_localizedDescription(err)->UTF8String(), source)
	}
	defer library->release()
	ns_fn := NS.String_alloc()->initWithOdinString(fn_name)
	defer ns_fn->release()
	fn := MTL.Library_newFunctionWithName(library, ns_fn)
	if fn == nil do fmt.panicf("Metal: function '%s' not found", fn_name)
	defer fn->release()
	pso, err2 := MTL.Device_newComputePipelineStateWithFunction(metal_ctx.device, fn)
	if err2 != nil do fmt.panicf("Metal: pipeline error: %s", NS.Error_localizedDescription(err2)->UTF8String())
	return pso
}

// Compile an MSL source once and cache the pipeline by function name.
metal_get_kernel :: proc(source: string, fn_name: string) -> ^MTL.ComputePipelineState {
	if pso, ok := metal_ctx.shaders[fn_name]; ok do return pso
	pso := metal_compile(source, fn_name)
	metal_ctx.shaders[strings.clone(fn_name, scratch())] = pso
	return pso
}

metal_new_buffer_empty :: proc(byte_len: int) -> ^MTL.Buffer {
	return MTL.Device_newBufferWithLength(metal_ctx.device, NS.UInteger(byte_len), {})
}

// ---- memory ---------------------------------------------------------------

// An allocator whose memory the GPU reads and writes directly (shared buffers).
metal_allocator :: proc() -> mem.Allocator {
	return {metal_allocator_proc, nil}
}

@(private = "file")
metal_allocator_proc :: proc(
	data: rawptr,
	mode: mem.Allocator_Mode,
	size, alignment: int,
	old_memory: rawptr,
	old_size: int,
	loc := #caller_location,
) -> (
	[]byte,
	mem.Allocator_Error,
) {
	#partial switch mode {
	case .Alloc, .Alloc_Non_Zeroed:
		buf := metal_new_buffer_empty(max(size, 1)) // shared storage; fresh pages are zero
		if buf == nil do return nil, .Out_Of_Memory
		contents := MTL.Buffer_contents(buf)
		append(&metal_ctx.blocks, Dev_Block{uintptr(raw_data(contents)), len(contents), buf})
		return contents[:size], nil
	case .Free:
		if old_memory == nil do return nil, nil
		for b, i in metal_ctx.blocks {
			if b.base == uintptr(old_memory) {
				b.buf->release()
				unordered_remove(&metal_ctx.blocks, i)
				return nil, nil
			}
		}
		return nil, .Invalid_Pointer
	case .Resize, .Resize_Non_Zeroed:
		new_mem, err := metal_allocator_proc(data, .Alloc, size, alignment, nil, 0, loc)
		if err != nil do return nil, err
		if old_memory != nil {
			copy(new_mem, ([^]byte)(old_memory)[:min(old_size, size)])
			metal_allocator_proc(data, .Free, 0, 0, old_memory, old_size, loc)
		}
		return new_mem, nil
	}
	return nil, .Mode_Not_Implemented
}

// Per-batch scratch (staging, temporaries), reset at sync.
@(private)
scratch_alloc :: proc(bytes: int) -> Dev_Ref {
	n := (bytes + 255) &~ 255
	for {
		if metal_ctx.scratch_i == len(metal_ctx.scratch) {
			append(&metal_ctx.scratch, metal_new_buffer_empty(max(64 * mem.Megabyte, n)))
			metal_ctx.scratch_off = 0
		}
		buf := metal_ctx.scratch[metal_ctx.scratch_i]
		if metal_ctx.scratch_off + n <= int(MTL.Buffer_length(buf)) {
			ref := Dev_Ref{buf, metal_ctx.scratch_off}
			metal_ctx.scratch_off += n
			return ref
		}
		metal_ctx.scratch_i += 1
		metal_ctx.scratch_off = 0
	}
}

@(private = "file")
ref_host :: proc(r: Dev_Ref, n: int) -> []f32 {
	return ([^]f32)(&MTL.Buffer_contents(r.buf)[r.off])[:n]
}

// Where a host slice lives on the device. Inputs from foreign memory are
// copied in; outputs to foreign memory are copied back at sync.
@(private)
resolve :: proc(s: []f32, output := false) -> Dev_Ref {
	p := uintptr(raw_data(s))
	if r, ok := metal_ctx.staged[p]; ok do return r
	for b in metal_ctx.blocks {
		if p >= b.base && p < b.base + uintptr(b.size) do return {b.buf, int(p - b.base)}
	}
	r := scratch_alloc(len(s) * size_of(f32))
	if output {
		append(&metal_ctx.copy_backs, Copy_Back{s, r})
	} else {
		copy(ref_host(r, len(s)), s)
	}
	metal_ctx.staged[p] = r
	return r
}

// ---- batching -------------------------------------------------------------

@(private = "file")
encoder :: proc() -> ^MTL.ComputeCommandEncoder {
	if metal_ctx.enc == nil {
		metal_ctx.pool = NS.AutoreleasePool_alloc()->init()
		metal_ctx.cmd = MTL.CommandQueue_commandBuffer(metal_ctx.queue)
		metal_ctx.enc = MTL.CommandBuffer_computeCommandEncoder(metal_ctx.cmd)
		metal_ctx.opened = time.tick_now()
		metal_ctx.n_dispatch = 0
	}
	return metal_ctx.enc
}

metal_sync :: proc() {
	if metal_ctx.enc == nil do return
	encode_ms := time.duration_milliseconds(time.tick_since(metal_ctx.opened))
	MTL.CommandEncoder_endEncoding(metal_ctx.enc)
	MTL.CommandBuffer_commit(metal_ctx.cmd)
	append(&metal_ctx.in_flight, metal_ctx.cmd)
	for cb in metal_ctx.in_flight {
		MTL.CommandBuffer_waitUntilCompleted(cb)
		if MTL.CommandBuffer_status(cb) == .Error {
			fmt.panicf("Metal: command buffer failed: %s", NS.Error_localizedDescription(MTL.CommandBuffer_error(cb))->UTF8String())
		}
	}
	if debug_level >= 1 {
		first, last := metal_ctx.in_flight[0], metal_ctx.in_flight[len(metal_ctx.in_flight) - 1]
		gpu_ms := f64(MTL.CommandBuffer_GPUEndTime(last) - MTL.CommandBuffer_GPUStartTime(first)) * 1000
		fmt.printfln("  metal: %d dispatches  encode %.2f ms  gpu %.2f ms  staged %d  copy-backs %d",
			metal_ctx.n_dispatch, encode_ms, gpu_ms, len(metal_ctx.staged), len(metal_ctx.copy_backs))
	}
	clear(&metal_ctx.in_flight)
	for cb in metal_ctx.copy_backs do copy(cb.host, ref_host(cb.ref, len(cb.host)))
	clear(&metal_ctx.copy_backs)
	clear(&metal_ctx.staged)
	metal_ctx.scratch_i, metal_ctx.scratch_off = 0, 0
	NS.AutoreleasePool_drain(metal_ctx.pool)
	metal_ctx.enc, metal_ctx.cmd, metal_ctx.pool = nil, nil, nil
}

// ML_DEBUG >= 2: submit each kernel on its own and print its GPU time
// (Apple GPUs timestamp per command buffer, not per dispatch). Slow; exact.
@(private)
dispatch :: proc(pso: ^MTL.ComputePipelineState, bufs: []Dev_Ref, params: []u32, grid: [3]int, group: [3]int, label := "") {
	enc := encoder()
	metal_ctx.n_dispatch += 1
	if metal_ctx.n_dispatch % COMMIT_EVERY == 0 && !kernel_timing() {
		MTL.CommandEncoder_endEncoding(enc)
		MTL.CommandBuffer_commit(metal_ctx.cmd)
		append(&metal_ctx.in_flight, metal_ctx.cmd)
		metal_ctx.cmd = MTL.CommandQueue_commandBuffer(metal_ctx.queue)
		metal_ctx.enc = MTL.CommandBuffer_computeCommandEncoder(metal_ctx.cmd)
		enc = metal_ctx.enc
	}
	MTL.ComputeCommandEncoder_setComputePipelineState(enc, pso)
	for r, i in bufs do if r.buf != nil do MTL.ComputeCommandEncoder_setBuffer(enc, r.buf, NS.UInteger(r.off), NS.UInteger(i))
	MTL.ComputeCommandEncoder_setBytes(enc, ([^]byte)(raw_data(params))[:len(params) * 4], 30)
	MTL.ComputeCommandEncoder_dispatchThreads(
		enc,
		{NS.Integer(grid[0]), NS.Integer(grid[1]), NS.Integer(grid[2])},
		{NS.Integer(group[0]), NS.Integer(group[1]), NS.Integer(group[2])},
	)
	if kernel_timing() {
		cmd := metal_ctx.cmd
		MTL.CommandEncoder_endEncoding(metal_ctx.enc)
		MTL.CommandBuffer_commit(cmd)
		MTL.CommandBuffer_waitUntilCompleted(cmd)
		gpu_ms := f64(MTL.CommandBuffer_GPUEndTime(cmd) - MTL.CommandBuffer_GPUStartTime(cmd)) * 1000
		profile_add_ms(gpu_ms, label)
		if debug_level >= 2 do fmt.printfln("  gpu     %-40s %8.4f ms", label, gpu_ms)
		// fresh command buffer; staged/scratch state carries over until the real sync
		NS.AutoreleasePool_drain(metal_ctx.pool)
		metal_ctx.pool = NS.AutoreleasePool_alloc()->init()
		metal_ctx.cmd = MTL.CommandQueue_commandBuffer(metal_ctx.queue)
		metal_ctx.enc = MTL.CommandBuffer_computeCommandEncoder(metal_ctx.cmd)
	}
}

// ---- kernel IR: rendered by kernel_render.odin, cached by structure ----------

@(private = "file")
program :: proc(k: ^Kernel, variant: GPU_Variant) -> ^MTL.ComputePipelineState {
	key := kernel_hash(k, int(variant))
	pso, ok := metal_ctx.programs[key]
	if !ok {
		pso = metal_compile(gpu_source(k, variant, .Metal), "k_main")
		metal_ctx.programs[key] = pso
	}
	return pso
}

metal_kernel :: proc(k: ^Kernel) {
	plan := gpu_plan(k)
	if plan.split > 1 {
		a, b, ap, bp := gpu_split(k, plan.split)
		partial := scratch_alloc(a.dims[0] * a.dims[2] * size_of(f32))
		run_plan(&a, .Reduce_Thread, a.dims[0] * a.dims[2], ap, partial)
		run_plan(&b, .Reduce_Thread, b.dims[0] * b.dims[2], bp, partial)
		return
	}
	run_plan(k, plan.variant, plan.threads, -1, {})
}

// One dispatch of k's program under a variant; buffer slot `partial_slot` is
// bound to the backend scratch `partial` (split reductions).
@(private = "file")
run_plan :: proc(k: ^Kernel, variant: GPU_Variant, threads, partial_slot: int, partial: Dev_Ref) {
	bufs: [MAX_KERNEL_BUFS]Dev_Ref
	for j in 0 ..< k.n_bufs do bufs[j] = j == partial_slot ? partial : resolve(k.bufs[j], output = j >= k.n_in)
	p: [GPU_PARAMS]u32
	n := gpu_params(k, &p)
	dispatch(program(k, variant), bufs[:k.n_bufs], p[:n], {threads, 1, 1}, {GPU_GROUP, 1, 1},
		k.label != "" ? fmt.tprintf("%v %s", variant, k.label) : "")
}

// ---- fixed kernels --------------------------------------------------------

@(private)
KERNELS :: `
#include <metal_stdlib>
using namespace metal;

// Split-K GEMM combine: [1, splits, m·n] → [m·n]. One thread per output.
kernel void reduce_sum_thread(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                              constant uint* p [[buffer(30)]], uint t [[thread_position_in_grid]]) {
    uint outer = p[0], red = p[1], inner = p[2];
    if (t >= outer * inner) return;
    uint o = t / inner, k = t % inner;
    float acc = 0.0f;
    for (uint r = 0; r < red; r++) acc += a[(o * red + r) * inner + k];
    out[t] = acc;
}

` + GEMM_P + `

// Tiny matrices (attention heads): one thread per output, no tiles. Grid (N, M, Z).
kernel void matmul_small(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant Gemm_P& p [[buffer(30)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.y, j = gid.x, z0 = gid.z / p.Z1, z1 = gid.z % p.Z1;
    if (i >= p.M || j >= p.N) return;
    device const float* a = A + z0 * p.a_b0 + z1 * p.a_b1 + i * p.a_rs;
    device const float* b = B + z0 * p.b_b0 + z1 * p.b_b1 + j * p.b_cs;
    float acc = 0.0f;
    for (uint q = 0; q < p.K; q++) acc += a[q * p.a_cs] * b[q * p.b_rs];
    C[z0 * p.c_b0 + z1 * p.c_b1 + i * p.c_rs + j * p.c_cs] = acc;
}
`

@(private)
round_up :: proc(x, m: int) -> int {
	return (x + m - 1) / m * m
}
