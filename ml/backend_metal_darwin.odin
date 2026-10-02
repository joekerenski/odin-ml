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
// Kernels. Fused elementwise groups are rendered to MSL from their register
// program and compiled once per program shape (cached by a structural hash).
// Reduce, permute and batched GEMM are fixed kernels. Fast-math is off so
// results match the CPU path closely.
// ============================================================================

import "core:fmt"
import "core:hash"
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
		fused     = metal_fused,
		reduce    = metal_reduce,
		permute   = metal_permute,
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
@(private = "file")
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
@(private = "file")
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
@(private = "file")
dispatch :: proc(pso: ^MTL.ComputePipelineState, bufs: []Dev_Ref, params: []u32, grid: [3]int, group: [3]int, label := "") {
	enc := encoder()
	metal_ctx.n_dispatch += 1
	if metal_ctx.n_dispatch % COMMIT_EVERY == 0 && debug_level < 2 {
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
	if debug_level >= 2 {
		cmd := metal_ctx.cmd
		MTL.CommandEncoder_endEncoding(metal_ctx.enc)
		MTL.CommandBuffer_commit(cmd)
		MTL.CommandBuffer_waitUntilCompleted(cmd)
		gpu_ms := f64(MTL.CommandBuffer_GPUEndTime(cmd) - MTL.CommandBuffer_GPUStartTime(cmd)) * 1000
		fmt.printfln("  gpu     %-40s %8.4f ms", label, gpu_ms)
		// fresh command buffer; staged/scratch state carries over until the real sync
		NS.AutoreleasePool_drain(metal_ctx.pool)
		metal_ctx.pool = NS.AutoreleasePool_alloc()->init()
		metal_ctx.cmd = MTL.CommandQueue_commandBuffer(metal_ctx.queue)
		metal_ctx.enc = MTL.CommandBuffer_computeCommandEncoder(metal_ctx.cmd)
	}
}

// ---- fused elementwise: render the register program to MSL ----------------

// Params (u32): p[0] = n; input k: p[1+2k] = n_k (a Const: its f32 bits), p[2+2k] = inner_k;
// G = 1+2·n_in: p[G] = ndim, p[G+1..] = out shape, then for each Generic
// input its broadcast strides over the out dims.
@(private = "file")
fused_hash :: proc(job: ^Fused_Job) -> u64 {
	key: [256]u8
	n := 0
	put :: proc(key: ^[256]u8, n: ^int, v: int) {
		key[n^] = u8(v)
		n^ += 1
	}
	put(&key, &n, len(job.inputs))
	put(&key, &n, len(job.out_shape))
	for inp in job.inputs do put(&key, &n, int(inp.mode))
	for insn in job.insns {
		put(&key, &n, int(insn.op))
		put(&key, &n, insn.a)
		put(&key, &n, insn.b)
	}
	for st in job.stores do put(&key, &n, st.slot)
	return hash.fnv64a(key[:n])
}

@(private = "file")
fused_expr :: proc(op: Op, a, b: string) -> string {
	#partial switch op {
	case .Add: return fmt.tprintf("%s + %s", a, b)
	case .Sub: return fmt.tprintf("%s - %s", a, b)
	case .Mul: return fmt.tprintf("%s * %s", a, b)
	case .Div: return fmt.tprintf("%s / %s", a, b)
	case .Max: return fmt.tprintf("(%s > %s ? %s : %s)", a, b, a, b)
	case .CmpLt: return fmt.tprintf("(%s < %s ? 1.0f : 0.0f)", a, b)
	case .Neg: return fmt.tprintf("-%s", a)
	case .Exp: return fmt.tprintf("exp(%s)", a)
	case .Log: return fmt.tprintf("log(%s)", a)
	case .Sqrt: return fmt.tprintf("sqrt(%s)", a)
	case .Expand: return a
	}
	fmt.panicf("fused_expr: %v", op)
}

@(private = "file")
fused_source :: proc(job: ^Fused_Job) -> string {
	b := strings.builder_make(context.temp_allocator)
	n_in := len(job.inputs)
	G := 1 + 2 * n_in
	nd := len(job.out_shape)
	strings.write_string(&b, "#include <metal_stdlib>\nusing namespace metal;\nkernel void fused(\n")
	for inp, k in job.inputs do if inp.mode != .Const do fmt.sbprintf(&b, "    device const float* in%d [[buffer(%d)]],\n", k, k)
	for k in 0 ..< len(job.stores) do fmt.sbprintf(&b, "    device float* out%d [[buffer(%d)]],\n", k, n_in + k)
	strings.write_string(&b, "    constant uint* p [[buffer(30)]],\n    uint i [[thread_position_in_grid]]) {\n    if (i >= p[0]) return;\n")
	gen := 0
	for inp, k in job.inputs {
		switch inp.mode {
		case .Direct: fmt.sbprintf(&b, "    float s%d = in%d[i];\n", k, k)
		case .Scalar: fmt.sbprintf(&b, "    float s%d = in%d[0];\n", k, k)
		case .Const: fmt.sbprintf(&b, "    float s%d = as_type<float>(p[%d]);\n", k, 1 + 2 * k)
		case .Row: fmt.sbprintf(&b, "    float s%d = in%d[i %% p[%d]];\n", k, k, 1 + 2 * k)
		case .Col: fmt.sbprintf(&b, "    float s%d = in%d[i / p[%d]];\n", k, k, 2 + 2 * k)
		case .Block: fmt.sbprintf(&b, "    float s%d = in%d[(i / p[%d]) %% p[%d]];\n", k, k, 2 + 2 * k, 1 + 2 * k)
		case .Generic:
			S := G + 1 + nd + gen * nd
			fmt.sbprintf(&b, "    float s%d; {{ uint r = i, o = 0;\n", k)
			fmt.sbprintf(&b, "      for (int d = %d; d >= 0; d--) {{ uint c = r %% p[%d + d]; r /= p[%d + d]; o += c * p[%d + d]; }}\n", nd - 1, G + 1, G + 1, S)
			fmt.sbprintf(&b, "      s%d = in%d[o]; }}\n", k, k)
			gen += 1
		}
	}
	for insn, j in job.insns {
		a, bb := fmt.tprintf("s%d", insn.a), fmt.tprintf("s%d", insn.b)
		fmt.sbprintf(&b, "    float s%d = %s;\n", n_in + j, fused_expr(insn.op, a, bb))
	}
	for st, k in job.stores do fmt.sbprintf(&b, "    out%d[i] = s%d;\n", k, st.slot)
	strings.write_string(&b, "}\n")
	return strings.to_string(b)
}

metal_fused :: proc(job: ^Fused_Job) {
	key := fused_hash(job)
	pso, ok := metal_ctx.programs[key]
	if !ok {
		pso = metal_compile(fused_source(job), "fused")
		metal_ctx.programs[key] = pso
	}
	n := int(numel(job.out_shape))
	params: [1 + 2 * MAX_FUSED_INPUTS + 1 + MAX_DIMS * (1 + MAX_FUSED_INPUTS)]u32
	params[0] = u32(n)
	n_in := len(job.inputs)
	G := 1 + 2 * n_in
	nd := len(job.out_shape)
	params[G] = u32(nd)
	for d in 0 ..< nd do params[G + 1 + d] = u32(job.out_shape[d])
	gen := 0
	bufs: [MAX_FUSED_INPUTS + MAX_FUSED_INSNS]Dev_Ref
	for inp, k in job.inputs {
		params[1 + 2 * k] = u32(inp.n)
		params[2 + 2 * k] = u32(inp.inner)
		if inp.mode == .Generic {
			S := G + 1 + nd + gen * nd
			pad := nd - len(inp.shape)
			for d in 0 ..< nd {
				sd := d - pad
				params[S + d] = sd >= 0 && inp.shape[sd] != 1 ? u32(stride_of(inp.shape, sd)) : 0
			}
			gen += 1
		}
		if inp.mode == .Const {
			params[1 + 2 * k] = transmute(u32)inp.value
			continue // no buffer: bufs[k] stays unbound
		}
		bufs[k] = resolve(inp.data)
	}
	for st, k in job.stores do bufs[n_in + k] = resolve(st.data, output = true)
	used := G + 1 + nd + gen * nd
	label := ""
	if debug_level >= 2 {
		b := strings.builder_make(context.temp_allocator)
		strings.write_string(&b, "fused[")
		for insn, i in job.insns do fmt.sbprintf(&b, "%s%v", i > 0 ? "," : "", insn.op)
		fmt.sbprintf(&b, "] %v", job.out_shape)
		label = strings.to_string(b)
	}
	dispatch(pso, bufs[:n_in + len(job.stores)], params[:used], {n, 1, 1}, {256, 1, 1}, label)
}

// ---- fixed kernels --------------------------------------------------------

@(private = "file")
KERNELS :: `
#include <metal_stdlib>
using namespace metal;

// [outer, red, inner] → [outer, inner]. One thread per output...
#define REDUCE_THREAD(NAME, INIT, ACC)                                          \
kernel void NAME(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], \
                 constant uint* p [[buffer(30)]], uint t [[thread_position_in_grid]]) { \
    uint outer = p[0], red = p[1], inner = p[2];                               \
    if (t >= outer * inner) return;                                            \
    uint o = t / inner, k = t % inner;                                         \
    float acc = INIT;                                                          \
    for (uint r = 0; r < red; r++) { float v = a[(o * red + r) * inner + k]; acc = ACC; } \
    out[t] = acc;                                                              \
}
// ...or, for long reductions, one 256-thread group per output.
#define REDUCE_GROUP(NAME, INIT, ACC, COMB)                                     \
kernel void NAME(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], \
                 constant uint* p [[buffer(30)]], uint t [[thread_index_in_threadgroup]], \
                 uint g [[threadgroup_position_in_grid]]) {                    \
    uint red = p[1], inner = p[2];                                             \
    uint o = g / inner, k = g % inner;                                         \
    threadgroup float sh[256];                                                 \
    float acc = INIT;                                                          \
    for (uint r = t; r < red; r += 256) { float v = a[(o * red + r) * inner + k]; acc = ACC; } \
    sh[t] = acc;                                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint s = 128; s > 0; s >>= 1) {                                       \
        if (t < s) { float x = sh[t], v = sh[t + s]; sh[t] = COMB; }           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    if (t == 0) out[g] = sh[0];                                                \
}
REDUCE_THREAD(reduce_sum_thread, 0.0f, acc + v)
REDUCE_THREAD(reduce_max_thread, -INFINITY, (v > acc ? v : acc))
REDUCE_GROUP(reduce_sum_group, 0.0f, acc + v, x + v)
REDUCE_GROUP(reduce_max_group, -INFINITY, (v > acc ? v : acc), (v > x ? v : x))

// out.shape[i] = shape[order[i]]; p = [nd, n, out_shape[8], src_stride[8]]
kernel void permute(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                    constant uint* p [[buffer(30)]], uint i [[thread_position_in_grid]]) {
    uint nd = p[0];
    if (i >= p[1]) return;
    uint r = i, off = 0;
    for (int d = int(nd) - 1; d >= 0; d--) { uint c = r % p[2 + d]; r /= p[2 + d]; off += c * p[10 + d]; }
    out[i] = a[off];
}

// C[z] = op(A[z]) @ op(B[z]); 16×16 tiles in threadgroup memory.
// p = [M, K, N, trans_a, trans_b, kc]. Grid (N↑16, M↑16, Z), groups 16×16.
// kc == 0: z is the batch index. kc > 0 (split-K): all z share A and B, z
// covers K range [z·kc, (z+1)·kc) and writes a partial C[z] (summed after).
kernel void matmul(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]], constant uint* p [[buffer(30)]],
                   uint3 gid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    uint M = p[0], K = p[1], N = p[2], kc = p[5];
    bool ta = p[3] != 0, tb = p[4] != 0;
    uint i = gid.y, j = gid.x;
    device const float* a = kc ? A : A + gid.z * M * K;
    device const float* b = kc ? B : B + gid.z * K * N;
    uint k_lo = kc ? gid.z * kc : 0, k_hi = kc ? min(K, k_lo + kc) : K;
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];
    float acc = 0.0f;
    for (uint t0 = k_lo; t0 < k_hi; t0 += 16) {
        uint pa = t0 + lid.x, pb = t0 + lid.y;
        As[lid.y][lid.x] = (i < M && pa < k_hi) ? (ta ? a[pa * M + i] : a[i * K + pa]) : 0.0f;
        Bs[lid.y][lid.x] = (pb < k_hi && j < N) ? (tb ? b[j * K + pb] : b[pb * N + j]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0; q < 16; q++) acc += As[lid.y][q] * Bs[q][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < M && j < N) C[gid.z * M * N + i * N + j] = acc;
}

// Hardware 8×8 matrix units (simdgroup_matrix — what tinygrad's Metal tensor
// cores use), fed from threadgroup memory. A 128-thread group computes a 32×32
// tile of C; per K step of 32 all threads load A[32×32] and B[32×32] with
// coalesced reads (index mapping picked per transpose flag), zero-padding the
// edges, then each of the 4 SIMD groups does 2×2 8×8 tiles × 4 k-steps.
// Any M, N, K. p = [M, K, N, trans_a, trans_b, kc] as in matmul.
// Grid: (⌈N/32⌉·128, ⌈M/32⌉, Z), groups of 128.
kernel void matmul_sg(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]], constant uint* p [[buffer(30)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    uint M = p[0], K = p[1], N = p[2], kc = p[5];
    bool ta = p[3] != 0, tb = p[4] != 0;
    device const float* a = kc ? A : A + tg.z * M * K;
    device const float* b = kc ? B : B + tg.z * K * N;
    device float* c = C + tg.z * M * N;
    uint k_lo = kc ? tg.z * kc : 0, k_hi = kc ? min(K, k_lo + kc) : K;
    uint i0 = tg.y * 32, j0 = tg.x * 32;
    threadgroup float As[32][32]; // [row i][k]
    threadgroup float Bs[32][32]; // [k][col j]
    uint si = (sg / 2) * 16, sj = (sg % 2) * 16; // this SIMD group's 16×16 corner
    simdgroup_float8x8 acc[2][2];
    for (uint r = 0; r < 2; r++) for (uint q = 0; q < 2; q++) acc[r][q] = simdgroup_float8x8(0.0f);
    for (uint k0 = k_lo; k0 < k_hi; k0 += 32) {
        for (uint e = tid; e < 1024; e += 128) {
            uint r = ta ? e % 32 : e / 32, kk = ta ? e / 32 : e % 32; // consecutive threads → consecutive addresses
            uint gi = i0 + r, gk = k0 + kk;
            As[r][kk] = (gi < M && gk < k_hi) ? (ta ? a[gk * M + gi] : a[gi * K + gk]) : 0.0f;
            uint q = tb ? e / 32 : e % 32, kb = tb ? e % 32 : e / 32;
            uint gj = j0 + q, gkb = k0 + kb;
            Bs[kb][q] = (gj < N && gkb < k_hi) ? (tb ? b[gj * K + gkb] : b[gkb * N + gj]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 am0, am1, bm0, bm1;
            simdgroup_load(am0, &As[si][kk], 32);
            simdgroup_load(am1, &As[si + 8][kk], 32);
            simdgroup_load(bm0, &Bs[kk][sj], 32);
            simdgroup_load(bm1, &Bs[kk][sj + 8], 32);
            simdgroup_multiply_accumulate(acc[0][0], am0, bm0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], am0, bm1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], am1, bm0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], am1, bm1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // through threadgroup memory, so edges are written with bounds checks
    for (uint r = 0; r < 2; r++) for (uint q = 0; q < 2; q++) simdgroup_store(acc[r][q], &As[si + 8 * r][sj + 8 * q], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 1024; e += 128) {
        uint r = e / 32, q = e % 32;
        if (i0 + r < M && j0 + q < N) c[(i0 + r) * N + j0 + q] = As[r][q];
    }
}

// Tiny matrices (attention heads): one thread per output, no tiles.
// p = [M, K, N, trans_a, trans_b]. Grid (N, M, batch).
kernel void matmul_small(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant uint* p [[buffer(30)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint M = p[0], K = p[1], N = p[2];
    bool ta = p[3] != 0, tb = p[4] != 0;
    uint i = gid.y, j = gid.x;
    if (i >= M || j >= N) return;
    device const float* a = A + gid.z * M * K;
    device const float* b = B + gid.z * K * N;
    float acc = 0.0f;
    for (uint q = 0; q < K; q++) acc += (ta ? a[q * M + i] : a[i * K + q]) * (tb ? b[j * K + q] : b[q * N + j]);
    C[gid.z * M * N + i * N + j] = acc;
}
`

@(private = "file")
round_up :: proc(x, m: int) -> int {
	return (x + m - 1) / m * m
}

// Same run decomposition as the CPU reduce_kernel; temporaries in scratch.
metal_reduce :: proc(op: Op, out, a: []f32, shape: []i32, axes: []i32) {
	red: [MAX_DIMS]bool
	for ax in axes do red[ax] = true
	cur_shape: [MAX_DIMS]i32
	copy(cur_shape[:], shape)
	nd := len(shape)
	src := resolve(a)
	did := false
	d := nd - 1
	for d >= 0 {
		if !red[d] || cur_shape[d] == 1 {
			d -= 1
			continue
		}
		hi := d + 1
		for d >= 0 && red[d] do d -= 1
		lo := d + 1
		outer := int(numel(cur_shape[:lo]))
		r := int(numel(cur_shape[lo:hi]))
		inner := int(numel(cur_shape[hi:nd]))
		for k in lo ..< hi do cur_shape[k] = 1
		more := false
		for k in 0 ..< lo do if red[k] && cur_shape[k] != 1 do more = true
		dst := more ? scratch_alloc(outer * inner * size_of(f32)) : resolve(out, output = true)
		rows := outer * inner
		params := []u32{u32(outer), u32(r), u32(inner)}
		name: string
		split := 1 // chunks of the reduced axis, for long column reductions
		if inner >= 16 && r >= 512 && rows < 8192 {
			for s in ([]int{64, 32, 16, 8}) do if r % s == 0 && split == 1 do split = s
		}
		if split > 1 {
			// [outer, split, r/split, inner] → partial [outer, split, inner] → dst
			name = op == .Sum ? "reduce_sum_thread" : "reduce_max_thread"
			partial := scratch_alloc(outer * split * inner * size_of(f32))
			dispatch(metal_get_kernel(KERNELS, name), {src, partial}, []u32{u32(outer * split), u32(r / split), u32(inner)}, {outer * split * inner, 1, 1}, {256, 1, 1},
				debug_level >= 2 ? fmt.tprintf("%s split %d×%d×%d/%d", name, outer, r, inner, split) : "")
			dispatch(metal_get_kernel(KERNELS, name), {partial, dst}, []u32{u32(outer), u32(split), u32(inner)}, {rows, 1, 1}, {256, 1, 1},
				debug_level >= 2 ? fmt.tprintf("%s combine", name) : "")
		} else if r >= 128 && rows < 8192 {
			name = op == .Sum ? "reduce_sum_group" : "reduce_max_group"
			dispatch(metal_get_kernel(KERNELS, name), {src, dst}, params, {rows * 256, 1, 1}, {256, 1, 1}, debug_level >= 2 ? fmt.tprintf("%s %d×%d×%d", name, outer, r, inner) : "")
		} else {
			name = op == .Sum ? "reduce_sum_thread" : "reduce_max_thread"
			dispatch(metal_get_kernel(KERNELS, name), {src, dst}, params, {rows, 1, 1}, {256, 1, 1}, debug_level >= 2 ? fmt.tprintf("%s %d×%d×%d", name, outer, r, inner) : "")
		}
		src = dst
		did = true
	}
	if !did do metal_permute(out, a, {i32(len(a))}, {0}) // nothing to reduce: copy
}

metal_permute :: proc(out, a: []f32, shape: []i32, order: []i32) {
	params: [18]u32
	nd := len(shape)
	params[0], params[1] = u32(nd), u32(len(out))
	for o, i in order {
		params[2 + i] = u32(shape[o])
		params[10 + i] = u32(stride_of(shape, int(o)))
	}
	dispatch(metal_get_kernel(KERNELS, "permute"), {resolve(a), resolve(out, output = true)}, params[:], {len(out), 1, 1}, {256, 1, 1}, debug_level >= 2 ? fmt.tprintf("permute %v %v", shape, order) : "")
}

metal_matmul :: proc(C, A, B: []f32, batch: int, M, K, N: i32, trans_a, trans_b: bool) {
	m, k, n := int(M), int(K), int(N)
	a, b, c := resolve(A), resolve(B), resolve(C, output = true)
	params := []u32{u32(M), u32(K), u32(N), u32(trans_a), u32(trans_b), 0}
	sg := true // hardware 8×8 tiles; the plain tiled kernel stays as a reference
	kernel := sg ? "matmul_sg" : "matmul"
	grid :: proc(sg: bool, m, n, z: int) -> [3]int {
		return sg ? {(n + 31) / 32 * 128, (m + 31) / 32, z} : {round_up(n, 16), round_up(m, 16), z}
	}
	group := sg ? [3]int{128, 1, 1} : [3]int{16, 16, 1}
	switch {
	case m <= 16 && n <= 16 && k <= 64:
		// attention-sized: one thread per output
		dispatch(metal_get_kernel(KERNELS, "matmul_small"), {a, b, c}, params, {n, m, batch}, {n, m, 1}, debug_level >= 2 ? fmt.tprintf("matmul_small %dx%dx%d b%d", m, k, n, batch) : "")
	case batch == 1 && k >= 1024 && m * n <= 64 * 1024:
		// few outputs, long K (weight grads): split K, then sum the partials
		splits := min(k / 256, 64)
		kc := round_up((k + splits - 1) / splits, 32) // whole K tiles per split
		splits = (k + kc - 1) / kc
		params[5] = u32(kc)
		partial := scratch_alloc(splits * m * n * size_of(f32))
		dispatch(metal_get_kernel(KERNELS, kernel), {a, b, partial}, params, grid(sg, m, n, splits), group, debug_level >= 2 ? fmt.tprintf("%s splitk %dx%dx%d s%d", kernel, m, k, n, splits) : "")
		dispatch(metal_get_kernel(KERNELS, "reduce_sum_thread"), {partial, c}, []u32{1, u32(splits), u32(m * n)}, {m * n, 1, 1}, {256, 1, 1}, "splitk_sum")
	case:
		dispatch(metal_get_kernel(KERNELS, kernel), {a, b, c}, params, grid(sg, m, n, batch), group, debug_level >= 2 ? fmt.tprintf("%s %dx%dx%d b%d", kernel, m, k, n, batch) : "")
	}
}
