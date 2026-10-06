package ml

// ============================================================================
// CUDA backend (NVIDIA GPU). Linux only; backend_cuda_stub.odin elsewhere.
//
// Loading. libcuda (driver API), NVRTC and cuBLAS are opened at runtime, so
// any build runs on machines without them: set_device(.CUDA) returns false.
//
// Memory. Tensor data stays a host []f32. cuda_allocator hands out managed
// memory (cuMemAllocManaged): one address valid on host and device, pages
// migrate on demand. ml.arena_init puts per-step arenas on it, so a whole
// step is zero-copy. Pages must not ping-pong: a step writes graph nodes and
// its input batch on the host and kernel outputs on the GPU, so on CUDA the
// arena sends everything but small allocations out of band, to this
// allocator, which caches freed blocks by size. Each step then gets back the
// same buffers — already on the GPU, never touched by the host — while the
// arena's own blocks hold the host-side data. Data from other allocators (heap params, test tensors)
// is staged on use — host memcpy into pinned memory, then an async copy into
// device scratch (a pageable source would make the copy wait for the whole
// stream) — and outputs landing there are copied back at sync.
//
// Execution. Kernels queue on one stream, in order (each sees the previous
// results); sync() waits once per realize.
//
// Kernels. Every scheduler kernel (kernel_ir.odin) is rendered to CUDA C by
// kernel_render.odin (the same renderer as Metal) and compiled by NVRTC once
// per structure (cached by kernel_hash). GEMM is cuBLAS in plain fp32 (no
// TF32), tiny batched GEMMs a hand-written kernel. No fast-math and no FMA
// contraction, so results match the CPU path closely.
// ============================================================================

import "core:dynlib"
import "core:fmt"
import "core:mem"
import "core:strings"
import "core:time"

// ---- driver API, NVRTC, cuBLAS (loaded at runtime) -------------------------

@(private = "file")
CUresult :: i32
@(private = "file")
CUdevice :: i32
@(private = "file")
CUdeviceptr :: u64
@(private = "file")
CUcontext :: distinct rawptr
@(private = "file")
CUstream :: distinct rawptr
@(private = "file")
CUmodule :: distinct rawptr
@(private = "file")
CUfunction :: distinct rawptr
@(private = "file")
CUevent :: distinct rawptr
@(private = "file")
CUmemLocation :: struct {
	type: i32,
	id:   i32,
}
@(private = "file")
nvrtcProgram :: distinct rawptr
@(private = "file")
cublasHandle :: distinct rawptr

@(private = "file")
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT :: 16
@(private = "file")
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR :: 75
@(private = "file")
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR :: 76
@(private = "file")
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS :: 89
@(private = "file")
CU_MEM_ATTACH_GLOBAL :: 1
@(private = "file")
CU_MEM_LOCATION_TYPE_DEVICE :: 1
@(private = "file")
CU_MEM_LOCATION_TYPE_HOST :: 2
@(private = "file")
CUBLAS_OP_N :: 0
@(private = "file")
CUBLAS_OP_T :: 1
@(private = "file")
CUBLAS_DEFAULT_MATH :: 0

@(private = "file")
Cuda_Api :: struct {
	__handle:                 dynlib.Library,
	cuInit:                   proc "c" (flags: u32) -> CUresult,
	cuDeviceGet:              proc "c" (dev: ^CUdevice, ordinal: i32) -> CUresult,
	cuDeviceGetAttribute:     proc "c" (v: ^i32, attrib: i32, dev: CUdevice) -> CUresult,
	cuDeviceGetName:          proc "c" (name: [^]u8, n: i32, dev: CUdevice) -> CUresult,
	cuDevicePrimaryCtxRetain: proc "c" (ctx: ^CUcontext, dev: CUdevice) -> CUresult,
	cuCtxSetCurrent:          proc "c" (ctx: CUcontext) -> CUresult,
	cuStreamCreate:           proc "c" (s: ^CUstream, flags: u32) -> CUresult,
	cuStreamSynchronize:      proc "c" (s: CUstream) -> CUresult,
	cuMemAllocManaged:        proc "c" (p: ^CUdeviceptr, bytes: uint, flags: u32) -> CUresult,
	cuMemAlloc:               proc "c" (p: ^CUdeviceptr, bytes: uint) -> CUresult `dynlib:"cuMemAlloc_v2"`,
	cuMemFree:                proc "c" (p: CUdeviceptr) -> CUresult `dynlib:"cuMemFree_v2"`,
	cuMemAllocHost:           proc "c" (p: ^rawptr, bytes: uint) -> CUresult `dynlib:"cuMemAllocHost_v2"`,
	cuMemcpyHtoDAsync:        proc "c" (dst: CUdeviceptr, src: rawptr, bytes: uint, s: CUstream) -> CUresult `dynlib:"cuMemcpyHtoDAsync_v2"`,
	cuMemcpyDtoH:             proc "c" (dst: rawptr, src: CUdeviceptr, bytes: uint) -> CUresult `dynlib:"cuMemcpyDtoH_v2"`,
	cuMemPrefetchAsync:       proc "c" (p: CUdeviceptr, bytes: uint, loc: CUmemLocation, flags: u32, s: CUstream) -> CUresult `dynlib:"cuMemPrefetchAsync_v2"`,
	cuModuleLoadData:         proc "c" (m: ^CUmodule, image: rawptr) -> CUresult,
	cuModuleGetFunction:      proc "c" (f: ^CUfunction, m: CUmodule, name: cstring) -> CUresult,
	cuLaunchKernel:           proc "c" (f: CUfunction, gx, gy, gz, bx, by, bz: u32, shmem: u32, s: CUstream, params: [^]rawptr, extra: [^]rawptr) -> CUresult,
	cuGetErrorString:         proc "c" (r: CUresult, s: ^cstring) -> CUresult,
	cuEventCreate:            proc "c" (e: ^CUevent, flags: u32) -> CUresult,
	cuEventRecord:            proc "c" (e: CUevent, s: CUstream) -> CUresult,
	cuEventSynchronize:       proc "c" (e: CUevent) -> CUresult,
	cuEventElapsedTime:       proc "c" (ms: ^f32, a, b: CUevent) -> CUresult `dynlib:"cuEventElapsedTime_v2"`,
}

@(private = "file")
Nvrtc_Api :: struct {
	__handle:               dynlib.Library,
	nvrtcCreateProgram:     proc "c" (p: ^nvrtcProgram, src, name: cstring, n_headers: i32, headers, include_names: [^]cstring) -> i32,
	nvrtcCompileProgram:    proc "c" (p: nvrtcProgram, n_opts: i32, opts: [^]cstring) -> i32,
	nvrtcGetCUBINSize:      proc "c" (p: nvrtcProgram, n: ^uint) -> i32,
	nvrtcGetCUBIN:          proc "c" (p: nvrtcProgram, out: [^]u8) -> i32,
	nvrtcGetProgramLogSize: proc "c" (p: nvrtcProgram, n: ^uint) -> i32,
	nvrtcGetProgramLog:     proc "c" (p: nvrtcProgram, out: [^]u8) -> i32,
	nvrtcDestroyProgram:    proc "c" (p: ^nvrtcProgram) -> i32,
}

@(private = "file")
Cublas_Api :: struct {
	__handle:                  dynlib.Library,
	cublasCreate:              proc "c" (h: ^cublasHandle) -> i32 `dynlib:"cublasCreate_v2"`,
	cublasSetStream:           proc "c" (h: cublasHandle, s: CUstream) -> i32 `dynlib:"cublasSetStream_v2"`,
	cublasSetMathMode:         proc "c" (h: cublasHandle, mode: i32) -> i32,
	cublasSgemm:               proc "c" (h: cublasHandle, ta, tb: i32, m, n, k: i32, alpha: ^f32, A: CUdeviceptr, lda: i32, B: CUdeviceptr, ldb: i32, beta: ^f32, C: CUdeviceptr, ldc: i32) -> i32 `dynlib:"cublasSgemm_v2"`,
	cublasSgemmStridedBatched: proc "c" (h: cublasHandle, ta, tb: i32, m, n, k: i32, alpha: ^f32, A: CUdeviceptr, lda: i32, sA: i64, B: CUdeviceptr, ldb: i32, sB: i64, beta: ^f32, C: CUdeviceptr, ldc: i32, sC: i64, batch: i32) -> i32,
}

@(private = "file")
cu: Cuda_Api
@(private = "file")
nvrtc: Nvrtc_Api
@(private = "file")
cublas: Cublas_Api

// Every proc field of an API table got a symbol.
@(private = "file")
all_loaded :: proc(table: ^$T) -> bool {
	ptrs := ([^]rawptr)(table)[:size_of(T) / size_of(rawptr)]
	for p in ptrs do if p == nil do return false
	return true
}

@(private = "file")
check :: proc(r: CUresult, what: string, loc := #caller_location) {
	if r == 0 do return
	msg: cstring = "?"
	cu.cuGetErrorString(r, &msg)
	fmt.panicf("CUDA: %s failed: %s (%d)", what, msg, r, loc = loc)
}

// ---- state ----------------------------------------------------------------

@(private = "file")
Dev_Block :: struct {
	base: uintptr,
	size: int,
}

@(private = "file")
Staged :: struct {
	ptr:       CUdeviceptr,
	copy_back: bool,
}

@(private = "file")
Copy_Back :: struct {
	host: []f32,
	ptr:  CUdeviceptr,
}

// Bump-allocated blocks, reset at sync.
@(private = "file")
Bump :: struct {
	blocks: [dynamic]uintptr,
	sizes:  [dynamic]int,
	i, off: int,
}

Cuda_Context :: struct {
	device:      CUdevice,
	ctx:         CUcontext,
	stream:      CUstream,
	blas:        cublasHandle,
	arch:        string, // "sm_89"
	name:        string,
	sms:         int,
	fixed:       CUmodule,
	kernels:     map[string]CUfunction, // fixed kernels, by name
	programs:    map[u64]CUfunction, // IR kernels, by structural hash
	initialized: bool,
	// memory
	blocks:      [dynamic]Dev_Block, // managed allocations (cuda_allocator), sorted by base
	cache:       map[int][dynamic]uintptr, // freed managed blocks by size, for reuse
	scratch:     Bump, // device memory: staging, temporaries
	pinned:      Bump, // pinned host memory: staging sources
	staged:      map[uintptr]Staged, // host slice → its device copy this batch
	copy_backs:  [dynamic]Copy_Back,
	// the open batch
	n_launch:    int,
	opened:      time.Tick,
	ev_start:    CUevent,
	ev_end:      CUevent,
	ev_a, ev_b:  CUevent, // ML_DEBUG=2 per-kernel timing
}

cuda_ctx: Cuda_Context

// Kernel parameters: one struct by value, the same size for every kernel.
// Fused: p[0] = n; input k: p[1+2k] = n_k (a Const: its f32 bits),
// p[2+2k] = inner_k; G = 1+2·n_in: p[G] = ndim, p[G+1..] = out shape, then
// each Generic input's broadcast strides over the out dims (as on Metal).
@(private = "file")
PARAMS :: GPU_PARAMS // the shared parameter block (kernel_render.odin)

@(private = "file")
BLOCK :: 256

cuda_init :: proc() -> bool {
	if cuda_ctx.initialized do return true
	if n, _ := dynlib.initialize_symbols(&cu, "libcuda.so.1"); n <= 0 || !all_loaded(&cu) do return false
	if cu.cuInit(0) != 0 do return false
	if cu.cuDeviceGet(&cuda_ctx.device, 0) != 0 do return false
	n_nvrtc, _ := dynlib.initialize_symbols(&nvrtc, "libnvrtc.so.13")
	if n_nvrtc <= 0 do n_nvrtc, _ = dynlib.initialize_symbols(&nvrtc, "libnvrtc.so")
	if n_nvrtc <= 0 || !all_loaded(&nvrtc) {
		fmt.eprintln("CUDA: libnvrtc not found (CUDA toolkit lib64 on the library path?)")
		return false
	}
	n_blas, _ := dynlib.initialize_symbols(&cublas, "libcublas.so.13")
	if n_blas <= 0 do n_blas, _ = dynlib.initialize_symbols(&cublas, "libcublas.so")
	if n_blas <= 0 || !all_loaded(&cublas) {
		fmt.eprintln("CUDA: libcublas not found (CUDA toolkit lib64 on the library path?)")
		return false
	}

	dev := cuda_ctx.device
	check(cu.cuDevicePrimaryCtxRetain(&cuda_ctx.ctx, dev), "cuDevicePrimaryCtxRetain")
	check(cu.cuCtxSetCurrent(cuda_ctx.ctx), "cuCtxSetCurrent")
	major, minor, sms, managed: i32
	cu.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
	cu.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
	cu.cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
	cu.cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, dev)
	if managed == 0 {
		fmt.eprintln("CUDA: device lacks concurrent managed access; not supported")
		return false
	}
	name: [256]u8
	cu.cuDeviceGetName(&name[0], len(name), dev)
	cuda_ctx.name = strings.clone_from_cstring(cstring(&name[0]), scratch())
	cuda_ctx.arch = fmt.aprintf("sm_%d%d", major, minor, allocator = scratch())
	cuda_ctx.sms = int(sms)

	check(cu.cuStreamCreate(&cuda_ctx.stream, 0), "cuStreamCreate")
	if cublas.cublasCreate(&cuda_ctx.blas) != 0 {
		fmt.eprintln("CUDA: cublasCreate failed")
		return false
	}
	cublas.cublasSetStream(cuda_ctx.blas, cuda_ctx.stream)
	cublas.cublasSetMathMode(cuda_ctx.blas, CUBLAS_DEFAULT_MATH) // fp32: no TF32 tensor ops
	for e in ([]^CUevent{&cuda_ctx.ev_start, &cuda_ctx.ev_end, &cuda_ctx.ev_a, &cuda_ctx.ev_b}) {
		check(cu.cuEventCreate(e, 0), "cuEventCreate")
	}

	cuda_ctx.kernels = make(map[string]CUfunction, scratch())
	cuda_ctx.programs = make(map[u64]CUfunction, scratch())
	cuda_ctx.blocks = make([dynamic]Dev_Block, scratch())
	cuda_ctx.cache = make(map[int][dynamic]uintptr, scratch())
	cuda_ctx.staged = make(map[uintptr]Staged, scratch())
	cuda_ctx.copy_backs = make([dynamic]Copy_Back, scratch())
	cuda_ctx.scratch = Bump{blocks = make([dynamic]uintptr, scratch()), sizes = make([dynamic]int, scratch())}
	cuda_ctx.pinned = Bump{blocks = make([dynamic]uintptr, scratch()), sizes = make([dynamic]int, scratch())}
	cuda_ctx.fixed = cuda_compile(KERNELS)
	cuda_ctx.initialized = true
	return true
}

cuda_backend :: proc() -> (Backend, bool) {
	if !cuda_init() do return {}, false
	return Backend {
		device    = .CUDA,
		allocator = cuda_allocator,
		kernel    = cuda_kernel,
		matmul    = cuda_matmul,
		sync      = cuda_sync,
		to_host   = cuda_to_host,
	}, true
}

// Name of the device, e.g. "NVIDIA GeForce RTX 4090 (sm_89, 128 SMs)".
cuda_device_name :: proc() -> string {
	if !cuda_ctx.initialized do return ""
	return fmt.tprintf("%s (%s, %d SMs)", cuda_ctx.name, cuda_ctx.arch, cuda_ctx.sms)
}

// ---- compile --------------------------------------------------------------

// CUDA C → cubin for this GPU (NVRTC), loaded as a module. Every source gets
// the params struct P (kernels take it by value, last).
@(private = "file")
cuda_compile :: proc(body: string) -> CUmodule {
	source := fmt.tprintf("struct P {{ unsigned int v[%d]; }};\n%s", PARAMS, body)
	src := strings.clone_to_cstring(source, context.temp_allocator)
	prog: nvrtcProgram
	if r := nvrtc.nvrtcCreateProgram(&prog, src, "odin_ml.cu", 0, nil, nil); r != 0 {
		fmt.panicf("NVRTC: create program failed (%d)", r)
	}
	defer nvrtc.nvrtcDestroyProgram(&prog)
	arch := strings.clone_to_cstring(fmt.tprintf("--gpu-architecture=%s", cuda_ctx.arch), context.temp_allocator)
	opts := []cstring{arch, "--fmad=false", "--std=c++17"}
	if nvrtc.nvrtcCompileProgram(prog, i32(len(opts)), raw_data(opts)) != 0 {
		n: uint
		nvrtc.nvrtcGetProgramLogSize(prog, &n)
		log := make([]u8, n + 1, context.temp_allocator)
		nvrtc.nvrtcGetProgramLog(prog, raw_data(log))
		fmt.panicf("NVRTC: compile error:\n%s\n%s", string(log[:n]), source)
	}
	n: uint
	nvrtc.nvrtcGetCUBINSize(prog, &n)
	cubin := make([]u8, n, context.temp_allocator)
	nvrtc.nvrtcGetCUBIN(prog, raw_data(cubin))
	mod: CUmodule
	check(cu.cuModuleLoadData(&mod, raw_data(cubin)), "cuModuleLoadData")
	return mod
}

@(private = "file")
get_function :: proc(mod: CUmodule, name: string) -> CUfunction {
	f: CUfunction
	check(cu.cuModuleGetFunction(&f, mod, strings.clone_to_cstring(name, context.temp_allocator)), fmt.tprintf("cuModuleGetFunction %s", name))
	return f
}

@(private = "file")
fixed_kernel :: proc(name: string) -> CUfunction {
	if f, ok := cuda_ctx.kernels[name]; ok do return f
	f := get_function(cuda_ctx.fixed, name)
	cuda_ctx.kernels[strings.clone(name, scratch())] = f
	return f
}

// ---- memory ---------------------------------------------------------------

// An allocator whose memory both the host and the GPU address directly.
cuda_allocator :: proc() -> mem.Allocator {
	return {cuda_allocator_proc, nil}
}

@(private = "file")
cuda_allocator_proc :: proc(
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
		n := (max(size, 1) + 4095) &~ 4095 // whole pages: blocks never share one
		base: uintptr
		reused := false
		if l, ok := &cuda_ctx.cache[n]; ok && len(l) > 0 {
			base, reused = pop(l), true
		} else {
			p: CUdeviceptr
			if cu.cuMemAllocManaged(&p, uint(n), CU_MEM_ATTACH_GLOBAL) != 0 do return nil, .Out_Of_Memory
			base = uintptr(p)
			i := block_index(base) + 1 // keep sorted
			inject_at(&cuda_ctx.blocks, i, Dev_Block{base, n})
		}
		bytes := ([^]byte)(base)[:size]
		if mode == .Alloc { // zeroed: the host writes it (a new input tensor)
			if reused do cuda_to_host(([^]f32)(base)[:size / size_of(f32)])
			mem.zero_slice(bytes)
		}
		return bytes, nil
	case .Free:
		if old_memory == nil do return nil, nil
		i := block_index(uintptr(old_memory))
		if i < 0 || cuda_ctx.blocks[i].base != uintptr(old_memory) do return nil, .Invalid_Pointer
		b := cuda_ctx.blocks[i]
		l, ok := &cuda_ctx.cache[b.size]
		if !ok {
			cuda_ctx.cache[b.size] = make([dynamic]uintptr, scratch())
			l = &cuda_ctx.cache[b.size]
		}
		append(l, b.base) // kept registered: still managed memory
		return nil, nil
	case .Resize, .Resize_Non_Zeroed:
		new_mem, err := cuda_allocator_proc(data, .Alloc, size, alignment, nil, 0, loc)
		if err != nil do return nil, err
		if old_memory != nil {
			copy(new_mem, ([^]byte)(old_memory)[:min(old_size, size)])
			cuda_allocator_proc(data, .Free, 0, 0, old_memory, old_size, loc)
		}
		return new_mem, nil
	}
	return nil, .Mode_Not_Implemented
}

// Index of the last managed block with base <= p (-1 if none).
@(private = "file")
block_index :: proc(p: uintptr) -> int {
	lo, hi := 0, len(cuda_ctx.blocks)
	for lo < hi {
		mid := (lo + hi) / 2
		if cuda_ctx.blocks[mid].base <= p do lo = mid + 1
		else do hi = mid
	}
	return lo - 1
}

@(private = "file")
bump_alloc :: proc(b: ^Bump, bytes: int, pinned: bool) -> uintptr {
	n := (bytes + 255) &~ 255
	for {
		if b.i == len(b.blocks) {
			size := max(64 * mem.Megabyte, n)
			p: uintptr
			if pinned {
				host: rawptr
				check(cu.cuMemAllocHost(&host, uint(size)), "cuMemAllocHost")
				p = uintptr(host)
			} else {
				d: CUdeviceptr
				check(cu.cuMemAlloc(&d, uint(size)), "cuMemAlloc")
				p = uintptr(d)
			}
			append(&b.blocks, p)
			append(&b.sizes, size)
			b.off = 0
		}
		if b.off + n <= b.sizes[b.i] {
			p := b.blocks[b.i] + uintptr(b.off)
			b.off += n
			return p
		}
		b.i += 1
		b.off = 0
	}
}

// Device scratch for this batch (staging, temporaries); reset at sync.
@(private = "file")
scratch_alloc :: proc(bytes: int) -> CUdeviceptr {
	return CUdeviceptr(bump_alloc(&cuda_ctx.scratch, bytes, false))
}

// Where a host slice lives on the device: itself if it's managed memory,
// else a staged copy (inputs copied in now, outputs copied back at sync).
@(private = "file")
resolve :: proc(s: []f32, output := false) -> CUdeviceptr {
	p := uintptr(raw_data(s))
	if st, ok := &cuda_ctx.staged[p]; ok {
		if output && !st.copy_back {
			append(&cuda_ctx.copy_backs, Copy_Back{s, st.ptr})
			st.copy_back = true
		}
		return st.ptr
	}
	if i := block_index(p); i >= 0 && p < cuda_ctx.blocks[i].base + uintptr(cuda_ctx.blocks[i].size) {
		return CUdeviceptr(p) // managed: same address on the device
	}
	bytes := len(s) * size_of(f32)
	d := scratch_alloc(bytes)
	if output {
		append(&cuda_ctx.copy_backs, Copy_Back{s, d})
	} else {
		pin := bump_alloc(&cuda_ctx.pinned, bytes, true)
		mem.copy_non_overlapping(rawptr(pin), raw_data(s), bytes)
		check(cu.cuMemcpyHtoDAsync(d, rawptr(pin), uint(bytes), cuda_ctx.stream), "cuMemcpyHtoDAsync")
	}
	cuda_ctx.staged[p] = Staged{d, output}
	return d
}

// Migrate a managed buffer to the host in one go, before CPU kernels (conv,
// pool fallbacks) read or write it: 20 threads faulting page by page on
// GPU-resident memory is ~100× slower. The GPU pulls it back on first use.
cuda_to_host :: proc(data: []f32) {
	p := uintptr(raw_data(data))
	i := block_index(p)
	if len(data) == 0 || i < 0 || p >= cuda_ctx.blocks[i].base + uintptr(cuda_ctx.blocks[i].size) do return
	check(cu.cuMemPrefetchAsync(CUdeviceptr(p), uint(len(data) * size_of(f32)), CUmemLocation{CU_MEM_LOCATION_TYPE_HOST, 0}, 0, cuda_ctx.stream), "cuMemPrefetchAsync")
	check(cu.cuStreamSynchronize(cuda_ctx.stream), "cuStreamSynchronize")
}

// ---- batching -------------------------------------------------------------

cuda_sync :: proc() {
	if cuda_ctx.n_launch == 0 && len(cuda_ctx.copy_backs) == 0 && len(cuda_ctx.staged) == 0 do return
	encode_ms := time.duration_milliseconds(time.tick_since(cuda_ctx.opened))
	if debug_level == 1 do cu.cuEventRecord(cuda_ctx.ev_end, cuda_ctx.stream)
	check(cu.cuStreamSynchronize(cuda_ctx.stream), "cuStreamSynchronize (a kernel failed?)")
	if debug_level == 1 && cuda_ctx.n_launch > 0 {
		gpu_ms: f32
		cu.cuEventElapsedTime(&gpu_ms, cuda_ctx.ev_start, cuda_ctx.ev_end)
		fmt.printfln("  cuda: %d launches  encode %.2f ms  gpu %.2f ms  staged %d  copy-backs %d",
			cuda_ctx.n_launch, encode_ms, gpu_ms, len(cuda_ctx.staged), len(cuda_ctx.copy_backs))
	}
	for cb in cuda_ctx.copy_backs {
		check(cu.cuMemcpyDtoH(raw_data(cb.host), cb.ptr, uint(len(cb.host) * size_of(f32))), "cuMemcpyDtoH")
	}
	clear(&cuda_ctx.copy_backs)
	clear(&cuda_ctx.staged)
	cuda_ctx.scratch.i, cuda_ctx.scratch.off = 0, 0
	cuda_ctx.pinned.i, cuda_ctx.pinned.off = 0, 0
	cuda_ctx.n_launch = 0
}

// Queue a kernel: buffer pointers, then the params struct. ML_DEBUG >= 2
// times each kernel on the GPU (events around it, then a sync). Slow; exact.
@(private = "file")
launch :: proc(f: CUfunction, bufs: []CUdeviceptr, params: []u32, grid: [3]int, block: [3]int, label := "") {
	if cuda_ctx.n_launch == 0 {
		cuda_ctx.opened = time.tick_now()
		if debug_level == 1 do cu.cuEventRecord(cuda_ctx.ev_start, cuda_ctx.stream)
	}
	cuda_ctx.n_launch += 1
	p: [PARAMS]u32
	copy(p[:], params)
	ptrs: [MAX_KERNEL_BUFS]CUdeviceptr
	args: [MAX_KERNEL_BUFS + 1]rawptr
	for b, i in bufs {
		ptrs[i] = b
		args[i] = &ptrs[i]
	}
	args[len(bufs)] = &p
	if kernel_timing() do cu.cuEventRecord(cuda_ctx.ev_a, cuda_ctx.stream)
	check(cu.cuLaunchKernel(f, u32(grid[0]), u32(grid[1]), u32(grid[2]), u32(block[0]), u32(block[1]), u32(block[2]), 0, cuda_ctx.stream, &args[0], nil), "cuLaunchKernel")
	if kernel_timing() {
		cu.cuEventRecord(cuda_ctx.ev_b, cuda_ctx.stream)
		check(cu.cuEventSynchronize(cuda_ctx.ev_b), fmt.tprintf("kernel %s", label))
		ms: f32
		cu.cuEventElapsedTime(&ms, cuda_ctx.ev_a, cuda_ctx.ev_b)
		profile_add_ms(f64(ms), label)
		if debug_level >= 2 do fmt.printfln("  gpu     %-40s %8.4f ms", label, ms)
	}
}

@(private = "file")
blocks_for :: proc(n: int) -> int {
	return max(1, (n + BLOCK - 1) / BLOCK)
}

// ---- kernel IR: rendered by kernel_render.odin, cached by structure ----------

@(private = "file")
program :: proc(k: ^Kernel, variant: GPU_Variant) -> CUfunction {
	key := kernel_hash(k, int(variant))
	f, ok := cuda_ctx.programs[key]
	if !ok {
		f = get_function(cuda_compile(gpu_source(k, variant, .CUDA)), "k_main")
		cuda_ctx.programs[key] = f
	}
	return f
}

// Launch shape of a plan: GPU_GROUP threads per block; groups = one block per output.
@(private = "file")
plan_grid :: proc(threads: int) -> [3]int {
	return {blocks_for(threads), 1, 1}
}

cuda_kernel :: proc(k: ^Kernel) {
	plan := gpu_plan(k)
	if plan.split > 1 {
		a, b, ap, bp := gpu_split(k, plan.split)
		partial := scratch_alloc(a.dims[0] * a.dims[2] * size_of(f32))
		run_plan(&a, .Reduce_Thread, a.dims[0] * a.dims[2], ap, partial)
		run_plan(&b, .Reduce_Thread, b.dims[0] * b.dims[2], bp, partial)
		return
	}
	run_plan(k, plan.variant, plan.threads, -1, 0)
}

// One launch of k's program under a variant; buffer slot `partial_slot` is
// bound to the backend scratch `partial` (split reductions).
@(private = "file")
run_plan :: proc(k: ^Kernel, variant: GPU_Variant, threads, partial_slot: int, partial: CUdeviceptr) {
	bufs: [MAX_KERNEL_BUFS]CUdeviceptr
	for j in 0 ..< k.n_bufs do bufs[j] = j == partial_slot ? partial : resolve(k.bufs[j], output = j >= k.n_in)
	p: [GPU_PARAMS]u32
	n := gpu_params(k, &p)
	launch(program(k, variant), bufs[:k.n_bufs], p[:n], plan_grid(threads), {GPU_GROUP, 1, 1},
		k.label != "" ? fmt.tprintf("%v %s", variant, k.label) : "")
}

// ---- fixed kernels --------------------------------------------------------

@(private = "file")
KERNELS :: `
// GEMM over strided operands, one thread per output: tiny batched matrices
// (attention heads; cuBLAS takes 60-150 µs for 8192 × [4×8]·[8×4], this a few)
// and layouts cuBLAS can't take. Element (z0, z1, i, j) of X at
// X[z0·b0 + z1·b1 + i·rs + j·cs]; p = [M, K, N, kc, Z1, then rs, cs, b0, b1 of A, B, C, Z].
extern "C" __global__ void matmul_small(const float* A, const float* B, float* C, const P p) {
    unsigned int M = p.v[0], K = p.v[1], N = p.v[2], Z1 = p.v[4], Z = p.v[17];
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Z * M * N) return;
    unsigned int z = t / (M * N), r = t % (M * N), i = r / N, j = r % N, z0 = z / Z1, z1 = z % Z1;
    const float* a = A + z0 * p.v[7] + z1 * p.v[8] + i * p.v[5];
    const float* b = B + z0 * p.v[11] + z1 * p.v[12] + j * p.v[10];
    float acc = 0.0f;
    for (unsigned int k = 0; k < K; k++) acc += a[k * p.v[6]] * b[k * p.v[9]];
    C[z0 * p.v[15] + z1 * p.v[16] + i * p.v[13] + j * p.v[14]] = acc;
}

`

// Row-major C = A·B is column-major Cᵀ = Bᵀ·Aᵀ: hand cuBLAS B first. A
// column-major C is row-major Cᵀ = Bᵀ·Aᵀ: swap the operands and transpose
// them. Batches: strided-batched over z1, one call per z0. Layouts without a
// unit stride go to the one-thread-per-output kernel.
cuda_matmul :: proc(g: ^Gemm) {
	M, K, N, Z := g.M, g.K, g.N, g.Z0 * g.Z1
	ta, lda, aok := gemm_layout(g.a, M, K)
	tb, ldb, bok := gemm_layout(g.b, K, N)
	tc, ldc, cok := gemm_layout(g.c, M, N)
	if (Z > 1 && M <= 16 && N <= 16 && K <= 64) || !(aok && bok && cok) {
		n := Z * M * N
		label := kernel_timing() ? fmt.tprintf("matmul_small %dx%dx%d b%d", M, K, N, Z) : ""
		p: [18]u32
		gp := gemm_params(g)
		copy(p[:], gp[:])
		p[17] = u32(Z)
		launch(fixed_kernel("matmul_small"), {resolve(g.a.data), resolve(g.b.data), resolve(g.c.data, output = true)},
			p[:], {blocks_for(n), 1, 1}, {BLOCK, 1, 1}, label)
		return
	}
	if cuda_ctx.n_launch == 0 {
		cuda_ctx.opened = time.tick_now()
		if debug_level == 1 do cu.cuEventRecord(cuda_ctx.ev_start, cuda_ctx.stream)
	}
	a, b, c := resolve(g.a.data), resolve(g.b.data), resolve(g.c.data, output = true)
	// column-major (first) operand X·Y with X [m, k]: cuBLAS op and leading dim
	x, y := b, a
	tx, ty, ldx, ldy := tb, ta, ldb, lda
	bx, by := g.b.bs, g.a.bs
	m, n := N, M
	if tc { // Cᵀ row-major = C column-major: Aᵀ… swap roles
		x, y = a, b
		tx, ty, ldx, ldy = !ta, !tb, lda, ldb
		bx, by = g.a.bs, g.b.bs
		m, n = M, N
	}
	alpha, beta: f32 = 1, 0
	op_x := i32(tx ? CUBLAS_OP_T : CUBLAS_OP_N)
	op_y := i32(ty ? CUBLAS_OP_T : CUBLAS_OP_N)
	if kernel_timing() do cu.cuEventRecord(cuda_ctx.ev_a, cuda_ctx.stream)
	st: i32
	for z0 in 0 ..< g.Z0 {
		off :: proc(p: CUdeviceptr, e: int) -> CUdeviceptr { return p + CUdeviceptr(e * size_of(f32)) }
		xp, yp, cp := off(x, z0 * bx[0]), off(y, z0 * by[0]), off(c, z0 * g.c.bs[0])
		if g.Z1 == 1 {
			st = cublas.cublasSgemm(cuda_ctx.blas, op_x, op_y, i32(m), i32(n), i32(K), &alpha, xp, i32(ldx), yp, i32(ldy), &beta, cp, i32(ldc))
		} else {
			st = cublas.cublasSgemmStridedBatched(cuda_ctx.blas, op_x, op_y, i32(m), i32(n), i32(K), &alpha,
				xp, i32(ldx), i64(bx[1]), yp, i32(ldy), i64(by[1]), &beta, cp, i32(ldc), i64(g.c.bs[1]), i32(g.Z1))
		}
		if st != 0 do fmt.panicf("cuBLAS: sgemm failed (%d) %dx%dx%d batch %d", st, M, K, N, Z)
		cuda_ctx.n_launch += 1
	}
	if kernel_timing() {
		cu.cuEventRecord(cuda_ctx.ev_b, cuda_ctx.stream)
		check(cu.cuEventSynchronize(cuda_ctx.ev_b), "sgemm")
		ms: f32
		cu.cuEventElapsedTime(&ms, cuda_ctx.ev_a, cuda_ctx.ev_b)
		profile_add_ms(f64(ms), Z == 1 ? "cublas_sgemm" : "cublas_sgemm_batched")
		if debug_level >= 2 do fmt.printfln("  gpu     %-40s %8.4f ms", fmt.tprintf("sgemm %dx%dx%d b%d", M, K, N, Z), ms)
	}
}
