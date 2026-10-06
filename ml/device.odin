package ml

// ============================================================================
// Devices. One setting picks where kernels run; model and training code
// don't change:
//
//   ml.set_device(.Metal)        or   ML_DEVICE=metal (ml.setup_from_env)
//   ml.set_device(.CUDA)         or   ML_DEVICE=cuda
//
// Tensor data stays a host []f32 on every device. Backends need memory the
// device can see too: unified memory on Metal, managed memory on CUDA.
// ml.arena_init gives per-step arenas on that memory, so a whole step is
// zero-copy. Data allocated elsewhere (heap params, test tensors) still works:
// backends stage it in and copy results back — slower, never wrong.
//
// A Backend runs the scheduler's kernels. Kernels may be queued; sync() makes
// every queued result visible on the host. Ops a backend doesn't implement
// (conv, pool) run on the CPU after a sync.
// ============================================================================

import "core:fmt"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"

Device :: enum {
	CPU,
	Metal,
	CUDA,
}

Backend :: struct {
	device:    Device,
	// memory the device can read and write directly (CPU: the heap)
	allocator: proc() -> mem.Allocator,
	kernel:    proc(k: ^Kernel), // any kernel of the IR (kernel_ir.odin)
	matmul:    proc(g: ^Gemm), // strided operands (kernel_ir.odin)
	sync:      proc(),
	// optional: move a buffer's pages to the host before CPU code touches it
	// (CUDA managed memory; one bulk migration instead of a fault per page)
	to_host:   proc(data: []f32),
}

CPU_BACKEND :: Backend {
	device    = .CPU,
	allocator = proc() -> mem.Allocator { return scratch() },
	kernel    = cpu_kernel,
	matmul    = cpu_gemm,
	sync      = proc() {},
}

backend: Backend = CPU_BACKEND

// Switch the device kernels run on. Returns false (and stays put) if the
// device isn't available on this machine.
set_device :: proc(d: Device) -> bool {
	backend.sync()
	switch d {
	case .CPU:
		backend = CPU_BACKEND
	case .Metal:
		b, ok := metal_backend()
		if !ok do return false
		backend = b
	case .CUDA:
		b, ok := cuda_backend()
		if !ok do return false
		backend = b
	}
	return true
}

get_device :: proc() -> Device {
	return backend.device
}

// Per-step arena on device-visible memory, with blocks big enough that
// tensors come from warm, reused memory after the first step.
// On CUDA, allocations of 4 KB and up go out of band to the backend allocator
// (cached, device-resident buffers; see backend_cuda_linux.odin) so kernel
// outputs never share managed pages with host-written graph nodes.
arena_init :: proc(a: ^mem.Dynamic_Arena, block_size := 64 * mem.Megabyte) {
	alloc := backend.allocator()
	out_band := backend.device == .CUDA ? 4096 : block_size / 2
	mem.dynamic_arena_init(a, block_allocator = alloc, array_allocator = alloc, block_size = block_size, out_band_size = out_band)
}

// ML_DEBUG=0..3, ML_DEVICE=cpu|metal|cuda|gpu, ML_THREADS=n (CPU threads, default
// all logical cores), ML_REUSE=0|1 (buffer reuse), ML_VIEWS=0|1 (views read
// in place), ML_REMAT=0|1 (recompute instead of store, remat.odin; default:
// GPUs only), ML_SEARCH=0|1 (time kernel choices on the device, search.odin)
// and ML_SCHED_CACHE=0|1 (replay scheduling decisions, schedule_cache.odin)
// from the environment.
setup_from_env :: proc() {
	debug_from_env()
	if v, ok := os.lookup_env_alloc("ML_REUSE", context.temp_allocator); ok do buffer_reuse = v != "0"
	if v, ok := os.lookup_env_alloc("ML_VIEWS", context.temp_allocator); ok do view_reads = v != "0"
	if v, ok := os.lookup_env_alloc("ML_REMAT", context.temp_allocator); ok do remat_mode = v == "0" ? .Off : .On
	if v, ok := os.lookup_env_alloc("ML_SEARCH", context.temp_allocator); ok do search_enabled = v != "0"
	if v, ok := os.lookup_env_alloc("ML_SCHED_CACHE", context.temp_allocator); ok do sched_cache_enabled = v != "0"
	if v, ok := os.lookup_env_alloc("ML_METAL_SERIAL", context.temp_allocator); ok do metal_concurrent = v == "0"
	if v, ok := os.lookup_env_alloc("ML_THREADS", context.temp_allocator); ok && !pool_ready {
		if n, ok2 := strconv.parse_int(v); ok2 && n > 0 do num_threads = n
	}
	v, found := os.lookup_env_alloc("ML_DEVICE", context.temp_allocator)
	if !found do return
	switch strings.to_lower(v, context.temp_allocator) {
	case "cpu":
		set_device(.CPU)
	case "metal":
		if !set_device(.Metal) do fmt.eprintln("ML_DEVICE=metal: Metal not available, staying on CPU")
	case "cuda":
		if !set_device(.CUDA) do fmt.eprintln("ML_DEVICE=cuda: CUDA not available, staying on CPU")
	case "gpu": // whichever this machine has
		if !set_device(.Metal) && !set_device(.CUDA) do fmt.eprintln("ML_DEVICE=gpu: no GPU backend available, staying on CPU")
	case:
		fmt.eprintfln("ML_DEVICE=%s: unknown device (cpu|metal|cuda|gpu)", v)
	}
}
