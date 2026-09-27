package ml

// ============================================================================
// Devices. One setting picks where kernels run; model and training code
// don't change:
//
//   ml.set_device(.Metal)        or   ML_DEVICE=metal (ml.setup_from_env)
//
// Tensor data stays a host []f32 on every device. Backends need memory the
// device can see too: unified memory on Metal (managed memory on CUDA later).
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
import "core:strings"

Device :: enum {
	CPU,
	Metal,
}

Backend :: struct {
	device:    Device,
	// memory the device can read and write directly (CPU: the heap)
	allocator: proc() -> mem.Allocator,
	fused:     proc(job: ^Fused_Job),
	reduce:    proc(op: Op, out, a: []f32, shape: []i32, axes: []i32),
	permute:   proc(out, a: []f32, shape: []i32, order: []i32),
	matmul:    proc(C, A, B: []f32, batch: int, M, K, N: i32, trans_a, trans_b: bool),
	sync:      proc(),
}

CPU_BACKEND :: Backend {
	device    = .CPU,
	allocator = proc() -> mem.Allocator { return scratch() },
	fused     = run_fused_kernel,
	reduce    = reduce_kernel,
	permute   = permute_kernel,
	matmul    = matmul_batched,
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
	}
	return true
}

get_device :: proc() -> Device {
	return backend.device
}

// Per-step arena on device-visible memory, with blocks big enough that
// tensors come from warm, reused memory after the first step.
arena_init :: proc(a: ^mem.Dynamic_Arena, block_size := 64 * mem.Megabyte) {
	alloc := backend.allocator()
	mem.dynamic_arena_init(a, block_allocator = alloc, array_allocator = alloc, block_size = block_size, out_band_size = block_size / 2)
}

// ML_DEBUG=0..3 and ML_DEVICE=cpu|metal from the environment.
setup_from_env :: proc() {
	debug_from_env()
	v, found := os.lookup_env_alloc("ML_DEVICE", context.temp_allocator)
	if !found do return
	switch strings.to_lower(v, context.temp_allocator) {
	case "cpu":
		set_device(.CPU)
	case "metal", "gpu":
		if !set_device(.Metal) do fmt.eprintln("ML_DEVICE=metal: Metal not available, staying on CPU")
	case:
		fmt.eprintfln("ML_DEVICE=%s: unknown device (cpu|metal)", v)
	}
}
