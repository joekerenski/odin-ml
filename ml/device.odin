package ml

// ============================================================================
// Device — where a tensor's data lives and who computes its ops.
//
// The Tensor struct gets a `device` field. Ops check it to dispatch:
//   .CPU    → pure SIMD kernels / Accelerate BLAS
//   .Metal  → Metal compute shaders (Apple Silicon GPU, zero-copy unified mem)
//
// On Linux, .Metal simply doesn't exist — `when ODIN_OS == .Darwin` gates the
// Metal backend file entirely. The Device enum and CPU path compile everywhere.
//
// This mirrors tinygrad's design: the graph is device-agnostic (just records
// what ops happened), the backend decides how to run them. A future
// .CUDA / .Vulkan path plugs in the same way.
// ============================================================================

Device :: enum {
	CPU,
	Metal,
}

// Default device — can be changed at runtime.
default_device: Device = .CPU

set_default_device :: proc "contextless" (d: Device) {
	default_device = d
}