package ml

// Non-Darwin fallback: metal_add runs on the CPU.
// The real implementation lives in backend_metal_darwin.odin.

when ODIN_OS != .Darwin {
	metal_init :: proc() -> bool { return false }

	metal_add :: proc(out, a, b: []f32) {
		for i in 0 ..< len(out) do out[i] = a[i] + b[i]
	}
}
