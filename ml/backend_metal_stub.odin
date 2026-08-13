package ml

// Non-Darwin fallback so realize.odin can call metal_add unconditionally.
// The real implementation lives in backend_metal_darwin.odin.

when ODIN_OS != .Darwin {
	metal_add :: proc(out, a, b: []f32) {
		add_f32_contiguous(out, a, b)
	}
}
