package ml

// Non-Darwin: no Metal. set_device(.Metal) returns false.

when ODIN_OS != .Darwin {
	metal_backend :: proc() -> (Backend, bool) {
		return {}, false
	}
	metal_concurrent := true
}
