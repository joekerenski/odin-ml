#+build !linux
package ml

// Not Linux: no CUDA backend. set_device(.CUDA) returns false.

cuda_backend :: proc() -> (Backend, bool) {
	return {}, false
}

cuda_device_name :: proc() -> string {
	return ""
}
