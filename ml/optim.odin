package ml

// ============================================================================
// Optimizers.
//
// SGD with optional (heavy) momentum:
//     v = momentum * v + grad
//     w = w - lr * v
//
// The optimizer holds pointers to the parameter tensors it should update and
// one velocity buffer per parameter. step() reads each param's .grad (which
// backward() populated) and nudges the param's data in place.
//
// Velocities are allocated once (with whatever allocator is current at
// new_sgd time — typically the persistent one) and reused across steps.
// ============================================================================

SGD :: struct {
	lr:         f32,
	momentum:   f32,
	params:     [dynamic]^Tensor,
	velocities: [dynamic][]f32,
}

new_sgd :: proc(lr, momentum: f32, params: ..^Tensor) -> ^SGD {
	opt := new(SGD)
	opt.lr = lr
	opt.momentum = momentum
	for p in params {
		if p == nil do continue
		append(&opt.params, p)
		append(&opt.velocities, make([]f32, len(p.data)))
	}
	return opt
}

// Build SGD from a collected param list (from collect_params).
new_sgd_list :: proc(lr, momentum: f32, params: []^Tensor) -> ^SGD {
	return new_sgd(lr, momentum, ..params)
}

clear_grad_list :: proc(params: []^Tensor) {
	clear_grads(..params)
}

sgd_step :: proc(opt: ^SGD) {
	for i in 0..<len(opt.params) {
		p := opt.params[i]
		v := opt.velocities[i]
		if p.grad == nil do continue
		for j in 0..<len(p.data) {
			v[j] = opt.momentum * v[j] + p.grad.data[j]
			p.data[j] -= opt.lr * v[j]
		}
	}
}