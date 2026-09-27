package ml

import "core:math"

// ============================================================================
// Optimizers: SGD (+momentum), Adam/AdamW, cosine LR schedule.
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

// ============================================================================
// Adam / AdamW (tinygrad semantics: decoupled weight decay added to the update)
//
//   m = b1*m + (1-b1)*g         m_hat = m / (1 - b1^t)
//   v = b2*v + (1-b2)*g²        v_hat = v / (1 - b2^t)
//   w -= lr * ( m_hat / (sqrt(v_hat) + eps) + wd*w )
// ============================================================================

Adam :: struct {
	lr, b1, b2, eps, weight_decay: f32,
	t:                             int,
	params:                        [dynamic]^Tensor,
	m, v:                          [dynamic][]f32,
}

new_adam :: proc(
	params: []^Tensor,
	lr: f32 = 1e-3,
	b1: f32 = 0.9,
	b2: f32 = 0.999,
	eps: f32 = 1e-8,
	weight_decay: f32 = 0,
) -> ^Adam {
	opt := new(Adam)
	opt^ = Adam{lr = lr, b1 = b1, b2 = b2, eps = eps, weight_decay = weight_decay}
	for p in params {
		append(&opt.params, p)
		append(&opt.m, make([]f32, len(p.data)))
		append(&opt.v, make([]f32, len(p.data)))
	}
	return opt
}

adam_step :: proc(opt: ^Adam) {
	opt.t += 1
	c1 := 1 - math.pow(opt.b1, f32(opt.t))
	c2 := 1 - math.pow(opt.b2, f32(opt.t))
	for p, i in opt.params {
		if p.grad == nil do continue
		m, v, g := opt.m[i], opt.v[i], p.grad.data
		for j in 0 ..< len(p.data) {
			m[j] = opt.b1 * m[j] + (1 - opt.b1) * g[j]
			v[j] = opt.b2 * v[j] + (1 - opt.b2) * g[j] * g[j]
			up := (m[j] / c1) / (math.sqrt(v[j] / c2) + opt.eps) + opt.weight_decay * p.data[j]
			p.data[j] -= opt.lr * up
		}
	}
}

// ---- any optimizer --------------------------------------------------------

Optimizer :: union {
	^SGD,
	^Adam,
}

optimizer_step :: proc(opt: Optimizer) {
	switch o in opt {
	case ^SGD: sgd_step(o)
	case ^Adam: adam_step(o)
	}
}

optimizer_set_lr :: proc(opt: Optimizer, lr: f32) {
	switch o in opt {
	case ^SGD: o.lr = lr
	case ^Adam: o.lr = lr
	}
}

// ---- schedules ------------------------------------------------------------

// Linear warmup from 0 to base over `warmup` steps, then cosine to min_lr at
// `total`. step counts from 0.
cosine_lr :: proc(step, total, warmup: int, base: f32, min_lr: f32 = 0) -> f32 {
	if step < warmup do return base * f32(step + 1) / f32(warmup)
	if step >= total do return min_lr
	p := f32(step - warmup) / f32(max(1, total - warmup))
	return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.PI * p))
}
