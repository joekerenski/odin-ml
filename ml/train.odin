package ml

// ============================================================================
// train — arena-backed epoch helper for classification loops.
//
// NOTE: Odin passes `context` by value into each procedure. Allocator switches
// MUST happen in the same procedure that allocates (or the caller). Helpers
// that set context.allocator and return do NOT affect the caller.
//
//   tr: Trainer
//   trainer_init(&tr, params[:], lr=0.05, momentum=0.9, batch_size=128)
//   defer trainer_destroy(&tr)
//
//   for epoch in 0..<epochs {
//       loss := trainer_epoch_ce(&tr, X, Y, forward)
//       acc  := eval_accuracy(forward, Xte, Yte, 256)
//   }
//
// Manual step (allocator switch stays in YOUR proc):
//
//   old := context.allocator
//   context.allocator = trainer_allocator(&tr)
//   loss := cross_entropy(forward(Xb), Yb)
//   v := trainer_backward_step(&tr, loss)
//   context.allocator = old
//   trainer_reclaim(&tr)
// ============================================================================

import "core:mem"

Trainer :: struct {
	params:     []^Tensor,
	opt:        ^SGD,
	arena:      mem.Dynamic_Arena,
	batch_size: int,
}

trainer_init :: proc(
	tr: ^Trainer,
	params: []^Tensor,
	lr: f32 = 0.05,
	momentum: f32 = 0.9,
	batch_size: int = 128,
	arena_block: int = 8 * mem.Megabyte,
) {
	tr.params = params
	tr.opt = new_sgd_list(lr, momentum, params)
	tr.batch_size = batch_size
	mem.dynamic_arena_init(&tr.arena, block_size = arena_block)
}

trainer_destroy :: proc(tr: ^Trainer) {
	mem.dynamic_arena_destroy(&tr.arena)
}

trainer_allocator :: proc(tr: ^Trainer) -> mem.Allocator {
	return mem.dynamic_arena_allocator(&tr.arena)
}

// clear → backward → sgd. Call with arena allocator already active.
trainer_backward_step :: proc(tr: ^Trainer, loss: ^Tensor) -> f32 {
	clear_grad_list(tr.params)
	backward(loss)
	v := item(loss)
	sgd_step(tr.opt)
	return v
}

// Reset arena (keep warm blocks) + nil dangling param grads.
// Call after restoring the heap allocator.
trainer_reclaim :: proc(tr: ^Trainer) {
	mem.dynamic_arena_reset(&tr.arena)
	clear_grad_list(tr.params)
}

// One epoch of CE classification. Shuffles each call. Returns mean batch loss.
// Allocator switch lives HERE (same proc as minibatch/forward) so context sticks.
trainer_epoch_ce :: proc(
	tr: ^Trainer,
	X: ^Tensor,
	Y: []u8,
	forward: proc(x: ^Tensor) -> ^Tensor,
) -> f32 {
	n := int(X.shape[0])
	bs := tr.batch_size
	assert(bs > 0)
	n_batches := n / bs
	if n_batches == 0 do return 0

	perm := random_permutation(n)
	defer delete(perm)

	total: f32 = 0
	for b in 0..<n_batches {
		old := context.allocator
		context.allocator = mem.dynamic_arena_allocator(&tr.arena)

		Xb, Yb := minibatch(X, Y, b, bs, perm)
		loss := cross_entropy(forward(Xb), Yb)
		total += trainer_backward_step(tr, loss)

		context.allocator = old
		trainer_reclaim(tr)
	}
	return total / f32(n_batches)
}
