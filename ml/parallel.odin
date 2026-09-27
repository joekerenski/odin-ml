package ml

// ============================================================================
// parallel_for — split [0, n) into parts and run them on all cores.
//
// Persistent workers sleep on a semaphore. A job publishes (fn, data, n),
// wakes them, and every thread (the caller too) grabs parts off an atomic
// counter until none are left, so fast and slow cores balance themselves.
// Small jobs (fewer than 2 parts of `grain`) run inline: no wakeup cost.
//
// Kernels passed here must not use context.allocator (it may be a caller's
// arena, which is not thread-safe); scratch() is fine.
// ============================================================================

import "base:intrinsics"
import "core:os"
import "core:sync"
import "core:thread"

Range_Proc :: proc(data: rawptr, lo, hi: int)

Pool :: struct {
	workers: []^thread.Thread,
	start:   sync.Sema,
	wg:      sync.Wait_Group,
	fn:      Range_Proc,
	data:    rawptr,
	n, part: int, // total size, size of one part
	parts:   int,
	next:    int, // atomic: next part to take
	quit:    bool,
}

@(private)
pool: Pool
@(private)
pool_ready: bool

// Threads used by parallel_for (caller included). 1 disables threading.
num_threads: int = 0

@(private)
pool_run_parts :: proc(p: ^Pool) {
	for {
		i := intrinsics.atomic_add(&p.next, 1)
		if i >= p.parts do return
		lo := i * p.part
		p.fn(p.data, lo, min(lo + p.part, p.n))
	}
}

@(private)
pool_worker :: proc(t: ^thread.Thread) {
	p := (^Pool)(t.data)
	for {
		sync.sema_wait(&p.start)
		if intrinsics.atomic_load(&p.quit) do return
		pool_run_parts(p)
		sync.wait_group_done(&p.wg)
	}
}

@(private)
pool_init :: proc() {
	pool_ready = true
	if num_threads == 0 do num_threads = max(1, os.get_processor_core_count())
	if num_threads == 1 do return
	pool.workers = make([]^thread.Thread, num_threads - 1, scratch())
	for &w in pool.workers {
		w = thread.create(pool_worker)
		w.data = &pool
		thread.start(w)
	}
}

// Run fn(data, lo, hi) over [0, n) in parts of at least `grain` items.
parallel_for :: proc(n, grain: int, fn: Range_Proc, data: rawptr) {
	if !pool_ready do pool_init()
	helpers := len(pool.workers)
	parts := min(n / max(grain, 1), 4 * (helpers + 1))
	if helpers == 0 || parts < 2 {
		fn(data, 0, n)
		return
	}
	pool.fn, pool.data, pool.n = fn, data, n
	pool.part = (n + parts - 1) / parts
	pool.parts = (n + pool.part - 1) / pool.part
	intrinsics.atomic_store(&pool.next, 0)
	sync.wait_group_add(&pool.wg, helpers)
	sync.sema_post(&pool.start, helpers)
	pool_run_parts(&pool)
	sync.wait_group_wait(&pool.wg)
}
