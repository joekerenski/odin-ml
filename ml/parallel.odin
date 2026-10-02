package ml

// ============================================================================
// parallel_for — split [0, n) into parts and run them on all cores.
//
// Persistent workers wait for a new job generation. A job publishes
// (fn, data, n), bumps the generation, and every thread (the caller too) grabs
// parts off an atomic counter until none are left, so fast and slow cores
// balance themselves. Small jobs (fewer than 2 parts of `grain`) run inline:
// no wakeup cost, and so does a parallel_for called from inside another one
// (one pool, no nesting).
//
// Waiting: a step is ~1300 kernels back to back, so workers spin (with
// pause) for a while after a job before sleeping on a futex. Waking sleeping
// threads costs ~13 µs per job on Linux; a spinning worker picks a job up in
// well under a microsecond. Between steps (host work) they go back to sleep.
//
// Kernels passed here must not use context.allocator (it may be a caller's
// arena, which is not thread-safe); scratch() is fine.
// ============================================================================

import "base:intrinsics"
import "core:os"
import "core:sync"
import "core:thread"

Range_Proc :: proc(data: rawptr, lo, hi: int)

@(private)
Job :: struct {
	fn:      Range_Proc,
	data:    rawptr,
	n, part: int, // total size, size of one part
	parts:   int,
}

// Each atomic on its own cache line: workers spin on gen while others bump
// claim and done.
Pool :: struct {
	workers:  []^thread.Thread,
	job:      Job, // written before gen is bumped; stable until done == parts
	_:        [64]u8,
	gen:      sync.Futex, // atomic: job generation; workers wait for it to change
	_:        [64]u8,
	claim:    u64, // atomic: gen << 32 | parts << 16 | next part to take
	_:        [64]u8,
	done:     int, // atomic: parts finished
	_:        [64]u8,
	sleepers: int, // atomic: workers blocked in futex_wait
	quit:     bool,
}

@(private)
pool: Pool
@(private)
pool_ready: bool

// pause iterations a worker spins before sleeping (~100 µs)
@(private)
SPIN :: 4000

// Threads used by parallel_for (caller included). 1 disables threading.
num_threads: int = 0

// True while this thread runs part of a parallel_for (worker or caller).
@(thread_local)
in_parallel_for: bool

// Threads parallel_for spreads work over (starts the pool).
thread_count :: proc() -> int {
	if !pool_ready do pool_init()
	return num_threads
}

// Take parts of job g until none are left. A part is claimed by CAS on one
// word holding (generation, part count, next index), so the decision never
// reads the job itself: a worker still holding an older generation never
// runs anything, and once a part is claimed the job can't change until it's
// done (the publisher waits for every part).
@(private)
pool_run_parts :: proc(p: ^Pool, g: u32) {
	for {
		v := intrinsics.atomic_load(&p.claim)
		i, parts := int(v & 0xffff), int((v >> 16) & 0xffff)
		if u32(v >> 32) != g || i >= parts do return
		if _, ok := intrinsics.atomic_compare_exchange_weak(&p.claim, v, v + 1); !ok do continue
		lo := i * p.job.part
		p.job.fn(p.job.data, lo, min(lo + p.job.part, p.job.n))
		intrinsics.atomic_add(&p.done, 1)
	}
}

@(private)
pool_worker :: proc(t: ^thread.Thread) {
	p := (^Pool)(t.data)
	in_parallel_for = true
	flush_denormals()
	seen := u32(0)
	for {
		for spins := 0; u32(sync.atomic_load(&p.gen)) == seen; spins += 1 {
			if spins < SPIN {
				intrinsics.cpu_relax()
				continue
			}
			// The publisher bumps gen, then checks sleepers; we count ourselves
			// first, and futex_wait returns at once if gen moved meanwhile.
			intrinsics.atomic_add(&p.sleepers, 1)
			sync.futex_wait(&p.gen, seen)
			intrinsics.atomic_sub(&p.sleepers, 1)
			spins = 0
		}
		if intrinsics.atomic_load(&p.quit) do return
		seen = u32(sync.atomic_load(&p.gen))
		pool_run_parts(p, seen)
	}
}

@(private)
pool_init :: proc() {
	pool_ready = true
	flush_denormals() // the caller's thread; workers set their own
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
	parts := min(n / max(grain, 1), 4 * (helpers + 1), 0xffff)
	if helpers == 0 || parts < 2 || in_parallel_for {
		fn(data, 0, n)
		return
	}
	part := (n + parts - 1) / parts
	pool.job = Job{fn, data, n, part, (n + part - 1) / part}
	intrinsics.atomic_store(&pool.done, 0)
	g := u32(sync.atomic_load(&pool.gen)) + 1
	intrinsics.atomic_store(&pool.claim, u64(g) << 32 | u64(pool.job.parts) << 16)
	sync.atomic_store(&pool.gen, sync.Futex(g))
	if intrinsics.atomic_load(&pool.sleepers) > 0 do sync.futex_broadcast(&pool.gen)
	in_parallel_for = true
	pool_run_parts(&pool, g)
	in_parallel_for = false
	// done when every part is: a worker that's slow to wake (or descheduled
	// before it claimed anything) doesn't hold the job up
	for intrinsics.atomic_load(&pool.done) < pool.job.parts do intrinsics.cpu_relax()
}
