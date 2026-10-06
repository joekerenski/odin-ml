package ml

// ============================================================================
// realize — schedule and run the UOp graph. The only executor.
//
// realize_all(sinks):
//   1. topo-sort unrealized nodes (realized nodes are buffers: stop there),
//   2. fuse: an ewise node joins its ewise src when shapes match and the src
//      has exactly one consumer → one loop, intermediates never stored,
//   3. fold: a last-two-axes Permute feeding only MatMuls → GEMM transpose flag,
//   4. run in topo order on the current backend (device.odin): fused groups,
//      views, primitive kernels; then sync so results are visible on the host.
//
// Buffer reuse: once every reader of an internal (backward-built, see uop.odin)
// buffer has run, it goes to a pool and the next output of the same size
// takes it — memory still hot in cache instead of fresh arena memory, which
// on x86 costs a DRAM read before the write. Views (Reshape, folded
// transposes) share their source's buffer and count as its readers' readers.
// Kernels run in order on every backend, so a buffer is never written while
// an earlier kernel could still read it.
//
// A node's .data is kept only if something outside its fused group (or the
// caller) needs it. Anything elided is simply recomputed if asked for later.
//
// debug_level: 1 step summary, 2 one line per kernel, 3 print graph first.
// ============================================================================

import "base:runtime"
import "core:fmt"
import "core:time"

realize :: proc(t: ^Tensor) -> ^Tensor {
	if t.data == nil do realize_all({t})
	return t
}

item :: proc(t: ^Tensor) -> f32 {
	realize(t)
	return t.data[0]
}

Schedule :: struct {
	topo:        [dynamic]^UOp,
	consumers:   map[^UOp]int, // distinct unrealized consumers
	sinks:       map[^UOp]bool,
	folded:      map[^UOp]bool, // Permutes absorbed into MatMul
	uf:          map[^UOp]^UOp, // fusion groups (union-find over ewise nodes)
	// buffer reuse
	owner:       map[^UOp]^UOp, // node → node whose buffer it uses (views)
	refs:        map[^UOp]int, // owner → readers still to run
	pinned:      map[^UOp]bool, // owner whose buffer must outlive the realize
	pool:        map[int][dynamic][]f32, // free buffers by length, LIFO
	temps:       [dynamic][]f32, // kernel scratch (multi-run reductions), freed after the sync
}

// Reuse dead internal buffers within a realize (ML_REUSE=0 turns it off).
buffer_reuse := true

@(private)
current_pool: ^map[int][dynamic][]f32

schedule_make :: proc() -> (s: Schedule) {
	s.topo = make([dynamic]^UOp, scratch())
	s.consumers = make(map[^UOp]int, scratch())
	s.sinks = make(map[^UOp]bool, scratch())
	s.folded = make(map[^UOp]bool, scratch())
	s.uf = make(map[^UOp]^UOp, scratch())
	s.owner = make(map[^UOp]^UOp, scratch())
	s.refs = make(map[^UOp]int, scratch())
	s.pinned = make(map[^UOp]bool, scratch())
	s.pool = make(map[int][dynamic][]f32, scratch())
	s.temps = make([dynamic][]f32, scratch())
	return
}

schedule_destroy :: proc(s: ^Schedule) {
	delete(s.topo)
	delete(s.consumers)
	delete(s.sinks)
	delete(s.folded)
	delete(s.uf)
	delete(s.owner)
	delete(s.refs)
	delete(s.pinned)
	for _, l in s.pool do delete(l)
	delete(s.pool)
	for t in s.temps do runtime.mem_free(raw_data(t), backend.allocator())
	delete(s.temps)
}

// u just got a buffer (its own, or a view of its src's): count its readers
// against the owning buffer.
@(private)
track_buffer :: proc(s: ^Schedule, u: ^UOp) {
	o := u
	if (u.op == .Reshape || u in s.folded) {
		if src_o, ok := s.owner[u.src[0]]; ok do o = src_o
		else do return // a view of something realized before: not ours
	}
	s.owner[u] = o
	s.refs[o] += s.consumers[u]
	if !u.internal || u in s.sinks do s.pinned[o] = true
}

// A reader of x ran.
@(private)
release_read :: proc(s: ^Schedule, x: ^UOp) {
	o, ok := s.owner[x]
	if !ok do return
	s.refs[o] -= 1
	if s.refs[o] > 0 || o in s.pinned || !buffer_reuse do return
	if o.data == nil do return
	l, has := &s.pool[len(o.data)]
	if !has {
		s.pool[len(o.data)] = make([dynamic][]f32, scratch())
		l = &s.pool[len(o.data)]
	}
	append(l, o.data)
	o.data = nil
}

// Every distinct src of u has been read.
@(private)
release_srcs :: proc(s: ^Schedule, u: ^UOp) {
	for x, i in u.src {
		dup := false
		for j in 0 ..< i do if u.src[j] == x do dup = true
		if !dup do release_read(s, x)
	}
}

// Post-order over unrealized nodes; realized nodes are leaves of the schedule.
schedule_visit :: proc(s: ^Schedule, u: ^UOp, visited: ^map[^UOp]bool) {
	if u in visited^ do return
	visited^[u] = true
	if u.data != nil do return
	for x in u.src do schedule_visit(s, x, visited)
	append(&s.topo, u)
}

uf_find :: proc(uf: ^map[^UOp]^UOp, x: ^UOp) -> ^UOp {
	p := uf[x]
	if p != x {
		p = uf_find(uf, p)
		uf[x] = p
	}
	return p
}

// Permute that only swaps the last two axes (a batched matrix transpose).
is_mt :: proc(u: ^UOp) -> bool {
	if u.op != .Permute do return false
	o := u.arg.([]i32)
	n := len(o)
	for i in 0 ..< n - 2 do if o[i] != i32(i) do return false
	return o[n - 2] == i32(n - 1) && o[n - 1] == i32(n - 2)
}

realize_all :: proc(sinks: []^UOp) {
	s := schedule_make()
	defer schedule_destroy(&s)

	visited := make(map[^UOp]bool, scratch())
	defer delete(visited)
	for u in sinks {
		schedule_visit(&s, u, &visited)
		s.sinks[u] = true
	}
	if len(s.topo) == 0 do return

	if debug_level >= 3 do print_schedule(&s)

	step_start: time.Tick
	if debug_level >= 1 do step_start = time.tick_now()
	k0 := counters.kernels

	// consumers (each consumer counted once per distinct src)
	matmul_uses := make(map[^UOp]int, scratch())
	defer delete(matmul_uses)
	for u in s.topo {
		for x, i in u.src {
			if x.data != nil do continue
			dup := false
			for j in 0 ..< i do if u.src[j] == x do dup = true
			if dup do continue
			s.consumers[x] += 1
			if u.op == .MatMul do matmul_uses[x] += 1
		}
	}

	// fold transposes into GEMM
	for u in s.topo {
		if is_mt(u) && !(u in s.sinks) && matmul_uses[u] == s.consumers[u] {
			s.folded[u] = true
		}
	}

	fuse_groups(&s)

	// members of each group in topo order, gathered in one pass; the last
	// member runs the group
	members := make(map[^UOp][dynamic]^UOp, scratch())
	defer {
		for _, m in members do delete(m)
		delete(members)
	}
	for u in s.topo {
		if !(u in s.uf) do continue
		root := uf_find(&s.uf, u)
		m, ok := &members[root]
		if !ok {
			members[root] = make([dynamic]^UOp, scratch())
			m = &members[root]
		}
		append(m, u)
	}

	current_pool = &s.pool
	defer current_pool = nil
	for u in s.topo {
		if u in s.folded {
			// a transpose read in place by its MatMuls: a view of its src
			track_buffer(&s, u)
			release_srcs(&s, u)
			continue
		}
		if u in s.uf {
			group := members[uf_find(&s.uf, u)][:]
			if group[len(group) - 1] != u do continue
			run_group(&s, group)
			for v in group do if v.data != nil do track_buffer(&s, v)
			for v in group do release_srcs(&s, v)
			continue
		}
		run_node(&s, u)
		track_buffer(&s, u)
		release_srcs(&s, u)
	}
	backend.sync()

	if debug_level == 1 {
		ms := f64(time.tick_since(step_start)) / 1e6
		fmt.printfln("  realize: %d kernels  %.3f ms", counters.kernels - k0, ms)
	}
}

// ---- fusion groups --------------------------------------------------------------
//
// Same-shape elementwise nodes are merged into groups (one kernel each). A group
// runs at the position of its last member, so a merge is allowed only if that
// stays valid: every outside user of every member must come after the merged
// group's last member (then inputs are ready when it runs, outputs before they
// are read, and no cycle can form). Values with outside users are stored; the
// rest live in registers. Merges also respect the kernel budget (ops, inputs).
@(private)
fuse_groups :: proc(s: ^Schedule) {
	N := len(s.topo)
	idx := make(map[^UOp]int, N, scratch())
	defer delete(idx)
	for u, i in s.topo do idx[u] = i
	// users of each node (each user once), by topo index
	start := make([]int, N + 1, scratch())
	defer delete(start, scratch())
	for u in s.topo do for x, i in u.src {
		j, ok := idx[x]
		if !ok || src_seen_before(u, i) do continue
		start[j + 1] += 1
	}
	for i in 0 ..< N do start[i + 1] += start[i]
	users := make([]int, start[N], scratch())
	defer delete(users, scratch())
	fill := make([]int, N, scratch())
	defer delete(fill, scratch())
	for u, ui in s.topo do for x, i in u.src {
		j, ok := idx[x]
		if !ok || src_seen_before(u, i) do continue
		users[start[j] + fill[j]] = ui
		fill[j] += 1
	}

	// union-find over topo indices; per root: size, last member, members (linked)
	parent := make([]int, N, scratch())
	size := make([]int, N, scratch())
	last := make([]int, N, scratch())
	next := make([]int, N, scratch())
	tail := make([]int, N, scratch())
	defer {
		delete(parent, scratch()); delete(size, scratch()); delete(last, scratch())
		delete(next, scratch()); delete(tail, scratch())
	}
	for u, i in s.topo {
		parent[i] = op_is_ewise(u.op) ? i : -1
		size[i], last[i], next[i], tail[i] = 1, i, -1, i
	}
	find :: proc(parent: []int, i: int) -> int {
		r := i
		for parent[r] != r do r = parent[r]
		for c := i; parent[c] != r; {
			n := parent[c]
			parent[c] = r
			c = n
		}
		return r
	}
	inputs: [2 * MAX_FUSED_INSNS * 2]^UOp
	for u, ui in s.topo {
		if parent[ui] < 0 do continue
		for x in u.src {
			xi, ok := idx[x]
			if !ok || parent[xi] < 0 || !shapes_equal(x.shape, u.shape) do continue
			a, b := find(parent, xi), find(parent, ui)
			if a == b || size[a] + size[b] > MAX_FUSED_INSNS do continue
			lst := max(last[a], last[b])
			ok_merge := true
			n_inputs := 0
			outer: for r in ([]int{a, b}) {
				for m := r; m >= 0; m = next[m] {
					for c in users[start[m]:start[m + 1]] {
						rc := parent[c] < 0 ? -1 : find(parent, c)
						if rc != a && rc != b && c < lst {
							ok_merge = false
							break outer
						}
					}
					for y in s.topo[m].src { // the merged kernel's inputs
						yi, inside := idx[y]
						if inside && parent[yi] >= 0 {
							ry := find(parent, yi)
							if ry == a || ry == b do continue
						}
						dup := false
						for z in inputs[:n_inputs] do if z == y do dup = true
						if dup do continue
						if n_inputs == MAX_FUSED_INPUTS {
							ok_merge = false
							break outer
						}
						inputs[n_inputs] = y
						n_inputs += 1
					}
				}
			}
			if !ok_merge do continue
			parent[a] = b
			size[b] += size[a]
			last[b] = lst
			next[tail[b]] = a
			tail[b] = tail[a]
		}
	}
	for u, i in s.topo do if parent[i] >= 0 do s.uf[u] = s.topo[find(parent, i)]
}

@(private)
src_seen_before :: proc(u: ^UOp, i: int) -> bool {
	for j in 0 ..< i do if u.src[j] == u.src[i] do return true
	return false
}

// Does anything outside `group` (or the caller) read u?
needs_store :: proc(s: ^Schedule, u: ^UOp, group: []^UOp) -> bool {
	if u in s.sinks do return true
	inside := 0
	for v in group {
		for x, i in v.src {
			if x != u do continue
			dup := false
			for j in 0 ..< i do if v.src[j] == x do dup = true
			if !dup do inside += 1
		}
	}
	return s.consumers[u] > inside
}

// Kernel outputs: every kernel writes its whole output, so skip zeroing.
// 64-byte aligned (cache line, SIMD-friendly).
alloc_out :: proc(u: ^UOp) {
	n := int(numel(u.shape))
	if current_pool != nil {
		if l, ok := &current_pool[n]; ok && len(l) > 0 {
			u.data = pop(l)
			return
		}
	}
	bytes, err := runtime.mem_alloc_non_zeroed(n * size_of(f32), 64, context.allocator)
	assert(err == nil, "alloc_out: out of memory")
	u.data = ([^]f32)(raw_data(bytes))[:n]
	counters.bytes_alloc += i64(n) * size_of(f32)
}

run_group :: proc(s: ^Schedule, group: []^UOp) {
	stores := make([dynamic]^UOp, scratch())
	defer delete(stores)
	for u in group do if needs_store(s, u, group) do append(&stores, u)
	assert(len(stores) > 0, "fused group has no stores")

	t0: time.Tick
	if debug_level >= 2 || profiling do t0 = time.tick_now()

	profile_open(.Fused, fused_bytes(group, stores[:]), i64(numel(group[len(group) - 1].shape)) * i64(len(group)), len(group))
	if k, ok := kernel_from_group(group, stores[:]); ok {
		launch_kernel(&k)
		profile_host_wall(t0, .Fused)
		counters.kernels += 1
		counters.fused_ops += len(group) - 1
	} else {
		// too big for one kernel: run each node on its own
		if profiling do pop(&profile_stats)
		for u in group {
			single := []^UOp{u}
			profile_open(.Fused, fused_bytes(single, single), i64(numel(u.shape)))
			tk := time.tick_now()
			ks, sok := kernel_from_group(single, single)
			assert(sok)
			launch_kernel(&ks)
			profile_host_wall(tk, .Fused)
			counters.kernels += 1
		}
	}

	if debug_level >= 2 {
		dt := i64(time.tick_since(t0))
		counters.time_ns += dt
		fmt.printf("  kernel  fused[")
		for u, i in group do fmt.printf("%s%v", i > 0 ? "," : "", u.op)
		fmt.printfln("] shape=%v  %7.3f ms", group[len(group) - 1].shape, f64(dt) / 1e6)
	}
}

// The buffer behind a src (a folded transpose reads its own src's).
gemm_operand_data :: proc(u: ^UOp) -> []f32 {
	d, _ := gemm_operand(u)
	return d
}

// Operand for GEMM: a realized buffer, or a folded transpose of one.
gemm_operand :: proc(u: ^UOp) -> (data: []f32, trans: bool) {
	if u.data == nil && is_mt(u) do return u.src[0].data, true
	return u.data, false
}

run_node :: proc(s: ^Schedule, u: ^UOp) {
	for x in u.src do assert(x.data != nil || is_mt(x), "run_node: src not realized")

	t0: time.Tick
	if debug_level >= 2 do t0 = time.tick_now()

	if u.op == .Reshape {
		u.data = u.src[0].data // view: same dense buffer
		return
	}

	alloc_out(u)
	kind := profile_kind(u)
	if profiling {
		profile_open(kind, node_bytes(u), node_flops(u))
		t0 = time.tick_now()
	}
	#partial switch u.op {
	case .Sum, .ReduceMax:
		run_reduce(s, u)
	case .Permute:
		k := kernel_from_permute(u)
		launch_kernel(&k)
	case .MatMul:
		a, ta := gemm_operand(u.src[0])
		b, tb := gemm_operand(u.src[1])
		n := len(u.shape)
		M, N, K := u.shape[n - 2], u.shape[n - 1], u.src[0].shape[n - 1]
		backend.matmul(u.data, a, b, int(numel(u.shape[:n - 2])), M, K, N, ta, tb)
	case .Conv2d, .Conv2dBwdInput, .Conv2dBwdWeight, .MaxPool2d, .MaxPool2dBwd:
		backend.sync() // CPU-only ops: inputs must be ready on the host
		if backend.to_host != nil {
			for x in u.src do backend.to_host(gemm_operand_data(x))
			backend.to_host(u.data)
		}
		run_cpu_node(u)
	case:
		fmt.panicf("run_node: no kernel for %v", u.op)
	}
	profile_host_wall(t0, kind)
	counters.kernels += 1

	if debug_level >= 2 {
		dt := i64(time.tick_since(t0))
		counters.time_ns += dt
		fmt.printfln("  kernel  %-16v shape=%v  %7.3f ms", u.op, u.shape, f64(dt) / 1e6)
	}
}

// ---- kernels -------------------------------------------------------------------

launch_kernel :: proc(k: ^Kernel) {
	if kernel_timing() do k.label = kernel_describe(k)
	backend.kernel(k)
}

// Reduce over the node's axes: each maximal run of adjacent reduced axes is one
// kernel, rightmost run first. Intermediates are realize temporaries.
@(private)
run_reduce :: proc(s: ^Schedule, u: ^UOp) {
	src := u.src[0]
	red: [MAX_DIMS]bool
	for ax in u.arg.([]i32) do red[ax] = true
	cur_shape: [MAX_DIMS]i32
	copy(cur_shape[:], src.shape)
	nd := len(src.shape)
	cur := src.data
	did := false
	d := nd - 1
	for d >= 0 {
		if !red[d] || cur_shape[d] == 1 {
			d -= 1
			continue
		}
		hi := d + 1
		for d >= 0 && red[d] do d -= 1
		lo := d + 1
		outer := int(numel(cur_shape[:lo]))
		r := int(numel(cur_shape[lo:hi]))
		inner := int(numel(cur_shape[hi:nd]))
		for k in lo ..< hi do cur_shape[k] = 1
		more := false
		for k in 0 ..< lo do if red[k] && cur_shape[k] != 1 do more = true
		dst := more ? realize_temp(s, outer * inner) : u.data
		k := kernel_reduce_run(u.op, dst, cur, outer, r, inner)
		launch_kernel(&k)
		cur = dst
		did = true
	}
	if !did { // nothing to reduce (all reduced axes have size 1): a copy
		k := kernel_copy(u.data, src.data, {len(src.data)}, {1})
		launch_kernel(&k)
	}
}

// Device-visible scratch that lives until the end of the realize (after its sync).
@(private)
realize_temp :: proc(s: ^Schedule, n: int) -> []f32 {
	bytes, err := runtime.mem_alloc_non_zeroed(n * size_of(f32), 64, backend.allocator())
	assert(err == nil, "realize_temp: out of memory")
	t := ([^]f32)(raw_data(bytes))[:n]
	append(&s.temps, t)
	return t
}

// ---- profile bookkeeping (profile.odin) -------------------------------------

// Host-run kernels report wall time; GPU kernels report device time themselves.
@(private)
profile_host_wall :: proc(t0: time.Tick, kind: Kernel_Kind) {
	if profiling && (backend.device == .CPU || kind == .Conv_Pool) {
		profile_set_wall(f64(time.tick_since(t0)) / 1e6)
	}
}

@(private)
profile_kind :: proc(u: ^UOp) -> Kernel_Kind {
	#partial switch u.op {
	case .Sum, .ReduceMax: return .Reduce
	case .Permute: return .Permute
	case .MatMul: return .GEMM
	}
	return .Conv_Pool
}

// Bytes a fused group reads (each outside input once, at its own size; constants
// are immediates) and writes (its stores).
@(private)
fused_bytes :: proc(group, stores: []^UOp) -> (b: i64) {
	seen := make([dynamic]^UOp, scratch())
	defer delete(seen)
	outer: for u in group {
		for x in u.src {
			if x.op == .Const do continue
			for v in group do if v == x do continue outer
			for v in seen do if v == x do continue outer
			append(&seen, x)
			b += i64(numel(x.shape)) * 4
		}
	}
	for u in stores do b += i64(numel(u.shape)) * 4
	return
}

@(private)
node_bytes :: proc(u: ^UOp) -> (b: i64) {
	b = i64(numel(u.shape))
	for x in u.src do b += i64(numel(x.shape))
	return 4 * b
}

@(private)
node_flops :: proc(u: ^UOp) -> i64 {
	#partial switch u.op {
	case .Sum, .ReduceMax:
		return i64(numel(u.src[0].shape))
	case .MatMul:
		return 2 * i64(numel(u.shape)) * i64(u.src[0].shape[len(u.src[0].shape) - 1])
	case .Conv2d:
		w := u.src[1].shape // [out_c, in_c, kH, kW]
		return 2 * i64(numel(u.shape)) * i64(w[1] * w[2] * w[3])
	}
	return 0
}

// Ops only the CPU implements (conv / pool and their backward).
run_cpu_node :: proc(u: ^UOp) {
	#partial switch u.op {
	case .Conv2d:
		x, w := u.src[0], u.src[1]
		win := u.arg.(Window)
		conv2d_f32(u.data, x.data, w.data, x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], win)
	case .Conv2dBwdInput:
		g, w := u.src[0], u.src[1]
		win := u.arg.(Window)
		conv2d_backward_input(u.data, g.data, w.data, u.shape[0], u.shape[1], u.shape[2], u.shape[3], w.shape[0], win)
	case .Conv2dBwdWeight:
		g, x := u.src[0], u.src[1]
		win := u.arg.(Window)
		conv2d_backward_weight(u.data, g.data, x.data, x.shape[0], x.shape[1], x.shape[2], x.shape[3], u.shape[0], win)
	case .MaxPool2d:
		x := u.src[0]
		maxpool2d_f32(u.data, x.data, x.shape[0], x.shape[1], x.shape[2], x.shape[3], u.arg.(Window))
	case .MaxPool2dBwd:
		g, x := u.src[0], u.src[1]
		maxpool2d_backward(u.data, g.data, x.data, x.shape[0], x.shape[1], x.shape[2], x.shape[3], u.arg.(Window))
	}
}
