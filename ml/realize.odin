package ml

// ============================================================================
// realize — schedule and run the UOp graph. The only executor.
//
// realize_all(sinks):
//   1. topo-sort unrealized nodes (realized nodes are buffers: stop there),
//   2. views: Permutes and Reshapes become strides over a base buffer, read in
//      place by their consumers (plan_views),
//   3. fuse: same-shape elementwise nodes into groups, reductions with their
//      producers and consumers (fuse_groups) → one kernel each,
//   4. run in topo order on the current backend (device.odin): fused groups,
//      views, primitive kernels; then sync so results are visible on the host.
//
// Buffer reuse: once every reader of an internal (backward-built, see uop.odin)
// buffer has run, it goes to a pool and the next output of the same size
// takes it — memory still hot in cache instead of fresh arena memory, which
// on x86 costs a DRAM read before the write. Views read in place (and aliases)
// share their base's buffer and count as its readers' readers.
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
import "core:slice"
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
	views:       map[^UOp]View, // Permutes / Reshapes read in place (no buffer)
	alias:       map[^UOp]^UOp, // view nodes that are their base's buffer as is
	store_into:  map[^UOp]Store_Into, // MatMuls writing straight into a view node's buffer
	epilogue:    map[^UOp]^UOp, // group root → the MatMul it is the epilogue of
	epi_gemm:    map[^UOp]^UOp, // that MatMul → its epilogue group's root
	uf:          map[^UOp]^UOp, // fusion groups (union-find over ewise nodes)
	order:       [dynamic]^UOp, // what the run loop visits: views, nodes, groups' last members
	clones:      [dynamic]^UOp, // nodes rematerialization added (remat.odin)
	clone_of:    [dynamic]^UOp, // the node each clone copies
	rewired:     [dynamic]Rewire, // srcs it pointed at clones, restored afterwards
	dead:        map[^UOp]bool, // group members nothing reads after rematerialization
	// buffer reuse
	owner:       map[^UOp]^UOp, // node → node whose buffer it uses (views)
	refs:        map[^UOp]int, // owner → readers still to run
	pinned:      map[^UOp]bool, // owner whose buffer must outlive the realize
	pool:        map[int][dynamic][]f32, // free buffers by length, LIFO
	temps:       [dynamic][]f32, // kernel scratch (multi-run reductions), freed after the sync
}

// Reuse dead internal buffers within a realize (ML_REUSE=0 turns it off).
buffer_reuse := true

// Recompute cheap elementwise values in each kernel that reads them instead of
// storing them (remat.odin). Auto: on GPUs (memory-bound), off on the CPU
// (its interpreter runs exp/log scalar: recompute costs more than the traffic
// it saves). ML_REMAT=0|1 forces it.
Remat_Mode :: enum {
	Auto,
	On,
	Off,
}
remat_mode: Remat_Mode = .Auto

// Read Permutes / Reshapes in place and let GEMMs store into them (ML_VIEWS=0:
// copy every non-dense view, the reference path).
view_reads := true

@(private)
current_pool: ^map[int][dynamic][]f32

schedule_make :: proc() -> (s: Schedule) {
	s.topo = make([dynamic]^UOp, scratch())
	s.consumers = make(map[^UOp]int, scratch())
	s.sinks = make(map[^UOp]bool, scratch())
	s.views = make(map[^UOp]View, scratch())
	s.alias = make(map[^UOp]^UOp, scratch())
	s.store_into = make(map[^UOp]Store_Into, scratch())
	s.epilogue = make(map[^UOp]^UOp, scratch())
	s.epi_gemm = make(map[^UOp]^UOp, scratch())
	s.uf = make(map[^UOp]^UOp, scratch())
	s.clones = make([dynamic]^UOp, scratch())
	s.order = make([dynamic]^UOp, scratch())
	s.clone_of = make([dynamic]^UOp, scratch())
	s.rewired = make([dynamic]Rewire, scratch())
	s.dead = make(map[^UOp]bool, scratch())
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
	delete(s.views)
	delete(s.alias)
	delete(s.store_into)
	delete(s.epilogue)
	delete(s.epi_gemm)
	delete(s.uf)
	delete(s.order)
	free_clones(s)
	delete(s.dead)
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
	base: ^UOp
	if v, ok := s.views[u]; ok do base = v.base
	if b, ok := s.alias[u]; ok do base = b
	if base != nil {
		if base_o, ok := s.owner[base]; ok do o = base_o
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
schedule_visit :: proc(s: ^Schedule, u: ^UOp) {
	if u.epoch == sched_epoch do return
	u.epoch = sched_epoch
	if u.data != nil {
		u.pos = LEAF_UNNUMBERED
		return
	}
	for x in u.src do schedule_visit(s, x)
	u.pos = i32(len(s.topo))
	append(&s.topo, u)
}

// Bumped per realize: UOp.epoch == sched_epoch marks this realize's nodes.
@(private)
sched_epoch: u32

@(private)
LEAF_UNNUMBERED :: max(i32)

// Is x a node this realize schedules (unrealized, in s.topo)?
@(private)
in_topo :: #force_inline proc(x: ^UOp) -> bool {
	return x.epoch == sched_epoch && x.pos >= 0 && x.pos != LEAF_UNNUMBERED
}

uf_find :: proc(uf: ^map[^UOp]^UOp, x: ^UOp) -> ^UOp {
	p := uf[x]
	if p != x {
		p = uf_find(uf, p)
		uf[x] = p
	}
	return p
}

realize_all :: proc(sinks: []^UOp) {
	s := schedule_make()
	defer schedule_destroy(&s)

	sched_epoch += 1
	for u in sinks {
		schedule_visit(&s, u)
		s.sinks[u] = true
	}
	if len(s.topo) == 0 do return

	if debug_level >= 3 do print_schedule(&s)

	step_start: time.Tick
	if debug_level >= 1 do step_start = time.tick_now()
	k0 := counters.kernels

	// members of each group in topo order; the last member runs the group
	members := make(map[^UOp][dynamic]^UOp, scratch())
	defer {
		for _, m in members do delete(m)
		delete(members)
	}
	// the same structure as a recent realize: replay its decisions
	ix: Sched_Index
	key: []u8
	defer if key != nil {
		delete(key, scratch())
		sched_index_destroy(&ix)
	}
	rec: ^Sched_Record
	if sched_cache_enabled {
		key = sched_key(&s, &ix)
		rec = sched_lookup(key)
	}
	if rec != nil {
		sched_replay(&s, &ix, rec, &members)
	} else {
		// consumers (each consumer counted once per distinct src)
		for u in s.topo {
			for x, i in u.src {
				if x.data != nil || src_seen_before(u, i) do continue
				s.consumers[x] += 1
			}
		}
		plan_views(&s)
		fuse_groups(&s)
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
		if remat_mode == .On || remat_mode == .Auto && backend.device != .CPU do remat(&s, &members)
		plan_epilogues(&s, &members)
		delete(s.order)
		s.order = run_order(&s, &members, backend.device == .Metal && metal_concurrent)
		if sched_cache_enabled do sched_record(&s, &ix, key, &members)
	}
	sched_ms: f64
	if debug_level >= 1 do sched_ms = f64(time.tick_since(step_start)) / 1e6

	current_pool = &s.pool
	defer current_pool = nil
	for u in s.order {
		if u in s.epi_gemm do continue // runs with its epilogue group
		if u.op == .Permute || u.op == .Reshape {
			run_view(&s, u)
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
			if m, ok := s.epilogue[uf_find(&s.uf, u)]; ok {
				if m.data != nil do track_buffer(&s, m)
				release_srcs(&s, m)
			}
			continue
		}
		run_node(&s, u)
		if !(u in s.store_into) do track_buffer(&s, u) // else its view node owns the buffer
		release_srcs(&s, u)
	}
	backend.sync()

	if debug_level == 1 {
		ms := f64(time.tick_since(step_start)) / 1e6
		fmt.printfln("  realize: %d kernels  %.3f ms (schedule %.3f ms)", counters.kernels - k0, ms, sched_ms)
	}
}

// ---- GEMM epilogues ----------------------------------------------------------------
//
// A MatMul whose result only one elementwise group reads — directly or through
// dense views (same element order) — runs as one kernel with that group as its
// epilogue (backend.matmul_epi), at the group's turn: the result is never
// stored. Typical: the bias add after a projection, a grad added to another.
@(private)
plan_epilogues :: proc(s: ^Schedule, members: ^map[^UOp][dynamic]^UOp) {
	if backend.matmul_epi == nil do return
	reader := make(map[^UOp]^UOp, scratch()) // some consumer of each node
	defer delete(reader)
	for c in s.topo do if !(c in s.dead) do for x in c.src do reader[x] = c
	for c in s.clones do if !(c in s.dead) do for x in c.src do reader[x] = c
	for m in s.topo {
		if m.op != .MatMul || m in s.dead || m in s.sinks || m in s.store_into || s.consumers[m] != 1 do continue
		// through single-reader dense views of m
		x := m
		c := reader[m]
		for ((c.op == .Permute || c.op == .Reshape) && c in s.views && s.consumers[x] == 1) {
			v := s.views[c]
			if v.base != m || c in s.sinks || !strides_dense(c.shape, v.st) do break
			x = c
			c = reader[c]
			if c == nil do break
		}
		if c == nil || !(c in s.uf) do continue
		G := uf_find(&s.uf, c)
		if G in s.epilogue do continue
		group := members[G][:]
		ok := numel(group[len(group) - 1].shape) == numel(m.shape)
		inside := 0
		for v in group {
			if v.op == .Sum || v.op == .ReduceMax do ok = false
			for y, i in v.src do if y == x && !src_seen_before(v, i) do inside += 1
		}
		if !ok || inside != s.consumers[x] do continue
		s.epilogue[G] = m
		s.epi_gemm[m] = G
	}
}

// ---- run order ------------------------------------------------------------------
//
// The units the run loop visits — view nodes, single nodes, groups (at their
// last member) — in topo order, or by level for a concurrent GPU encoder:
// level = longest chain of kernels from the graph's inputs, so the kernels of
// one level are independent and need no barrier between them (topo order
// from a DFS puts dependent kernels next to each other).
@(private)
run_order :: proc(s: ^Schedule, members: ^map[^UOp][dynamic]^UOp, by_level: bool) -> [dynamic]^UOp {
	order := make([dynamic]^UOp, scratch())
	unit :: proc(s: ^Schedule, u: ^UOp) -> ^UOp {
		if u in s.uf do return uf_find(&s.uf, u)
		return u
	}
	for u in s.topo {
		if u in s.dead || u in s.epi_gemm do continue
		if u in s.uf {
			g := members[uf_find(&s.uf, u)]
			if g[len(g) - 1] != u do continue
		}
		append(&order, u)
	}
	if !by_level do return order
	stored := make(map[^UOp]bool, scratch()) // view nodes a GEMM writes: no kernel of their own
	defer delete(stored)
	for _, into in s.store_into do stored[into.v] = true
	level := make(map[^UOp]int, len(order), scratch())
	defer delete(level)
	Item :: struct {
		u:          ^UOp,
		level, pos: int,
	}
	items := make([]Item, len(order), scratch())
	defer delete(items, scratch())
	for t, pos in order {
		me := unit(s, t)
		group := t in s.uf ? members[me][:] : []^UOp{t}
		lv := 0
		for v in group do for x in v.src {
			if x.data != nil do continue
			ux := unit(s, x)
			if ux != me do lv = max(lv, level[ux])
		}
		if m, ok := s.epilogue[me]; ok do for x in m.src { // the GEMM runs with this group
			if x.data != nil do continue
			lv = max(lv, level[unit(s, x)])
		}
		kernel := !((t.op == .Permute || t.op == .Reshape) && (t in s.views || t in s.alias || t in stored))
		level[me] = lv + (kernel ? 1 : 0)
		items[pos] = Item{t, level[me], pos}
	}
	slice.sort_by(items, proc(a, b: Item) -> bool { return a.level < b.level || a.level == b.level && a.pos < b.pos })
	for it, i in items do order[i] = it.u
	return order
}

// ---- views ----------------------------------------------------------------------
//
// Permutes and Reshapes compose into strides over a base buffer: the nearest
// node with a buffer of its own (not a view, or a view that was copied). A
// view is read in place by its consumers (s.views, no kernel) when each can
// address it: elementwise kernels and other views always, reductions when it
// merges into [outer, r, inner], GEMMs when its batch dims merge into ≤ 2.
// Otherwise it is materialized:
//   alias       dense over its base: the base's buffer as is (no kernel)
//   store_into  a rearrangement of a MatMul's output nothing else reads: the
//               GEMM writes straight into the view's layout (no kernel)
//   copy        anything else: one strided copy kernel (a Reshape strides
//               can't express copies over its src's shape)

Store_Into :: struct {
	v:  ^UOp,          // the view node that owns the buffer
	st: [MAX_DIMS]int, // where the MatMul's elements go, over its shape
}

// Per view node: its strides over its base and, if materialized, how.
@(private)
View_Plan :: struct {
	using view: View,
	cut:        bool, // a Reshape strides can't express (copy over its src's shape)
	copied:     bool, // has its own buffer (copy or store_into)
}

@(private)
plan_views :: proc(s: ^Schedule) {
	plan := make(map[^UOp]View_Plan, scratch())
	defer delete(plan)
	users := make(map[^UOp][dynamic]^UOp, scratch())
	defer {
		for _, l in users do delete(l)
		delete(users)
	}
	for c in s.topo do for x, i in c.src {
		if (x.op != .Permute && x.op != .Reshape) || x.data != nil || src_seen_before(c, i) do continue
		l, ok := &users[x]
		if !ok {
			users[x] = make([dynamic]^UOp, scratch())
			l = &users[x]
		}
		append(l, c)
	}
	// in topo order: a view's base is its src's base unless the src has a buffer
	for u in s.topo {
		if u.op != .Permute && u.op != .Reshape do continue
		src := u.src[0]
		sv := View{src, shape_strides(src.shape)}
		if p, ok := plan[src]; ok && !p.copied do sv = p.view
		p := View_Plan{view = View{base = sv.base}}
		if u.op == .Permute {
			for o, i in u.arg.([]i32) do p.st[i] = sv.st[o]
		} else {
			st, ok := reshape_strides(src.shape, sv.st, u.shape)
			if ok do p.st = st
			else do p.cut = true
		}
		readable := view_reads && !p.cut && !(u in s.sinks)
		if readable do for c in users[u] {
			for x, i in c.src do if x == u && !view_readable(c, i, p.view) do readable = false
		}
		switch {
		case readable: s.views[u] = p.view
		case !p.cut && strides_dense(u.shape, p.st): s.alias[u] = p.base
		case: p.copied = true
		}
		plan[u] = p
	}
	if len(plan) == 0 do return

	// a MatMul reading two views: their batch dims must merge together
	for c in s.topo {
		if c.op != .MatMul || !(c.src[0] in s.views || c.src[1] in s.views) do continue
		ops: [3][MAX_DIMS]int
		for j in 0 ..< 2 do _, ops[j] = src_view(&s.views, c.src[j])
		ops[2] = shape_strides(c.shape)
		if _, _, _, ok := gemm_batch(c.shape, ops); ok do continue
		for j in 0 ..< 2 do if v, ok := s.views[c.src[j]]; ok {
			delete_key(&s.views, c.src[j])
			p := &plan[c.src[j]]
			if strides_dense(c.src[j].shape, v.st) do s.alias[c.src[j]] = v.base
			else do p.copied = true
		}
	}
	if view_reads do for u, p in plan do if p.copied do plan_store_into(s, &plan, u)
}

// Can consumer c read its src i through view v?
@(private)
view_readable :: proc(c: ^UOp, i: int, v: View) -> bool {
	x := c.src[i]
	switch {
	case c.op == .Permute || c.op == .Reshape || op_is_ewise(c.op):
		return true
	case c.op == .Sum || c.op == .ReduceMax:
		lo, hi, ok := reduce_run(x.shape, c.arg.([]i32))
		if !ok do return false
		_, mok := merge3(x.shape, broadcast_view(x.shape, v.st, len(x.shape)), lo, hi)
		return mok
	case c.op == .MatMul:
		ops: [3][MAX_DIMS]int
		for j in 0 ..< 2 do ops[j] = shape_strides(c.src[j].shape)
		ops[i] = v.st
		ops[2] = shape_strides(c.shape)
		_, _, _, ok := gemm_batch(c.shape, ops)
		return ok
	}
	return false
}

// A copied view u whose base is a MatMul output read by nothing else: the
// MatMul stores into u's layout instead. The copy reads the MatMul's dense
// output through strides; each copy dim is one or more MatMul dims (inner to
// outer), and each MatMul dim lands where its copy dim (part) goes.
@(private)
plan_store_into :: proc(s: ^Schedule, plan: ^map[^UOp]View_Plan, u: ^UOp) {
	// the copy: over u's shape, or (cut) over its src's shape through the src's view
	dims := u.shape
	src := plan[u].view
	if plan[u].cut {
		dims = u.src[0].shape
		p, ok := plan[u.src[0]]
		if !ok || p.copied do return
		src = p.view
	}
	m := src.base
	if m.op != .MatMul || m in s.sinks || m in s.store_into || s.consumers[m] != 1 do return
	for x := u.src[0]; x != m; x = x.src[0] {
		if s.consumers[x] != 1 || x in s.sinks do return // a single chain of views
	}
	mst := shape_strides(m.shape)
	ust := shape_strides(dims)
	into := Store_Into{v = u}
	used: [MAX_DIMS]bool
	for d in 0 ..< len(dims) {
		n, st, dest := int(dims[d]), src.st[d], ust[d]
		for n > 1 {
			j := -1
			for k in 0 ..< len(m.shape) do if !used[k] && m.shape[k] != 1 && mst[k] == st && n % int(m.shape[k]) == 0 do j = k
			if j < 0 do return
			used[j] = true
			into.st[j] = dest
			n /= int(m.shape[j])
			st *= int(m.shape[j])
			dest *= int(m.shape[j])
		}
	}
	for k in 0 ..< len(m.shape) do if m.shape[k] != 1 && !used[k] do return
	ops: [3][MAX_DIMS]int
	for j in 0 ..< 2 do _, ops[j] = src_view(&s.views, m.src[j])
	ops[2] = into.st
	if _, _, _, ok := gemm_batch(m.shape, ops); !ok do return
	s.store_into[m] = into
}

// A view node's turn: read in place (nothing to do), an alias, already written
// by its MatMul, or a copy.
@(private)
run_view :: proc(s: ^Schedule, u: ^UOp) {
	if u in s.views || u.data != nil do return
	if b, ok := s.alias[u]; ok {
		u.data = b.data
		return
	}
	// copy: over u's shape through its view, or (a cut reshape) over its src's
	dims := u.shape
	data, st := src_view(&s.views, u.src[0])
	if u.op == .Permute {
		pst := st
		for o, i in u.arg.([]i32) do st[i] = pst[o]
	} else {
		dims = u.src[0].shape
	}
	t0: time.Tick
	if profiling || debug_level >= 2 do t0 = time.tick_now()
	alloc_out(u)
	profile_open(.Permute, 2 * i64(numel(u.shape)) * 4, 0)
	d: [MAX_DIMS]int
	for x, i in dims do d[i] = int(x)
	k := kernel_copy(u.data, data, d[:len(dims)], st[:len(dims)])
	launch_kernel(&k)
	profile_host_wall(t0, .Permute)
	counters.kernels += 1
	if debug_level >= 2 {
		dt := i64(time.tick_since(t0))
		counters.time_ns += dt
		fmt.printfln("  kernel  copy %-11v shape=%v  %7.3f ms", u.op, u.shape, f64(dt) / 1e6)
	}
}

// Batch dims of a MatMul (shape [batch..., M, N]) merged across the operands'
// strides (A, B, C over their own shapes): ≤ 2 left, else false.
gemm_batch :: proc(shape: []i32, ops: [3][MAX_DIMS]int) -> (Z: [2]int, bs: [3][2]int, n: int, ok: bool) {
	nb := len(shape) - 2
	size: [MAX_DIMS]int
	st: [MAX_DIMS][3]int
	for d in 0 ..< nb {
		if shape[d] == 1 do continue
		if n > 0 {
			merge := true
			for j in 0 ..< 3 do if st[n - 1][j] != ops[j][d] * int(shape[d]) do merge = false
			if merge {
				size[n - 1] *= int(shape[d])
				for j in 0 ..< 3 do st[n - 1][j] = ops[j][d]
				continue
			}
		}
		size[n] = int(shape[d])
		for j in 0 ..< 3 do st[n][j] = ops[j][d]
		n += 1
	}
	if n > 2 do return
	Z = {1, 1}
	for d in 0 ..< n {
		e := d + 2 - n // one dim: z1; two: z0, z1
		Z[e] = size[d]
		for j in 0 ..< 3 do bs[j][e] = st[d][j]
	}
	return Z, bs, n, true
}

// The GEMM of MatMul u: operands through their views, C into u's buffer or
// the view it stores into.
@(private)
gemm_from_node :: proc(s: ^Schedule, u: ^UOp) -> Gemm {
	ops: [3][MAX_DIMS]int
	data: [3][]f32
	for j in 0 ..< 2 do data[j], ops[j] = src_view(&s.views, u.src[j])
	if into, ok := s.store_into[u]; ok {
		data[2], ops[2] = into.v.data, into.st
	} else {
		data[2], ops[2] = u.data, shape_strides(u.shape)
	}
	Z, bs, _, ok := gemm_batch(u.shape, ops)
	assert(ok, "gemm_from_node: batch dims don't merge (plan_views should have copied)")
	n := len(u.shape)
	g := Gemm{Z0 = Z[0], Z1 = Z[1], M = int(u.shape[n - 2]), K = int(u.src[0].shape[n - 1]), N = int(u.shape[n - 1])}
	o := [3]^Gemm_Operand{&g.a, &g.b, &g.c}
	for j in 0 ..< 3 do o[j]^ = Gemm_Operand{data[j], ops[j][n - 2], ops[j][n - 1], bs[j]}
	return g
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
Fuse :: struct {
	s:                       ^Schedule,
	idx:                     map[^UOp]int,
	start, users:            []int, // users of node i: users[start[i]:start[i+1]] (topo indices)
	parent, size, last:      []int, // union-find; per root: members, last member
	next, tail:              []int, // members of a root, linked
	red:                     []int, // per root: its reduce node, or -1
}

@(private)
fuse_find :: proc(f: ^Fuse, i: int) -> int {
	r := i
	for f.parent[r] != r do r = f.parent[r]
	for c := i; f.parent[c] != r; {
		n := f.parent[c]
		f.parent[c] = r
		c = n
	}
	return r
}

// Would merging groups a and b keep "run at the last member" valid (every
// outside user of a member after the merged group's last member)?
@(private)
fuse_valid :: proc(f: ^Fuse, a, b: int) -> bool {
	lst := max(f.last[a], f.last[b])
	for r in ([]int{a, b}) {
		for m := r; m >= 0; m = f.next[m] {
			for c in f.users[f.start[m]:f.start[m + 1]] {
				rc := f.parent[c] < 0 ? -1 : fuse_find(f, c)
				if rc != a && rc != b && c < lst do return false
			}
		}
	}
	return true
}

// Distinct outside inputs (buffers and constants) of the members of a ∪ b with
// the given shape (nil: any), plus each in `extra` not already counted.
@(private)
fuse_inputs :: proc(f: ^Fuse, a, b: int, shape: []i32, seen: ^[dynamic]^UOp) -> int {
	for r in ([]int{a, b}) {
		for m := r; m >= 0; m = f.next[m] {
			u := f.s.topo[m]
			if shape != nil && !shapes_equal(u.shape, shape) do continue
			for y in u.src {
				if yi, inside := f.idx[y]; inside && f.parent[yi] >= 0 {
					ry := fuse_find(f, yi)
					if ry == a || ry == b do continue
				}
				dup := false
				for z in seen do if z == y do dup = true
				if !dup do append(seen, y)
			}
		}
	}
	return len(seen)
}

// Buffers a kernel for a ∪ b binds: its input buffers (in seen, from
// fuse_inputs) plus at most one store per member read outside a ∪ b.
@(private)
fuse_bufs :: proc(f: ^Fuse, a, b: int, seen: []^UOp) -> (n: int) {
	for y in seen do if y.op != .Const do n += 1
	for r in ([]int{a, b}) {
		for m := r; m >= 0; m = f.next[m] {
			if f.s.topo[m] in f.s.sinks {
				n += 1
				continue
			}
			for c in f.users[f.start[m]:f.start[m + 1]] {
				rc := f.parent[c] < 0 ? -1 : fuse_find(f, c)
				if rc != a && rc != b {
					n += 1
					break
				}
			}
		}
	}
	return
}

@(private)
fuse_union :: proc(f: ^Fuse, a, b: int) {
	f.parent[a] = b
	f.size[b] += f.size[a]
	f.last[b] = max(f.last[a], f.last[b])
	f.next[f.tail[b]] = a
	f.tail[b] = f.tail[a]
	if f.red[a] >= 0 do f.red[b] = f.red[a]
}

// A reduction absorbing group a (its prologue or epilogue) into its group b:
// inputs addressable in the canonical space, per-phase input budget, and no
// input-shaped member reading the reduce or an output-shaped member.
@(private)
fuse_reduce_ok :: proc(f: ^Fuse, a, b: int) -> bool {
	r := f.s.topo[f.red[b]]
	X := r.src[0].shape
	seen := make([dynamic]^UOp, scratch())
	defer delete(seen)
	n_pro := fuse_inputs(f, a, b, X, &seen)
	views := &f.s.views
	for y in seen do if y.op != .Const && !reduce_mergeable(r, y, views) do return false
	for y in r.src do if _, inside := f.idx[y]; !inside || f.parent[f.idx[y]] < 0 || (fuse_find(f, f.idx[y]) != a && fuse_find(f, f.idx[y]) != b) {
		if y.op != .Const && !reduce_mergeable(r, y, views) do return false
		n_pro += 1
	}
	clear(&seen)
	n_epi := fuse_inputs(f, a, b, r.shape, &seen)
	for y in seen do if y != r && y.op != .Const && !reduce_mergeable(r, y, views) do return false
	if n_pro + n_epi > MAX_FUSED_INPUTS do return false
	clear(&seen)
	fuse_inputs(f, a, b, nil, &seen)
	if fuse_bufs(f, a, b, seen[:]) + 1 > MAX_FUSED_BUFS do return false // + the reduce's own input
	for g in ([]int{a, b}) {
		for m := g; m >= 0; m = f.next[m] {
			u := f.s.topo[m]
			if u == r || !shapes_equal(u.shape, X) do continue
			for y in u.src {
				if y == r do return false
				yi, inside := f.idx[y]
				if !inside || f.parent[yi] < 0 do continue
				ry := fuse_find(f, yi)
				if (ry == a || ry == b) && y != r && shapes_equal(y.shape, r.shape) && !shapes_equal(y.shape, X) do return false
			}
		}
	}
	return true
}

// ---- grouping ---------------------------------------------------------------------
//
// 1. Same-shape elementwise nodes merge into groups (one kernel each). A group
//    runs at the position of its last member, so a merge is allowed only if
//    that stays valid (fuse_valid). Values with outside users are stored; the
//    rest live in registers. Merges respect the kernel budget (ops, inputs).
// 2. Each single-run reduction absorbs the group producing its input (prologue,
//    evaluated per [o, r, k]) and groups consuming its result at its shape
//    (epilogue, once per output), under the same rule (fuse_reduce_ok).
@(private)
fuse_groups :: proc(s: ^Schedule) {
	N := len(s.topo)
	f := Fuse{s = s}
	f.idx = make(map[^UOp]int, N, scratch())
	for u, i in s.topo do f.idx[u] = i
	f.start = make([]int, N + 1, scratch())
	for u in s.topo do for x, i in u.src {
		j, ok := f.idx[x]
		if !ok || src_seen_before(u, i) do continue
		f.start[j + 1] += 1
	}
	for i in 0 ..< N do f.start[i + 1] += f.start[i]
	f.users = make([]int, f.start[N], scratch())
	fill := make([]int, N, scratch())
	for u, ui in s.topo do for x, i in u.src {
		j, ok := f.idx[x]
		if !ok || src_seen_before(u, i) do continue
		f.users[f.start[j] + fill[j]] = ui
		fill[j] += 1
	}
	f.parent = make([]int, N, scratch())
	f.size = make([]int, N, scratch())
	f.last = make([]int, N, scratch())
	f.next = make([]int, N, scratch())
	f.tail = make([]int, N, scratch())
	f.red = make([]int, N, scratch())
	defer {
		delete(f.idx)
		for a in ([][]int{f.start, f.users, fill, f.parent, f.size, f.last, f.next, f.tail, f.red}) do delete(a, scratch())
	}
	for u, i in s.topo {
		f.parent[i] = op_is_ewise(u.op) ? i : -1
		f.size[i], f.last[i], f.next[i], f.tail[i], f.red[i] = 1, i, -1, i, -1
	}

	seen := make([dynamic]^UOp, scratch())
	defer delete(seen)
	for u, ui in s.topo {
		if f.parent[ui] < 0 do continue
		for x in u.src {
			xi, ok := f.idx[x]
			if !ok || f.parent[xi] < 0 || !shapes_equal(x.shape, u.shape) do continue
			a, b := fuse_find(&f, xi), fuse_find(&f, ui)
			if a == b || f.size[a] + f.size[b] > MAX_FUSED_INSNS do continue
			clear(&seen)
			if fuse_inputs(&f, a, b, nil, &seen) > MAX_FUSED_INPUTS || fuse_bufs(&f, a, b, seen[:]) > MAX_FUSED_BUFS || !fuse_valid(&f, a, b) do continue
			fuse_union(&f, a, b)
		}
	}

	for u, ui in s.topo {
		if u.op != .Sum && u.op != .ReduceMax do continue
		if _, _, ok := reduce_run(u.src[0].shape, u.arg.([]i32)); !ok do continue
		f.parent[ui], f.red[ui] = ui, ui
		// prologue: the group producing the input
		if xi, ok := f.idx[u.src[0]]; ok && f.parent[xi] >= 0 {
			a, b := fuse_find(&f, xi), fuse_find(&f, ui)
			if a != b && f.red[a] < 0 && f.size[a] + f.size[b] <= MAX_FUSED_INSNS + 1 && fuse_valid(&f, a, b) && fuse_reduce_ok(&f, a, b) {
				fuse_union(&f, a, b)
			}
		}
		// epilogue: groups consuming the result at its shape
		for c in f.users[f.start[ui]:f.start[ui + 1]] {
			if f.parent[c] < 0 || !shapes_equal(s.topo[c].shape, u.shape) do continue
			a, b := fuse_find(&f, c), fuse_find(&f, ui)
			if a != b && f.red[a] < 0 && f.size[a] + f.size[b] <= MAX_FUSED_INSNS + 1 && fuse_valid(&f, a, b) && fuse_reduce_ok(&f, a, b) {
				fuse_union(&f, a, b)
			}
		}
	}
	for u, i in s.topo do if f.parent[i] >= 0 do s.uf[u] = s.topo[fuse_find(&f, i)]
}

shapes_equal_n :: proc(a, b: ^UOp) -> bool { return numel(a.shape) == numel(b.shape) }

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
	if debug_level >= 4 {
		red := false
		for u in group do if u.op == .Sum || u.op == .ReduceMax do red = true
		mm, only := 0, 0
		for u in group do for x in u.src {
			y := x
			for (y.op == .Reshape || y.op == .Permute) && y.data == nil do y = y.src[0]
			if y.op == .MatMul && shapes_equal_n(y, u) {
				mm += 1
				if s.consumers[y] == 1 && s.consumers[x] == 1 do only += 1
			}
		}
		fmt.printfln("GROUP red=%v n=%d mm_in=%d mm_only=%d stores=%d", red, len(group), mm, only, len(stores))
	}
	assert(len(stores) > 0, "fused group has no stores")

	t0: time.Tick
	if debug_level >= 2 || profiling do t0 = time.tick_now()

	for u in group do if u.op == .Sum || u.op == .ReduceMax {
		run_reduce_group(s, group, stores[:], t0)
		return
	}
	if m, ok := s.epilogue[uf_find(&s.uf, group[0])]; ok {
		profile_open(.GEMM, node_bytes(m) + fused_bytes(group, stores[:]), node_flops(m), len(group))
		g := gemm_from_node(s, m) // C unused
		if k, kok := kernel_from_group(group, stores[:], &s.views, m); kok {
			if kernel_timing() do k.label = kernel_describe(&k)
			if backend.matmul_epi(&g, &k) {
				profile_host_wall(t0, .GEMM)
				counters.kernels += 1
				counters.fused_ops += len(group)
				return
			}
		}
		// the backend can't: the GEMM into its own buffer, then the group as usual
		if profiling do pop(&profile_stats)
		profile_open(.GEMM, node_bytes(m), node_flops(m))
		alloc_out(m)
		g = gemm_from_node(s, m)
		backend.matmul(&g)
		profile_host_wall(t0, .GEMM)
		counters.kernels += 1
	}
	profile_open(.Fused, fused_bytes(group, stores[:]), i64(numel(group[len(group) - 1].shape)) * i64(len(group)), len(group))
	if k, ok := kernel_from_group(group, stores[:], &s.views); ok {
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
			ks, sok := kernel_from_group(single, single, &s.views)
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

// A reduction with its fused prologue / epilogue: one kernel.
@(private)
run_reduce_group :: proc(s: ^Schedule, group, stores: []^UOp, t0: time.Tick) {
	flops: i64
	for u in group do flops += i64(numel(u.op == .Sum || u.op == .ReduceMax ? u.src[0].shape : u.shape))
	profile_open(.Reduce, fused_bytes(group, stores), flops, len(group))
	k, ok := kernel_from_reduce_group(group, stores, &s.views)
	assert(ok, "fused reduction over budget (fuse_groups should have refused it)")
	launch_kernel(&k)
	profile_host_wall(t0, .Reduce)
	counters.kernels += 1
	counters.fused_ops += len(group) - 1
	if debug_level >= 2 {
		fmt.printf("  kernel  reduce-group[")
		for u, i in group do fmt.printf("%s%v", i > 0 ? "," : "", u.op)
		fmt.printfln("]")
	}
}

run_node :: proc(s: ^Schedule, u: ^UOp) {
	for x in u.src do assert(x.data != nil || x in s.views, "run_node: src not realized")

	t0: time.Tick
	if debug_level >= 2 do t0 = time.tick_now()

	if into, ok := s.store_into[u]; ok do alloc_out(into.v)
	else do alloc_out(u)
	kind := profile_kind(u)
	if profiling {
		profile_open(kind, node_bytes(u), node_flops(u))
		t0 = time.tick_now()
	}
	#partial switch u.op {
	case .Sum, .ReduceMax:
		run_reduce(s, u)
	case .MatMul:
		g := gemm_from_node(s, u)
		backend.matmul(&g)
	case .Conv2d, .Conv2dBwdInput, .Conv2dBwdWeight, .MaxPool2d, .MaxPool2dBwd:
		backend.sync() // CPU-only ops: inputs must be ready on the host
		if backend.to_host != nil {
			for x in u.src do backend.to_host(x.data)
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
