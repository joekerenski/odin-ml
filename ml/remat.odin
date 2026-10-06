package ml

// ============================================================================
// Rematerialization: recompute instead of store.
//
// After fusion, a member of one kernel that other kernels read is stored once
// and read by each. When its expression is cheap to redo — elementwise, and
// within its group reading at most one full-size buffer — each reading group
// gets its own copy of that expression (the cone: group members it depends on,
// down to the group's inputs and members stored anyway) and the value is never
// written. Typical case: an activation's intermediates, needed again by the
// backward pass, recomputed there from the pre-activation (one read) instead
// of n stores and n reads.
//
// Clones are fresh nodes owned by the schedule; group members nothing reads
// any more are dropped (s.dead).
// ============================================================================

@(private)
REMAT_MAX_CONE :: 32 // ops cloned per value

@(private)
Remat :: struct {
	s:       ^Schedule,
	members: ^map[^UOp][dynamic]^UOp,
	users:   map[^UOp][dynamic]^UOp, // distinct consumers, kept up to date
	stored:  map[^UOp]bool, // members read outside their group that stay stored
	clones:  map[^UOp]map[^UOp]^UOp, // per group root: original → clone
	n_front: map[^UOp]int, // per group root: clones at the front of its members
}

@(private)
remat :: proc(s: ^Schedule, members: ^map[^UOp][dynamic]^UOp) {
	r := Remat{s = s, members = members}
	r.users = make(map[^UOp][dynamic]^UOp, scratch())
	r.stored = make(map[^UOp]bool, scratch())
	r.clones = make(map[^UOp]map[^UOp]^UOp, scratch())
	r.n_front = make(map[^UOp]int, scratch())
	defer {
		delete(r.n_front)
		for _, l in r.users do delete(l)
		delete(r.users)
		delete(r.stored)
		for _, m in r.clones do delete(m)
		delete(r.clones)
	}
	for c in s.topo do for x, i in c.src {
		if x.data != nil || src_seen_before(c, i) do continue
		add_user(&r, x, c)
	}

	// candidates: members read outside their group only by group members that
	// can take a clone of them
	candidates := make([dynamic]^UOp, scratch())
	defer delete(candidates)
	for u in s.topo {
		if !(u in s.uf) do continue
		A := uf_find(&s.uf, u)
		outside, forced := false, u in s.sinks || !op_is_ewise(u.op)
		for c in users_of(&r, u) {
			if c in s.uf && uf_find(&s.uf, c) == A do continue
			outside = true
			if !(c in s.uf) || !(op_is_ewise(c.op) && shapes_equal(c.shape, u.shape) || (c.op == .Sum || c.op == .ReduceMax) && c.src[0] == u) {
				forced = true
			}
		}
		if !outside do continue
		if forced do r.stored[u] = true
		else do append(&candidates, u)
	}
	if len(candidates) == 0 do return

	// which stay stored: a cone too big or reading more than one full-size
	// buffer; a store cuts other cones, so repeat until nothing changes
	cone := make([dynamic]^UOp, scratch())
	leaves := make([dynamic]^UOp, scratch())
	defer {
		delete(cone)
		delete(leaves)
	}
	for changed := true; changed; {
		n := len(r.stored)
		for u in candidates {
			if u in r.stored do continue
			if !remat_cone(&r, u, &cone, &leaves) do r.stored[u] = true
		}
		changed = len(r.stored) != n
	}

	// clone into each reading group, budgets permitting
	for u in candidates {
		if u in r.stored do continue
		remat_cone(&r, u, &cone, &leaves)
		A := uf_find(&s.uf, u)
		readers := make([dynamic]^UOp, scratch()) // reading group roots
		defer delete(readers)
		for c in users_of(&r, u) {
			B := uf_find(&s.uf, c)
			if B == A do continue
			dup := false
			for x in readers do if x == B do dup = true
			if !dup do append(&readers, B)
		}
		for B in readers {
			if remat_fits(&r, B, u, cone[:], leaves[:]) do remat_inline(&r, B, u, cone[:])
			else do r.stored[u] = true // read by B as before
		}
	}

	remat_prune(&r)
}

@(private)
add_user :: proc(r: ^Remat, x, c: ^UOp) {
	l, ok := &r.users[x]
	if !ok {
		r.users[x] = make([dynamic]^UOp, scratch())
		l = &r.users[x]
	}
	append(l, c)
}

@(private)
users_of :: proc(r: ^Remat, x: ^UOp) -> []^UOp {
	if l, ok := r.users[x]; ok do return l[:]
	return nil
}

@(private)
remove_user :: proc(r: ^Remat, x, c: ^UOp) {
	l, ok := &r.users[x]
	if !ok do return
	for y, i in l do if y == c {
		unordered_remove(l, i)
		return
	}
}

// u's cone in its group (topo order, u last) and its leaves (group inputs,
// stored members). false: too big, or more than one full-size leaf. Reductions
// reached are leaves and get stored.
@(private)
remat_cone :: proc(r: ^Remat, u: ^UOp, cone, leaves: ^[dynamic]^UOp) -> bool {
	clear(cone)
	clear(leaves)
	A := uf_find(&r.s.uf, u)
	visit :: proc(r: ^Remat, A, x: ^UOp, cone, leaves: ^[dynamic]^UOp, top: bool) {
		for y in cone do if y == x do return
		for y in leaves do if y == x do return
		member := x in r.s.uf && uf_find(&r.s.uf, x) == A
		if !member || (!top && x in r.stored) || !op_is_ewise(x.op) {
			if member && !op_is_ewise(x.op) do r.stored[x] = true
			if x.op != .Const do append(leaves, x)
			return
		}
		for y in x.src do visit(r, A, y, cone, leaves, false)
		append(cone, x)
	}
	visit(r, A, u, cone, leaves, true)
	if len(cone) > REMAT_MAX_CONE do return false
	big := 0
	for y in leaves do if numel(y.shape) == numel(u.shape) do big += 1
	return big <= 1
}

// Would group B still fit one kernel with u's cone cloned in? Counted as the
// kernel builders count: inputs per phase for a reduction (prologue: shaped
// like its input, epilogue: like its output), buffers deduplicated.
@(private)
remat_fits :: proc(r: ^Remat, B, u: ^UOp, cone, leaves: []^UOp) -> bool {
	group := r.members[B][:]
	have := r.clones[B]
	red: ^UOp
	for v in group do if v.op == .Sum || v.op == .ReduceMax do red = v
	insns := len(group)
	for x in cone do if !(x in have) do insns += 1
	if insns > MAX_FUSED_INSNS + (red != nil ? 1 : 0) do return false
	if red != nil {
		for y in leaves do if !reduce_mergeable(red, y, &r.s.views) do return false
	}
	// the group with the cone's originals added (u then counts as inside)
	all := make([dynamic]^UOp, scratch())
	defer delete(all)
	append(&all, ..group)
	append(&all, ..cone)
	inside :: proc(all: []^UOp, x: ^UOp) -> bool {
		for v in all do if v == x do return true
		return false
	}
	phase_of :: proc(red, v: ^UOp) -> int {
		if red == nil || v == red || shapes_equal(v.shape, red.src[0].shape) do return 0
		return 1
	}
	n_in := 0
	bufs := make([dynamic]^UOp, scratch())
	defer delete(bufs)
	seen := make([dynamic]^UOp, scratch())
	defer delete(seen)
	for phase in 0 ..< 2 {
		clear(&seen)
		for v in all {
			if phase_of(red, v) != phase do continue
			for x in v.src {
				if inside(all[:], x) || inside(seen[:], x) do continue
				append(&seen, x)
				if x.op != .Const && !inside(bufs[:], x) do append(&bufs, x)
			}
		}
		n_in += len(seen)
	}
	stores := 0
	for v in group {
		if v in r.s.sinks || r.stored[v] {
			stores += 1
			continue
		}
		for c in users_of(r, v) do if !inside(all[:], c) {
			stores += 1
			break
		}
	}
	return n_in <= MAX_FUSED_INPUTS && len(bufs) + stores + (red != nil ? 1 : 0) <= MAX_FUSED_BUFS
}

// Clone u's cone into group B (reusing clones B already has) and point B's
// readers of u at the clone.
@(private)
remat_inline :: proc(r: ^Remat, B, u: ^UOp, cone: []^UOp) {
	s := r.s
	if !(B in r.clones) do r.clones[B] = make(map[^UOp]^UOp, scratch())
	have := &r.clones[B]
	fresh := make([dynamic]^UOp, scratch())
	defer delete(fresh)
	for x in cone {
		if x in have^ do continue
		c := new(UOp, scratch())
		c^ = UOp{op = x.op, arg = x.arg, shape = x.shape, internal = true}
		c.src = make([]^UOp, len(x.src), scratch())
		for y, i in x.src {
			if cy, ok := have[y]; ok do c.src[i] = cy
			else do c.src[i] = y
		}
		for y, i in c.src {
			if src_seen_before(c, i) do continue
			s.consumers[y] += 1
			add_user(r, y, c)
		}
		have[x] = c
		s.uf[c] = B
		append(&fresh, c)
		append(&s.clones, c)
	}
	// B's readers of u now read the clone
	cu := have[u]
	group := &r.members[B]
	for v in group {
		hit := false
		for &y, i in v.src do if y == u {
			append(&s.rewired, Rewire{v, i, u}) // undone after the realize
			y = cu
			hit = true
		}
		if !hit do continue
		s.consumers[u] -= 1
		s.consumers[cu] += 1
		remove_user(r, u, v)
		add_user(r, cu, v)
	}
	// clones first (after earlier clones): they only read leaves and each other
	inject_at(group, r.n_front[B], ..fresh[:])
	r.n_front[B] += len(fresh)
}

// Members nothing reads any more (and not stored) don't run; their srcs lose
// a reader. Reverse topo order, so whole dead chains go.
@(private)
remat_prune :: proc(r: ^Remat) {
	s := r.s
	for i := len(s.topo) - 1; i >= 0; i -= 1 {
		u := s.topo[i]
		if !(u in s.uf) || u in s.sinks || u in s.dead || s.consumers[u] > 0 do continue
		kill(r, u)
	}
	// clones are never sinks; one might have lost all readers to a later clone
	for c in s.clones {
		if c in s.dead || s.consumers[c] > 0 do continue
		kill(r, c)
	}
	kill :: proc(r: ^Remat, u: ^UOp) {
		s := r.s
		s.dead[u] = true
		B := uf_find(&s.uf, u)
		group := &r.members[B]
		for v, j in group do if v == u {
			ordered_remove(group, j)
			if j < r.n_front[B] do r.n_front[B] -= 1
			break
		}
		for y, i in u.src {
			if src_seen_before(u, i) do continue
			s.consumers[y] -= 1
			remove_user(r, y, u)
			if y in s.uf && !(y in s.sinks) && s.consumers[y] == 0 && !(y in s.dead) do kill(r, y)
		}
	}
}

// An original node's src pointed at a clone for the length of a realize.
@(private)
Rewire :: struct {
	node: ^UOp,
	i:    int,
	src:  ^UOp,
}

// Put the graph back (rewired srcs) and free the clones.
@(private)
free_clones :: proc(s: ^Schedule) {
	for w in s.rewired do w.node.src[w.i] = w.src
	delete(s.rewired)
	for c in s.clones {
		delete(c.src, scratch())
		free(c, scratch())
	}
	delete(s.clones)
}
