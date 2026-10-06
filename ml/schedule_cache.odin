package ml

// ============================================================================
// Schedule cache: a training step builds the same graph every time (new
// nodes, same structure), so the scheduler's decisions — views, fusion
// groups and their member order, rematerialized clones, dead members,
// consumer counts — are recorded by position and replayed onto the next
// graph with that structure, skipping plan_views / fuse_groups / remat.
//
// Structure = the topo order with each node's op, shape, arg (not Const
// values: those are kernel parameters), flags, and its srcs as topo
// positions or leaves (realized nodes, numbered by first use, with their
// shapes); plus the settings decisions depend on (device, ML_VIEWS, remat,
// the Metal encoder kind).
// The key bytes are compared in full, not just hashed. ML_SCHED_CACHE=0 off.
// ============================================================================

import "core:hash"
import "core:mem"

sched_cache_enabled := true

// A node in a record: ≥ 0 topo position, < 0 leaf -(k + 1), ≥ REF_CLONE clone.
@(private)
Ref :: i32
@(private)
REF_CLONE :: Ref(1 << 24)

@(private)
Sched_Record :: struct {
	key:        []u8,
	hash:       u64,
	views:      [dynamic]Rec_View,
	alias:      [dynamic][2]Ref, // node, base
	store_into: [dynamic]Rec_Store,
	uf:         [dynamic][2]Ref, // node, root (clones too)
	members:    [dynamic]Rec_Group,
	clones:     [dynamic]Rec_Clone,
	rewired:    [dynamic][3]Ref, // node, src slot, clone
	dead:       [dynamic]Ref,
	consumers:  [dynamic][2]Ref, // node, count
	order:      [dynamic]Ref,
}

@(private)
Rec_View :: struct {
	node, base: Ref,
	st:         [MAX_DIMS]int,
}

@(private)
Rec_Store :: struct {
	m, v: Ref,
	st:   [MAX_DIMS]int,
}

@(private)
Rec_Group :: struct {
	root: Ref,
	list: []Ref,
}

@(private)
Rec_Clone :: struct {
	orig: Ref, // the node it copies (op, arg, shape): original or earlier clone
	srcs: []Ref,
}

// Positions of this schedule's nodes: topo index, leaves by first use.
@(private)
Sched_Index :: struct {
	of:     map[^UOp]Ref,
	leaves: [dynamic]^UOp,
	clones: map[^UOp]Ref,
}

@(private)
sched_records: [dynamic]^Sched_Record

@(private)
SCHED_RECORDS_MAX :: 8

// The structure key of s (and the index that maps nodes to positions).
@(private)
sched_key :: proc(s: ^Schedule, ix: ^Sched_Index) -> []u8 {
	ix.of = make(map[^UOp]Ref, len(s.topo), scratch())
	ix.leaves = make([dynamic]^UOp, scratch())
	ix.clones = make(map[^UOp]Ref, scratch())
	for u, i in s.topo do ix.of[u] = Ref(i)
	key := make([dynamic]u8, 0, 64 * len(s.topo), scratch())
	put :: proc(key: ^[dynamic]u8, v: int) {
		x := u32(v)
		append(key, u8(x), u8(x >> 8), u8(x >> 16), u8(x >> 24))
	}
	put(&key, int(backend.device))
	put(&key, int(view_reads))
	put(&key, int(remat_mode))
	put(&key, int(metal_concurrent))
	for u in s.topo {
		put(&key, int(u.op))
		put(&key, len(u.shape))
		for d in u.shape do put(&key, int(d))
		switch a in u.arg {
		case f32: // a kernel parameter, not structure
		case []i32:
			put(&key, len(a))
			for x in a do put(&key, int(x))
		case Window:
			put(&key, int(a.kH)); put(&key, int(a.kW)); put(&key, int(a.sH)); put(&key, int(a.sW)); put(&key, int(a.pH)); put(&key, int(a.pW))
		}
		put(&key, int(u.internal) | int(u in s.sinks) << 1)
		put(&key, len(u.src))
		for x in u.src {
			if r, ok := ix.of[x]; ok {
				put(&key, int(r))
				continue
			}
			ix.of[x] = Ref(-len(ix.leaves) - 1)
			append(&ix.leaves, x)
			put(&key, int(ix.of[x]))
			put(&key, int(x.op))
			put(&key, len(x.shape))
			for d in x.shape do put(&key, int(d))
		}
	}
	return key[:]
}

@(private)
sched_index_destroy :: proc(ix: ^Sched_Index) {
	delete(ix.of)
	delete(ix.leaves)
	delete(ix.clones)
}

@(private)
sched_lookup :: proc(key: []u8) -> ^Sched_Record {
	h := hash.fnv64a(key)
	for r in sched_records do if r.hash == h && mem.compare(r.key, key) == 0 do return r
	return nil
}

// Record s's decisions (after plan_views, fuse_groups, remat).
@(private)
sched_record :: proc(s: ^Schedule, ix: ^Sched_Index, key: []u8, members: ^map[^UOp][dynamic]^UOp) {
	a := scratch()
	r := new(Sched_Record, a)
	r.key = make([]u8, len(key), a)
	copy(r.key, key)
	r.hash = hash.fnv64a(key)
	for c, i in s.clones do ix.clones[c] = REF_CLONE + Ref(i)
	ref :: proc(ix: ^Sched_Index, x: ^UOp) -> Ref {
		if r, ok := ix.clones[x]; ok do return r
		r, ok := ix.of[x]
		assert(ok, "sched_record: node outside the schedule")
		return r
	}
	r.views = make([dynamic]Rec_View, a)
	for u, v in s.views do append(&r.views, Rec_View{ref(ix, u), ref(ix, v.base), v.st})
	r.alias = make([dynamic][2]Ref, a)
	for u, b in s.alias do append(&r.alias, [2]Ref{ref(ix, u), ref(ix, b)})
	r.store_into = make([dynamic]Rec_Store, a)
	for m, into in s.store_into do append(&r.store_into, Rec_Store{ref(ix, m), ref(ix, into.v), into.st})
	r.uf = make([dynamic][2]Ref, a)
	for u in s.uf do append(&r.uf, [2]Ref{ref(ix, u), ref(ix, uf_find(&s.uf, u))})
	r.members = make([dynamic]Rec_Group, a)
	for root, list in members {
		g := Rec_Group{ref(ix, root), make([]Ref, len(list), a)}
		for x, i in list do g.list[i] = ref(ix, x)
		append(&r.members, g)
	}
	r.clones = make([dynamic]Rec_Clone, a)
	for c, i in s.clones {
		rc := Rec_Clone{ref(ix, s.clone_of[i]), make([]Ref, len(c.src), a)}
		for x, i in c.src do rc.srcs[i] = ref(ix, x)
		append(&r.clones, rc)
	}
	r.rewired = make([dynamic][3]Ref, a)
	for w in s.rewired do append(&r.rewired, [3]Ref{ref(ix, w.node), Ref(w.i), ref(ix, w.node.src[w.i])})
	r.dead = make([dynamic]Ref, a)
	for u in s.dead do append(&r.dead, ref(ix, u))
	r.consumers = make([dynamic][2]Ref, a)
	for u, n in s.consumers {
		x, ok := ix.clones[u]
		if !ok do x, ok = ix.of[u]
		if !ok || x < 0 do continue // leaves are never tracked
		append(&r.consumers, [2]Ref{x, Ref(n)})
	}
	r.order = make([dynamic]Ref, a)
	for u in s.order do append(&r.order, ref(ix, u))
	if len(sched_records) == SCHED_RECORDS_MAX {
		sched_record_free(sched_records[0])
		ordered_remove(&sched_records, 0)
	}
	if sched_records == nil do sched_records = make([dynamic]^Sched_Record, scratch())
	append(&sched_records, r)
}

@(private)
sched_record_free :: proc(r: ^Sched_Record) {
	a := scratch()
	delete(r.key, a)
	delete(r.views)
	delete(r.alias)
	delete(r.store_into)
	delete(r.uf)
	for g in r.members do delete(g.list, a)
	delete(r.members)
	for c in r.clones do delete(c.srcs, a)
	delete(r.clones)
	delete(r.rewired)
	delete(r.dead)
	delete(r.consumers)
	delete(r.order)
	free(r, a)
}

// Replay a record onto s (same structure): its decisions, with this graph's nodes.
@(private)
sched_replay :: proc(s: ^Schedule, ix: ^Sched_Index, r: ^Sched_Record, members: ^map[^UOp][dynamic]^UOp) {
	clones := make([]^UOp, len(r.clones), scratch())
	defer delete(clones, scratch())
	node :: proc(s: ^Schedule, ix: ^Sched_Index, clones: []^UOp, x: Ref) -> ^UOp {
		if x >= REF_CLONE do return clones[x - REF_CLONE]
		if x < 0 do return ix.leaves[-x - 1]
		return s.topo[x]
	}
	// clones first (their originals and srcs may be clones made before them)
	for rc, i in r.clones {
		o := node(s, ix, clones, rc.orig)
		c := new(UOp, scratch())
		c^ = UOp{op = o.op, arg = o.arg, shape = o.shape, internal = true}
		c.src = make([]^UOp, len(rc.srcs), scratch())
		clones[i] = c
		append(&s.clones, c)
		append(&s.clone_of, o)
	}
	for rc, i in r.clones do for x, j in rc.srcs do clones[i].src[j] = node(s, ix, clones, x)
	for v in r.views do s.views[node(s, ix, clones, v.node)] = View{node(s, ix, clones, v.base), v.st}
	for p in r.alias do s.alias[node(s, ix, clones, p[0])] = node(s, ix, clones, p[1])
	for st in r.store_into do s.store_into[node(s, ix, clones, st.m)] = Store_Into{node(s, ix, clones, st.v), st.st}
	for p in r.uf do s.uf[node(s, ix, clones, p[0])] = node(s, ix, clones, p[1])
	for g in r.members {
		list := make([dynamic]^UOp, len(g.list), scratch())
		for x, i in g.list do list[i] = node(s, ix, clones, x)
		members[node(s, ix, clones, g.root)] = list
	}
	for w in r.rewired {
		n := node(s, ix, clones, w[0])
		append(&s.rewired, Rewire{n, int(w[1]), n.src[w[1]]})
		n.src[w[1]] = node(s, ix, clones, w[2])
	}
	for x in r.dead do s.dead[node(s, ix, clones, x)] = true
	clear(&s.consumers)
	for p in r.consumers do s.consumers[node(s, ix, clones, p[0])] = int(p[1])
	for x in r.order do append(&s.order, node(s, ix, clones, x))
}
