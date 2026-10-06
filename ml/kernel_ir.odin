package ml

// ============================================================================
// Kernel IR: one description for every kernel the scheduler emits (GEMM
// aside), rendered by each backend (kernel_render.odin for Metal/CUDA,
// kernel_cpu.odin for the CPU).
//
//   index space   dims[0..nd), row-major; dims [red_lo, red_hi) are reduced
//   loads         a buffer seen through a view: strides per index dim
//                 (0 = broadcast) and an offset
//   nodes         an expression DAG in topological order: Load, Const, ALU
//                 ops, and at most one Reduce
//   stores        node values written densely over the output space (the
//                 index space without the reduced dims)
//
// Nodes before the Reduce run once per index point (the prologue); nodes
// after it once per output point (the epilogue, which may only read the
// Reduce, Consts and loads with zero strides on the reduced dims).
//
// A fused elementwise group is a kernel without a reduce; a reduction is
// Load → Reduce; a permute is a single Load through a permuted view. Stage 3+
// of the performance plan grow these (prologues, epilogues, views) without new
// kernel types. Programs are shape-generic: dims, strides, offsets and Const
// values travel as parameters, so the compiled program is cached by structure.
// ============================================================================

import "core:fmt"
import "core:hash"

MAX_KERNEL_BUFS :: MAX_FUSED_INPUTS + MAX_FUSED_INSNS
MAX_KERNEL_NODES :: 2 * MAX_FUSED_INPUTS + MAX_FUSED_INSNS + 2

// How a renderer can address a load (picked from its strides over the dims).
View_Mode :: enum u8 {
	Direct,  // contiguous over all dims: data[i]
	Scalar,  // all strides 0: data[0]
	Row,     // contiguous over trailing dims [lo, nd), 0 before:   data[i % n]
	Col,     // contiguous over leading dims [0, hi), 0 after:     data[i / inner]
	Block,   // contiguous over [lo, hi), 0 elsewhere:              data[(i / inner) % n]
	Generic, // anything else: unravel i and dot with strides
}

K_View :: struct {
	buf:     int,
	offset:  int,
	strides: [MAX_DIMS]int,
	mode:    View_Mode, // set by kernel_finish (elementwise kernels; reductions address by strides)
	n:       int,       // Row/Block: elements in the contiguous run
	inner:   int,       // Col/Block: index points per element
}

K_Kind :: enum u8 {
	Load,
	Const,
	ALU,
	Reduce,
}

K_Node :: struct {
	kind:  K_Kind,
	op:    Op,  // ALU: the elementwise op; Reduce: .Sum or .ReduceMax
	a, b:  int, // operand nodes (b unused by unary ops)
	load:  int, // Load: index into loads
	value: f32, // Const
}

K_Store :: struct {
	node: int,
	buf:  int,
}

Kernel :: struct {
	nd:             int,
	dims:           [MAX_DIMS]int,
	red_lo, red_hi: int, // reduced dims; equal: no reduction
	bufs:           [MAX_KERNEL_BUFS][]f32,
	n_bufs, n_in:   int, // bufs[0..n_in) are read, the rest written
	loads:          [MAX_KERNEL_BUFS]K_View,
	n_loads:        int,
	nodes:          [MAX_KERNEL_NODES]K_Node,
	n_nodes:        int,
	stores:         [MAX_FUSED_INSNS]K_Store,
	n_stores:       int,
	label:          string, // for profiles and ML_DEBUG=2 (only built when timing)
}

kernel_has_reduce :: proc(k: ^Kernel) -> bool {
	return k.red_hi > k.red_lo
}

// Points of the output space (index space without the reduced dims).
kernel_out_points :: proc(k: ^Kernel) -> int {
	n := 1
	for d in 0 ..< k.nd do if d < k.red_lo || d >= k.red_hi do n *= k.dims[d]
	return n
}

kernel_points :: proc(k: ^Kernel) -> int {
	n := 1
	for d in 0 ..< k.nd do n *= k.dims[d]
	return n
}

// The Reduce node, or -1.
kernel_reduce_node :: proc(k: ^Kernel) -> int {
	for n, i in k.nodes[:k.n_nodes] do if n.kind == .Reduce do return i
	return -1
}

// ---- building -----------------------------------------------------------------

// Inputs are shared between loads of the same buffer; outputs (dedup = false)
// always get their own slot, after every input.
@(private)
kernel_add_buf :: proc(k: ^Kernel, data: []f32, dedup := true) -> int {
	if dedup do for b, i in k.bufs[:k.n_bufs] do if raw_data(b) == raw_data(data) && len(b) == len(data) do return i
	k.bufs[k.n_bufs] = data
	k.n_bufs += 1
	return k.n_bufs - 1
}

@(private)
kernel_add_node :: proc(k: ^Kernel, n: K_Node) -> int {
	k.nodes[k.n_nodes] = n
	k.n_nodes += 1
	return k.n_nodes - 1
}

@(private)
kernel_add_load :: proc(k: ^Kernel, data: []f32, strides: []int, offset := 0) -> int {
	v := K_View{buf = kernel_add_buf(k, data), offset = offset}
	copy(v.strides[:], strides)
	k.loads[k.n_loads] = v
	k.n_loads += 1
	return kernel_add_node(k, K_Node{kind = .Load, load = k.n_loads - 1})
}

// Strides of a tensor broadcast (right-aligned) to the kernel's dims.
@(private)
broadcast_strides :: proc(shape: []i32, nd: int) -> (st: [MAX_DIMS]int) {
	pad := nd - len(shape)
	for d in 0 ..< len(shape) do if shape[d] != 1 do st[pad + d] = int(stride_of(shape, d))
	return
}

@(private)
dense_strides :: proc(dims: []int) -> (st: [MAX_DIMS]int) {
	s := 1
	for d := len(dims) - 1; d >= 0; d -= 1 {
		st[d] = s
		s *= dims[d]
	}
	return
}

// A fused elementwise group → one kernel. false: over the budget (the caller
// runs the nodes one by one).
kernel_from_group :: proc(group, stores: []^UOp) -> (k: Kernel, ok: bool) {
	if len(group) > MAX_FUSED_INSNS do return
	out := group[len(group) - 1].shape
	k.nd = len(out)
	for d in 0 ..< k.nd do k.dims[d] = int(out[d])
	node_of := make(map[^UOp]int, scratch())
	defer delete(node_of)
	for u in group do node_of[u] = -1
	n_inputs := 0
	for u in group {
		for x in u.src {
			if x in node_of do continue
			if n_inputs == MAX_FUSED_INPUTS do return // buffers + constants, as before
			n_inputs += 1
			if x.op == .Const {
				node_of[x] = kernel_add_node(&k, K_Node{kind = .Const, value = x.arg.(f32)})
				continue
			}
			st := broadcast_strides(x.shape, k.nd)
			node_of[x] = kernel_add_load(&k, x.data, st[:k.nd])
		}
	}
	k.n_in = k.n_bufs
	for u in group {
		b := op_is_binary(u.op) ? node_of[u.src[1]] : 0
		node_of[u] = kernel_add_node(&k, K_Node{kind = .ALU, op = u.op, a = node_of[u.src[0]], b = b})
	}
	for u in stores {
		alloc_out(u)
		k.stores[k.n_stores] = K_Store{node = node_of[u], buf = kernel_add_buf(&k, u.data, false)}
		k.n_stores += 1
	}
	kernel_finish(&k)
	return k, true
}

// out[i] = src through a view (permute, or a plain copy).
kernel_copy :: proc(out, src: []f32, dims: []int, strides: []int) -> (k: Kernel) {
	k.nd = len(dims)
	copy(k.dims[:], dims)
	l := kernel_add_load(&k, src, strides)
	k.n_in = k.n_bufs
	k.stores[0] = K_Store{node = l, buf = kernel_add_buf(&k, out, false)}
	k.n_stores = 1
	kernel_finish(&k)
	return
}

kernel_from_permute :: proc(u: ^UOp) -> Kernel {
	src := u.src[0]
	order := u.arg.([]i32)
	dims, st: [MAX_DIMS]int
	for o, i in order {
		dims[i] = int(src.shape[o])
		st[i] = int(stride_of(src.shape, int(o)))
	}
	return kernel_copy(u.data, src.data, dims[:len(order)], st[:len(order)])
}

// [outer, r, inner] → [outer, inner]: one run of adjacent reduced axes.
kernel_reduce_run :: proc(op: Op, out, src: []f32, outer, r, inner: int) -> (k: Kernel) {
	k.nd = 3
	k.dims[0], k.dims[1], k.dims[2] = outer, r, inner
	k.red_lo, k.red_hi = 1, 2
	l := kernel_add_load(&k, src, {r * inner, inner, 1})
	k.n_in = k.n_bufs
	red := kernel_add_node(&k, K_Node{kind = .Reduce, op = op, a = l})
	k.stores[0] = K_Store{node = red, buf = kernel_add_buf(&k, out, false)}
	k.n_stores = 1
	kernel_finish(&k)
	return
}

// Classify every load (elementwise kernels; harmless for reductions).
kernel_finish :: proc(k: ^Kernel) {
	for &v in k.loads[:k.n_loads] do classify_view(&v, k.dims[:k.nd])
}

// The cheapest addressing for a view over dims.
classify_view :: proc(v: ^K_View, dims: []int) {
	nd := len(dims)
	dense := dense_strides(dims)
	lo, hi := -1, -1 // the dims the view actually moves along (size > 1, stride ≠ 0)
	for d in 0 ..< nd {
		if dims[d] != 1 && v.strides[d] != 0 {
			if lo < 0 do lo = d
			hi = d + 1
		}
	}
	v.n, v.inner = 1, 1
	if lo < 0 {
		v.mode = .Scalar
		return
	}
	// contiguous over [lo, hi), broadcast (stride 0 or size 1) elsewhere?
	run := 1
	for d := hi - 1; d >= lo; d -= 1 {
		if dims[d] != 1 && v.strides[d] != run {
			v.mode = .Generic
			return
		}
		run *= dims[d]
	}
	for d in lo ..< hi do if dims[d] != 1 && v.strides[d] == 0 {
		v.mode = .Generic
		return
	}
	v.n = run
	for d in hi ..< nd do v.inner *= dims[d]
	all := true
	for d in 0 ..< nd do if dims[d] != 1 && v.strides[d] != dense[d] do all = false
	switch {
	case all: v.mode = .Direct
	case v.inner == 1: v.mode = .Row
	case lo == 0 || all_ones(dims[:lo]): v.mode = .Col
	case: v.mode = .Block
	}
}

@(private)
all_ones :: proc(dims: []int) -> bool {
	for d in dims do if d != 1 do return false
	return true
}

// Structural hash: what the compiled program depends on (not dims, strides,
// offsets or constant values, which are parameters).
kernel_hash :: proc(k: ^Kernel, variant: int) -> u64 {
	key: [1024]u8
	n := 0
	put :: proc(key: ^[1024]u8, n: ^int, v: int) {
		key[n^] = u8(v)
		key[n^ + 1] = u8(v >> 8)
		n^ += 2
	}
	put(&key, &n, variant)
	put(&key, &n, k.nd)
	put(&key, &n, k.red_lo)
	put(&key, &n, k.red_hi)
	put(&key, &n, k.n_bufs)
	put(&key, &n, k.n_in)
	for v in k.loads[:k.n_loads] {
		put(&key, &n, v.buf)
		put(&key, &n, int(v.mode))
	}
	for x in k.nodes[:k.n_nodes] {
		put(&key, &n, int(x.kind))
		put(&key, &n, int(x.op))
		put(&key, &n, x.a)
		put(&key, &n, x.b)
		put(&key, &n, x.load)
	}
	for s in k.stores[:k.n_stores] {
		put(&key, &n, s.node)
		put(&key, &n, s.buf)
	}
	return hash.fnv64a(key[:n])
}

// "fused[Add,Mul] [1024 4 64]", "reduce[Sum] [64 5000 1]", "copy [4 8 5 8]"
kernel_describe :: proc(k: ^Kernel) -> string {
	b := make([dynamic]u8, context.temp_allocator)
	w :: proc(b: ^[dynamic]u8, s: string) { append(b, s) }
	red := kernel_reduce_node(k)
	n_alu := 0
	for n in k.nodes[:k.n_nodes] do if n.kind == .ALU do n_alu += 1
	switch {
	case red >= 0: w(&b, fmt.tprintf("reduce[%v", k.nodes[red].op))
	case n_alu == 0: w(&b, "copy[")
	case: w(&b, "fused[")
	}
	first := red < 0
	for n in k.nodes[:k.n_nodes] {
		if n.kind != .ALU do continue
		w(&b, fmt.tprintf("%s%v", first ? "" : ",", n.op))
		first = false
	}
	w(&b, "] [")
	for d in 0 ..< k.nd do w(&b, fmt.tprintf("%s%d", d > 0 ? " " : "", k.dims[d]))
	w(&b, "]")
	return string(b[:])
}

// ---- fused reductions -----------------------------------------------------------
//
// A reduction with its producers (prologue: elementwise members shaped like the
// reduce input, evaluated per [o, r, k]) and its consumers (epilogue: members
// shaped like the reduce output, evaluated per [o, k]). The input's axes are
// merged to the canonical [outer, r, inner]: every tensor read must then have
// one stride per merged range (reduce_mergeable checks it beforehand).

// The single run of reduced axes [lo, hi) of shape (size-1 axes ignored), if one.
reduce_run :: proc(shape: []i32, axes: []i32) -> (lo, hi: int, ok: bool) {
	red: [MAX_DIMS]bool
	for a in axes do red[a] = true
	lo, hi = -1, -1
	for d in 0 ..< len(shape) {
		if !red[d] || shape[d] == 1 do continue
		if lo < 0 do lo = d
		else if hi >= 0 && hi < d {
			for e in hi ..< d do if shape[e] != 1 do return // a second run
		}
		hi = d + 1
	}
	return lo, hi, lo >= 0
}

// Strides over `dims` merged into the ranges [0, lo), [lo, hi), [hi, nd).
merge3 :: proc(dims: []i32, st: [MAX_DIMS]int, lo, hi: int) -> (m: [3]int, ok: bool) {
	ranges := [3][2]int{{0, lo}, {lo, hi}, {hi, len(dims)}}
	for r, i in ranges {
		inner := -1 // stride of the innermost non-1 dim seen so far (walking left)
		size := 1
		zero, moving := false, false
		for d := r[1] - 1; d >= r[0]; d -= 1 {
			if dims[d] == 1 do continue
			if st[d] == 0 {
				zero = true
				continue
			}
			moving = true
			if inner < 0 {
				inner = st[d]
				m[i] = st[d]
			} else if st[d] != inner * size {
				return // not one stride
			}
			inner, size = st[d], int(dims[d])
		}
		if zero && moving do return // broadcast and moving within one range
	}
	return m, true
}

@(private)
reduce_dims :: proc(r: ^UOp) -> (X: []i32, lo, hi: int, ok: bool) {
	X = r.src[0].shape
	lo, hi, ok = reduce_run(X, r.arg.([]i32))
	return
}

// Can tensor t (broadcastable to the reduce input) be read in r's canonical space?
reduce_mergeable :: proc(r: ^UOp, t: ^UOp) -> bool {
	X, lo, hi, ok := reduce_dims(r)
	if !ok do return false
	_, mok := merge3(X, broadcast_strides(t.shape, len(X)), lo, hi)
	return mok
}

// group: topo order, exactly one Sum/ReduceMax; members shaped like its input
// (prologue) or its output (epilogue).
kernel_from_reduce_group :: proc(group, stores: []^UOp) -> (k: Kernel, ok: bool) {
	r: ^UOp
	for u in group do if u.op == .Sum || u.op == .ReduceMax do r = u
	X, lo, hi, rok := reduce_dims(r)
	if !rok do return
	k.nd = 3
	k.dims[0], k.dims[1], k.dims[2] = int(numel(X[:lo])), int(numel(X[lo:hi])), int(numel(X[hi:]))
	k.red_lo, k.red_hi = 1, 2
	// members by node; outside inputs per phase (prologue values live inside
	// the reduction loop, so the epilogue loads its own)
	node_of := make(map[^UOp]int, scratch())
	pro_in := make(map[^UOp]int, scratch())
	epi_in := make(map[^UOp]int, scratch())
	defer {
		delete(node_of)
		delete(pro_in)
		delete(epi_in)
	}
	members := make(map[^UOp]bool, scratch())
	defer delete(members)
	for u in group do members[u] = true
	n_inputs := 0
	operand :: proc(node_of, inputs: ^map[^UOp]int, x: ^UOp) -> int {
		if n, ok := node_of[x]; ok do return n
		return inputs[x]
	}
	add_inputs :: proc(k: ^Kernel, members: ^map[^UOp]bool, inputs: ^map[^UOp]int, n_inputs: ^int, u: ^UOp, X: []i32, lo, hi: int) -> bool {
		for x in u.src {
			if x in members^ || x in inputs^ do continue
			if n_inputs^ == MAX_FUSED_INPUTS do return false
			n_inputs^ += 1
			if x.op == .Const {
				inputs[x] = kernel_add_node(k, K_Node{kind = .Const, value = x.arg.(f32)})
				continue
			}
			m, ok := merge3(X, broadcast_strides(x.shape, len(X)), lo, hi)
			if !ok do return false
			inputs[x] = kernel_add_load(k, x.data, m[:])
		}
		return true
	}
	alu :: proc(k: ^Kernel, node_of, inputs: ^map[^UOp]int, u: ^UOp) {
		b := op_is_binary(u.op) ? operand(node_of, inputs, u.src[1]) : 0
		node_of[u] = kernel_add_node(k, K_Node{kind = .ALU, op = u.op, a = operand(node_of, inputs, u.src[0]), b = b})
	}
	// prologue (shaped like the input), the reduce, epilogue (shaped like the output)
	for u in group do if u != r && shapes_equal(u.shape, X) {
		if !add_inputs(&k, &members, &pro_in, &n_inputs, u, X, lo, hi) do return
	}
	if !add_inputs(&k, &members, &pro_in, &n_inputs, r, X, lo, hi) do return
	for u in group do if u != r && shapes_equal(u.shape, X) do alu(&k, &node_of, &pro_in, u)
	node_of[r] = kernel_add_node(&k, K_Node{kind = .Reduce, op = r.op, a = operand(&node_of, &pro_in, r.src[0])})
	for u in group do if u != r && !shapes_equal(u.shape, X) {
		if !add_inputs(&k, &members, &epi_in, &n_inputs, u, X, lo, hi) do return
	}
	for u in group do if u != r && !shapes_equal(u.shape, X) do alu(&k, &node_of, &epi_in, u)
	k.n_in = k.n_bufs
	for u in stores {
		alloc_out(u)
		k.stores[k.n_stores] = K_Store{node = node_of[u], buf = kernel_add_buf(&k, u.data, false)}
		k.n_stores += 1
	}
	kernel_finish(&k)
	return k, true
}
