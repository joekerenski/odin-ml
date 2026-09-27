package ml

// ============================================================================
// realize — schedule and run the UOp graph. The only executor.
//
// realize_all(sinks):
//   1. topo-sort unrealized nodes (realized nodes are buffers: stop there),
//   2. fuse: an ewise node joins its ewise src when shapes match and the src
//      has exactly one consumer → one loop, intermediates never stored,
//   3. fold: a 2D Permute feeding only MatMuls becomes a GEMM transpose flag,
//   4. run in topo order: fused groups (fuse.odin), views, primitive kernels.
//
// A node's .data is kept only if something outside its fused group (or the
// caller) needs it. Anything elided is simply recomputed if asked for later.
//
// debug_level: 1 step summary, 2 one line per kernel, 3 print graph first.
// ============================================================================

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
}

schedule_destroy :: proc(s: ^Schedule) {
	delete(s.topo)
	delete(s.consumers)
	delete(s.sinks)
	delete(s.folded)
	delete(s.uf)
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

is_2d_swap :: proc(u: ^UOp) -> bool {
	if u.op != .Permute || len(u.shape) != 2 do return false
	o := u.arg.([]i32)
	return o[0] == 1 && o[1] == 0
}

realize_all :: proc(sinks: []^UOp) {
	s: Schedule
	defer schedule_destroy(&s)

	visited: map[^UOp]bool
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
	matmul_uses: map[^UOp]int
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
		if is_2d_swap(u) && !(u in s.sinks) && matmul_uses[u] == s.consumers[u] {
			s.folded[u] = true
		}
	}

	// fusion groups
	for u in s.topo do if op_is_ewise(u.op) do s.uf[u] = u
	for u in s.topo {
		if !op_is_ewise(u.op) do continue
		for x in u.src {
			if !(x in s.uf) || s.consumers[x] != 1 do continue
			if !shapes_equal(x.shape, u.shape) do continue
			ra, rb := uf_find(&s.uf, x), uf_find(&s.uf, u)
			if ra != rb do s.uf[ra] = rb
		}
	}

	// last member (in topo order) of each group runs the group
	last: map[^UOp]^UOp
	defer delete(last)
	for u in s.topo do if u in s.uf do last[uf_find(&s.uf, u)] = u

	group: [dynamic]^UOp
	defer delete(group)
	for u in s.topo {
		if u in s.folded do continue
		if u in s.uf {
			root := uf_find(&s.uf, u)
			if last[root] != u do continue
			clear(&group)
			for v in s.topo do if v in s.uf && uf_find(&s.uf, v) == root do append(&group, v)
			run_group(&s, group[:])
			continue
		}
		run_node(&s, u)
	}

	if debug_level == 1 {
		ms := f64(time.tick_since(step_start)) / 1e6
		fmt.printfln("  realize: %d kernels  %.3f ms", counters.kernels - k0, ms)
	}
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

alloc_out :: proc(u: ^UOp) {
	n := numel(u.shape)
	u.data = make([]f32, n)
	counters.bytes_alloc += i64(n) * size_of(f32)
}

run_group :: proc(s: ^Schedule, group: []^UOp) {
	stores: [dynamic]^UOp
	defer delete(stores)
	for u in group do if needs_store(s, u, group) do append(&stores, u)
	assert(len(stores) > 0, "fused group has no stores")

	t0: time.Tick
	if debug_level >= 2 do t0 = time.tick_now()

	if !run_fused(group, stores[:]) {
		// too big for one kernel: run each node on its own
		for u in group {
			single := []^UOp{u}
			ok := run_fused(single, single)
			assert(ok)
			counters.kernels += 1
		}
	} else {
		counters.kernels += 1
		counters.fused_ops += len(group) - 1
	}

	if debug_level >= 2 {
		dt := i64(time.tick_since(t0))
		counters.time_ns += dt
		fmt.printf("  kernel  fused[")
		for u, i in group do fmt.printf("%s%v", i > 0 ? "," : "", u.op)
		fmt.printfln("] shape=%v  %7.3f ms", group[len(group) - 1].shape, f64(dt) / 1e6)
	}
}

// Operand for GEMM: a realized buffer, or a folded transpose of one.
gemm_operand :: proc(u: ^UOp) -> (data: []f32, trans: bool) {
	if u.data == nil && is_2d_swap(u) do return u.src[0].data, true
	return u.data, false
}

run_node :: proc(s: ^Schedule, u: ^UOp) {
	for x in u.src do assert(x.data != nil || is_2d_swap(x), "run_node: src not realized")

	t0: time.Tick
	if debug_level >= 2 do t0 = time.tick_now()

	if u.op == .Reshape {
		u.data = u.src[0].data // view: same dense buffer
		return
	}

	alloc_out(u)
	#partial switch u.op {
	case .Sum:
		sum_kernel(u.data, u.src[0].data, u.src[0].shape, u.arg.([]i32))
	case .Permute:
		permute_kernel(u.data, u.src[0].data, u.src[0].shape, u.arg.([]i32))
	case .MatMul:
		a, ta := gemm_operand(u.src[0])
		b, tb := gemm_operand(u.src[1])
		M, N := u.shape[0], u.shape[1]
		K := u.src[0].shape[1]
		matmul_f32(u.data, a, b, M, K, N, ta, tb)
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
	case .CrossEntropy:
		x := u.src[0]
		u.data[0] = cross_entropy_f32(x.data, x.shape[0], x.shape[1], u.arg.([]u8))
	case .CrossEntropyBwd:
		g, x := u.src[0], u.src[1]
		cross_entropy_backward(u.data, g.data[0], x.data, x.shape[0], x.shape[1], u.arg.([]u8))
	case:
		fmt.panicf("run_node: no kernel for %v", u.op)
	}
	counters.kernels += 1

	if debug_level >= 2 {
		dt := i64(time.tick_since(t0))
		counters.time_ns += dt
		fmt.printfln("  kernel  %-16v shape=%v  %7.3f ms", u.op, u.shape, f64(dt) / 1e6)
	}
}
