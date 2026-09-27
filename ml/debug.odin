package ml

// ============================================================================
// Debug / inspectability (tinygrad-style DEBUG levels).
//
//   debug_level = 0  silent
//   debug_level = 1  realize/backward step summary (counters)
//   debug_level = 2  per-node lines: order, op, shape, bytes, ms
//   debug_level = 3  also print_graph before execute
//
//   ml.debug_level = 2
//   ml.counters_reset()
//   ml.print_graph(loss)   // inspect lazy DAG without running
//   ml.backward(loss)
//   ml.counters_print("step")
// ============================================================================

import "core:fmt"
import "core:os"
import "core:strconv"

debug_level: int = 0

Counters :: struct {
	kernels:     int, // kernel launches (fused group = 1; views = 0)
	bwd_ops:     int, // grad rules applied by backward()
	bytes_alloc: i64, // node buffers allocated by realize
	time_ns:     i64, // sum of timed kernel / bwd wall time
	nodes:       int, // op nodes seen in last print_graph / realize topo
	fused_ops:   int, // extra ewise ops absorbed into fused kernels (n-1 per group)
}

counters: Counters

counters_reset :: proc() {
	counters = {}
}

counters_print :: proc(label := "step") {
	ms := f64(counters.time_ns) / 1e6
	kb := f64(counters.bytes_alloc) / 1024.0
	fmt.printfln(
		"  [%s] kernels=%d fused=%d bwd=%d alloc=%.1f KB time=%.3f ms nodes=%d",
		label, counters.kernels, counters.fused_ops, counters.bwd_ops, kb, ms, counters.nodes,
	)
}

// Call once at startup if you want ML_DEBUG=N from the environment.
debug_from_env :: proc() {
	v, found := os.lookup_env_alloc("ML_DEBUG", context.allocator)
	if !found do return
	defer delete(v)
	if n, ok := strconv.parse_int(v); ok {
		debug_level = n
	}
}

// Print the DAG ending at sink (no execution). Sources-first topo order.
print_graph :: proc(sink: ^Tensor, label := "graph") {
	topo: [dynamic]^UOp
	visited: map[^UOp]bool
	defer delete(topo)
	defer delete(visited)
	toposort(sink, &topo, &visited)

	index: map[^UOp]int
	defer delete(index)
	for u, i in topo do index[u] = i

	op_nodes := 0
	fmt.printfln("  [%s] %d nodes (sources → sink)", label, len(topo))
	for u, i in topo {
		srcs: [8]int
		for x, j in u.src do srcs[j] = index[x]
		if u.op != .Input && u.op != .Const do op_nodes += 1
		fmt.printfln("  %3d %-16v shape=%-14s <- %-10s realized=%v grad=%v",
			i, u.op, fmt.tprint(u.shape), fmt.tprint(srcs[:len(u.src)]), u.data != nil, u.requires_grad)
	}
	counters.nodes = op_nodes
}

// What realize_all is about to run: unrealized nodes + their fusion group.
print_schedule :: proc(s: ^Schedule) {
	fmt.printfln("  [schedule] %d nodes", len(s.topo))
	for u, i in s.topo {
		fmt.printfln("  %3d %-16v shape=%v%s", i, u.op, u.shape, u in s.sinks ? "  (sink)" : "")
	}
}

// Training loops are ~10x slower without optimization (the fused kernels rely
// on inlining + vectorization). Call at startup of long-running programs.
warn_if_unoptimized :: proc() {
	when ODIN_OPTIMIZATION_MODE == .None {
		fmt.eprintln("note: unoptimized build (~10x slower). Use `make` targets or -o:speed.")
	}
}
