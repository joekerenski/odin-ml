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
	kernels:     int, // forward kernel launches (fused group = 1; views = 0)
	bwd_ops:     int, // backward_op dispatches
	bytes_alloc: i64, // output buffers allocated in realize
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

op_name :: proc(op: Op) -> string {
	switch op {
	case .Add: return "Add"
	case .Sub: return "Sub"
	case .Mul: return "Mul"
	case .Div: return "Div"
	case .Neg: return "Neg"
	case .MatMul: return "MatMul"
	case .Sum: return "Sum"
	case .Reshape: return "Reshape"
	case .Transpose: return "Transpose"
	case .ReLU: return "ReLU"
	case .Sigmoid: return "Sigmoid"
	case .CrossEntropy: return "CrossEntropy"
	case .Conv2d: return "Conv2d"
	case .MaxPool2d: return "MaxPool2d"
	}
	return "?"
}

// Print the lazy DAG ending at sink (no execution). Sources-first topo order.
print_graph :: proc(sink: ^Tensor, label := "graph") {
	if sink == nil {
		fmt.printfln("  [%s] (nil)", label)
		return
	}
	topo: [dynamic]^Tensor
	visited: map[^Tensor]bool
	defer delete(topo)
	defer delete(visited)
	topo_sort(sink, &topo, &visited)

	index_of :: proc(topo: []^Tensor, t: ^Tensor) -> int {
		for n, i in topo do if n == t do return i
		return -1
	}

	op_nodes := 0
	fmt.printfln("  [%s] %d tensors (sources → sink)", label, len(topo))
	for node, i in topo {
		if node.ctx == nil {
			fmt.printfln("  %3d leaf  %-14s shape=%v  n=%d  data=%v  requires_grad=%v",
				i, "Leaf", node.shape, numel(node.shape[:]), node.data != nil, node.requires_grad)
			continue
		}
		op_nodes += 1
		// parent topo indices
		pids: [8]int
		np := len(node.ctx.parents)
		assert(np <= 8)
		for j in 0..<np do pids[j] = index_of(topo[:], node.ctx.parents[j])
		fmt.printfln("  %3d op    %-14s shape=%v  <- %v  realized=%v",
			i, op_name(node.ctx.op), node.shape, pids[:np], node.data != nil)
	}
	counters.nodes = op_nodes
	fmt.printfln("  [%s] %d op nodes", label, op_nodes)
}
