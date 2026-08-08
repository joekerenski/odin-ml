package ml

// ============================================================================
// Lazy evaluation — build the graph, then realize.
//
// Op          — what kind of computation (Add, MatMul, …)
// Context     — a LazyOp: Op + parents + meta (axis, cache, …)
// Tensor.ctx  — nil for leaves (sources); set for op nodes
// Tensor.data — nil until realized (except leaves, which always have data)
//
// Sources → intermediates → sink
//   zeros/from_data     add/mul/…      loss you realize()
//
// realize(sink):
//   1. topo-sort the DAG ending at sink (parents before children)
//   2. for each unrealized node: allocate data, run forward kernel
//
// backward() calls realize() first so grads see actual values.
// ============================================================================

import "core:fmt"
import "core:math"
import "core:time"

// A leaf (source) has no ctx and always holds data.
// An op tensor is realized once its data buffer exists.
is_realized :: proc(t: ^Tensor) -> bool {
	if t == nil do return true
	if t.ctx == nil do return true // leaf
	return t.data != nil
}

// Shape-only tensor: no data yet. Used by ops when building the graph.
new_tensor_lazy :: proc(shape: []i32, requires_grad := false, device := default_device) -> ^Tensor {
	assert(len(shape) <= MAX_DIMS, "new_tensor_lazy: too many dims")
	t := new(Tensor)
	t.data = nil
	t.shape = copy_shape(shape)
	t.strides = compute_strides(t.shape[:])
	t.requires_grad = requires_grad
	t.device = device
	return t
}

// Ensure t holds data. Leaves are no-ops. Op nodes run the whole subgraph.
// debug_level >= 2 logs each kernel; >= 3 prints the graph first; >= 1 step summary.
realize :: proc(t: ^Tensor) {
	if t == nil || is_realized(t) do return

	topo: [dynamic]^Tensor = make([dynamic]^Tensor, 0)
	visited: [dynamic]^Tensor = make([dynamic]^Tensor, 0)
	defer delete(topo)
	defer delete(visited)
	topo_sort(t, &topo, &visited)

	if debug_level >= 3 {
		print_graph(t, "realize")
	}

	step_start: time.Tick
	if debug_level >= 1 do step_start = time.tick_now()

	k := 0
	for node in topo {
		if node.ctx == nil do continue // leaf — already has data
		if node.data != nil do continue // already realized (shared subgraph)
		for p in node.ctx.parents {
			assert(is_realized(p), "realize: parent not realized")
		}
		n := numel(node.shape[:])
		nbytes := i64(n) * size_of(f32)
		node.data = make([]f32, n)
		counters.bytes_alloc += nbytes

		t0: time.Tick
		if debug_level >= 2 do t0 = time.tick_now()
		// CrossEntropy softmax cache is an extra buffer
		if node.ctx.op == .CrossEntropy && node.ctx.cache != nil && node.ctx.cache.data == nil {
			cn := numel(node.ctx.cache.shape[:])
			counters.bytes_alloc += i64(cn) * size_of(f32)
		}
		forward_op(node)
		if debug_level >= 2 {
			dt_ns := i64(time.tick_since(t0))
			counters.time_ns += dt_ns
			fmt.printfln(
				"  fwd %3d %-14s shape=%v  %6.1f KB  %7.3f ms  %s",
				k, op_name(node.ctx.op), node.shape,
				f64(nbytes) / 1024.0,
				f64(dt_ns) / 1e6,
				node.device == .Metal ? "Metal" : "CPU",
			)
		}
		counters.kernels += 1
		k += 1
	}
	counters.nodes += k

	if debug_level == 1 {
		ms := f64(time.tick_since(step_start)) / 1e6
		fmt.printfln("  realize: %d kernels  %.3f ms", k, ms)
	}
}

// Read a scalar after realizing (handy for loss logging).
item :: proc(t: ^Tensor) -> f32 {
	realize(t)
	assert(len(t.data) >= 1, "item: empty tensor")
	return t.data[0]
}

// Run the forward kernel for one already-allocated op node.
forward_op :: proc(out: ^Tensor) {
	ctx := out.ctx
	assert(ctx != nil)
	p := ctx.parents

	switch ctx.op {
	case .Add:
		a, b := p[0], p[1]
		if out.device == .Metal && shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
			metal_add(out.data, a.data, b.data)
		} else if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
			add_f32_contiguous(out.data, a.data, b.data)
		} else if out.device == .CPU && is_scalar_shape(b.shape[:]) && is_contiguous(a) {
			add_scalar_contiguous(out.data, a.data, b.data[0])
		} else if out.device == .CPU && is_scalar_shape(a.shape[:]) && is_contiguous(b) {
			add_scalar_contiguous(out.data, b.data, a.data[0])
		} else if out.device == .CPU && is_contiguous(a) {
			if matches, inner_n := inner_match_size(a.shape[:], b.shape[:]); matches && numel(b.shape[:]) == inner_n && inner_n > 1 {
				outer := numel(a.shape[:]) / i32(inner_n)
				add_row_broadcast(out.data, a.data, b.data, outer, inner_n)
			} else if is_contiguous(b) {
				if matches, outer, inner_n := col_broadcast_match(a.shape[:], b.shape[:]); matches && inner_n > 1 {
					add_col_broadcast(out.data, a.data, b.data, outer, inner_n)
				} else if matches, outer, inner_n := col_broadcast_match(b.shape[:], a.shape[:]); matches && inner_n > 1 {
					add_col_broadcast(out.data, b.data, a.data, outer, inner_n)
				} else {
					for i in 0..<len(out.data) do out.data[i] = 0
					broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
					broadcast_add_into(out.data, out.shape[:], b.data, b.shape[:])
				}
			} else {
				for i in 0..<len(out.data) do out.data[i] = 0
				broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
				broadcast_add_into(out.data, out.shape[:], b.data, b.shape[:])
			}
		} else {
			for i in 0..<len(out.data) do out.data[i] = 0
			broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
			broadcast_add_into(out.data, out.shape[:], b.data, b.shape[:])
		}

	case .Sub:
		a, b := p[0], p[1]
		if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
			sub_f32_contiguous(out.data, a.data, b.data)
		} else if is_scalar_shape(b.shape[:]) && is_contiguous(a) {
			add_scalar_contiguous(out.data, a.data, -b.data[0])
		} else {
			for i in 0..<len(out.data) do out.data[i] = 0
			broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
			neg_b := make([]f32, len(b.data))
			for i in 0..<len(b.data) do neg_b[i] = -b.data[i]
			broadcast_add_into(out.data, out.shape[:], neg_b, b.shape[:])
		}

	case .Mul:
		a, b := p[0], p[1]
		if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
			mul_f32_contiguous(out.data, a.data, b.data)
		} else if is_scalar_shape(b.shape[:]) && is_contiguous(a) {
			mul_scalar_contiguous(out.data, a.data, b.data[0])
		} else if is_scalar_shape(a.shape[:]) && is_contiguous(b) {
			mul_scalar_contiguous(out.data, b.data, a.data[0])
		} else if is_contiguous(a) {
			if matches, inner_n := inner_match_size(a.shape[:], b.shape[:]); matches && numel(b.shape[:]) == inner_n && inner_n > 1 {
				outer := numel(a.shape[:]) / i32(inner_n)
				mul_row_broadcast(out.data, a.data, b.data, outer, inner_n)
			} else if is_contiguous(b) {
				if matches, outer, inner_n := col_broadcast_match(a.shape[:], b.shape[:]); matches && inner_n > 1 {
					mul_col_broadcast(out.data, a.data, b.data, outer, inner_n)
				} else if matches, outer, inner_n := col_broadcast_match(b.shape[:], a.shape[:]); matches && inner_n > 1 {
					mul_col_broadcast(out.data, b.data, a.data, outer, inner_n)
				} else {
					mul_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
				}
			} else {
				mul_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
			}
		} else {
			mul_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
		}

	case .Div:
		a, b := p[0], p[1]
		if shapes_equal(a.shape[:], b.shape[:]) && is_contiguous(a) && is_contiguous(b) {
			div_f32_contiguous(out.data, a.data, b.data)
		} else {
			div_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
		}

	case .Neg:
		neg_f32_contiguous(out.data, p[0].data)

	case .ReLU:
		relu_f32_contiguous(out.data, p[0].data)

	case .Sigmoid:
		a := p[0]
		for i in 0..<len(out.data) {
			out.data[i] = 1.0 / (1.0 + math.exp(-a.data[i]))
		}

	case .Sum:
		a := p[0]
		axis := ctx.axis
		if axis == -1 {
			s: f32 = 0
			for v in a.data do s += v
			out.data[0] = s
		} else {
			for i in 0..<len(out.data) do out.data[i] = 0
			idx_buf: [MAX_DIMS]i32
			for flat in 0..<len(a.data) {
				unravel_index(i32(flat), a.shape[:], idx_buf[:])
				out_idx: i32 = 0
				for d in 0..<len(a.shape) {
					v := idx_buf[d]
					if d == int(axis) do v = 0
					out_idx += v * stride_of(out.shape[:], d)
				}
				out.data[out_idx] += a.data[flat]
			}
		}

	case .Reshape:
		a := p[0]
		for i in 0..<len(out.data) do out.data[i] = a.data[i]

	case .Transpose:
		a := p[0]
		axis0, axis1 := ctx.axis, ctx.axis1
		idx_buf: [MAX_DIMS]i32
		for flat in 0..<len(a.data) {
			unravel_index(i32(flat), a.shape[:], idx_buf[:])
			idx_buf[axis0], idx_buf[axis1] = idx_buf[axis1], idx_buf[axis0]
			o_flat: i32 = 0
			for d in 0..<len(out.shape) do o_flat += idx_buf[d] * stride_of(out.shape[:], d)
			out.data[o_flat] = a.data[flat]
		}

	case .MatMul:
		a, b := p[0], p[1]
		M, K, N := a.shape[0], a.shape[1], b.shape[1]
		matmul_f32(out.data, a.data, b.data, M, K, N)

	case .CrossEntropy:
		logits := p[0]
		B := logits.shape[0]
		C := logits.shape[1]
		if ctx.cache == nil {
			ctx.cache = new_tensor_lazy({B, C})
		}
		softmax := ctx.cache
		if softmax.data == nil {
			softmax.data = make([]f32, numel(softmax.shape[:]))
		}
		for b in 0..<B {
			row := b * C
			max_val := logits.data[row]
			for c in 1..<C {
				if logits.data[row + c] > max_val do max_val = logits.data[row + c]
			}
			sum_exp: f32 = 0
			for c in 0..<C {
				e := math.exp(logits.data[row + c] - max_val)
				softmax.data[row + c] = e
				sum_exp += e
			}
			inv := 1.0 / sum_exp
			for c in 0..<C do softmax.data[row + c] *= inv
		}
		one_hot := p[1]
		total: f32 = 0
		for b in 0..<B {
			for c in 0..<C {
				if one_hot.data[b*C + c] > 0.5 {
					total += -math.ln(softmax.data[b*C + c] + 1e-12)
					break
				}
			}
		}
		out.data[0] = total / f32(B)
	}
}
