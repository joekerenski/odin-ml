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
// this checks for CE specifically, this is too literal. How do we realize in a general way??
realize :: proc(t: ^Tensor) {
	if t == nil || is_realized(t) do return

	topo: [dynamic]^Tensor = make([dynamic]^Tensor, 0)
	visited: map[^Tensor]bool
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
		// Movement ops (Reshape/Transpose) are views — no buffer alloc.
		is_view := node.ctx.op == .Reshape || node.ctx.op == .Transpose
		nbytes: i64 = 0
		if !is_view {
			nbytes = i64(n) * size_of(f32)
			node.data = make([]f32, n)
			counters.bytes_alloc += nbytes
		}

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
			tag := is_view ? "view" : (node.device == .Metal ? "Metal" : "CPU")
			fmt.printfln(
				"  fwd %3d %-14s shape=%v  %6.1f KB  %7.3f ms  %s",
				k, op_name(node.ctx.op), node.shape,
				f64(nbytes) / 1024.0,
				f64(dt_ns) / 1e6,
				tag,
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
// NOTE: this has to be done better, we cannot rely on enumerating all these cases??
forward_op :: proc(out: ^Tensor) {
	ctx := out.ctx
	assert(ctx != nil)
	p := ctx.parents

	switch ctx.op {
	case .Add:
		a, b := ensure_contig(p[0]), ensure_contig(p[1])
		cls := classify_binary(a, b, out.shape[:])
		switch cls.kind {
		case .Same:
			if out.device == .Metal {
				metal_add(out.data, a.data, b.data)
			} else {
				add_f32_contiguous(out.data, a.data, b.data)
			}
		case .Scalar_B:
			add_scalar_contiguous(out.data, a.data, b.data[0])
		case .Scalar_A:
			add_scalar_contiguous(out.data, b.data, a.data[0])
		case .Row_B:
			add_row_broadcast(out.data, a.data, b.data, cls.outer, cls.inner)
		case .Col_B:
			add_col_broadcast(out.data, a.data, b.data, cls.outer, cls.inner)
		case .Col_A:
			add_col_broadcast(out.data, b.data, a.data, cls.outer, cls.inner)
		case .NCHW_Bias_B:
			copy(out.data, a.data)
			bias_add_nchw(out.data, b.data, a.shape[0], a.shape[1], a.shape[2], a.shape[3])
		case .NCHW_Bias_A:
			copy(out.data, b.data)
			bias_add_nchw(out.data, a.data, b.shape[0], b.shape[1], b.shape[2], b.shape[3])
		case .Generic:
			for i in 0..<len(out.data) do out.data[i] = 0
			broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
			broadcast_add_into(out.data, out.shape[:], b.data, b.shape[:])
		}

	case .Sub:
		a, b := ensure_contig(p[0]), ensure_contig(p[1])
		cls := classify_binary(a, b, out.shape[:])
		switch cls.kind {
		case .Same:
			sub_f32_contiguous(out.data, a.data, b.data)
		case .Scalar_B:
			add_scalar_contiguous(out.data, a.data, -b.data[0])
		case .Scalar_A:
			neg_f32_contiguous(out.data, b.data)
			add_scalar_contiguous(out.data, out.data, a.data[0])
		case .Row_B, .Col_B, .Col_A, .NCHW_Bias_B, .NCHW_Bias_A, .Generic:
			for i in 0..<len(out.data) do out.data[i] = 0
			broadcast_add_into(out.data, out.shape[:], a.data, a.shape[:])
			neg_b := make([]f32, len(b.data))
			for i in 0..<len(b.data) do neg_b[i] = -b.data[i]
			broadcast_add_into(out.data, out.shape[:], neg_b, b.shape[:])
		}

	case .Mul:
		a, b := ensure_contig(p[0]), ensure_contig(p[1])
		cls := classify_binary(a, b, out.shape[:])
		switch cls.kind {
		case .Same:
			mul_f32_contiguous(out.data, a.data, b.data)
		case .Scalar_B:
			mul_scalar_contiguous(out.data, a.data, b.data[0])
		case .Scalar_A:
			mul_scalar_contiguous(out.data, b.data, a.data[0])
		case .Row_B:
			mul_row_broadcast(out.data, a.data, b.data, cls.outer, cls.inner)
		case .Col_B:
			mul_col_broadcast(out.data, a.data, b.data, cls.outer, cls.inner)
		case .Col_A:
			mul_col_broadcast(out.data, b.data, a.data, cls.outer, cls.inner)
		case .NCHW_Bias_B, .NCHW_Bias_A, .Generic:
			mul_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
		}

	case .Div:
		a, b := ensure_contig(p[0]), ensure_contig(p[1])
		if shapes_equal(a.shape[:], b.shape[:]) {
			div_f32_contiguous(out.data, a.data, b.data)
		} else if is_scalar_shape(b.shape[:]) {
			mul_scalar_contiguous(out.data, a.data, 1.0 / b.data[0])
		} else {
			div_broadcast_into(out.data, out.shape[:], a.data, a.shape[:], b.data, b.shape[:])
		}

	case .Neg:
		neg_f32_contiguous(out.data, contig_data(p[0]))

	case .ReLU:
		relu_f32_contiguous(out.data, contig_data(p[0]))

	case .Sigmoid:
		ad := contig_data(p[0])
		for i in 0..<len(out.data) {
			out.data[i] = 1.0 / (1.0 + math.exp(-ad[i]))
		}

	case .Sum:
		a := ensure_contig(p[0])
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
		// View when parent is contiguous (same numel → dense strides already set).
		// Otherwise densify into a fresh buffer.
		a := p[0]
		if is_contiguous(a) {
			out.data = a.data
			// out.strides already dense from new_tensor_lazy
		} else {
			out.data = make([]f32, numel(out.shape[:]))
			counters.bytes_alloc += i64(len(out.data)) * size_of(f32)
			materialize_to(out.data, a)
		}

	case .Transpose:
		// Zero-copy view: share data, swap strides.
		a := p[0]
		out.data = a.data
		for i in 0..<len(a.strides) do out.strides[i] = a.strides[i]
		ax0, ax1 := int(ctx.axis), int(ctx.axis1)
		out.strides[ax0], out.strides[ax1] = out.strides[ax1], out.strides[ax0]

	case .MatMul:
		a, b := contig_data(p[0]), contig_data(p[1])
		M, K, N := p[0].shape[0], p[0].shape[1], p[1].shape[1]
		matmul_f32(out.data, a, b, M, K, N)

	case .CrossEntropy:
		logits_t := ensure_contig(p[0])
		logits := logits_t.data
		B := p[0].shape[0]
		C := p[0].shape[1]
		if ctx.cache == nil {
			ctx.cache = new_tensor_lazy({B, C})
		}
		softmax := ctx.cache
		if softmax.data == nil {
			softmax.data = make([]f32, numel(softmax.shape[:]))
		}
		for b in 0..<B {
			row := b * C
			max_val := logits[row]
			for c in 1..<C {
				if logits[row + c] > max_val do max_val = logits[row + c]
			}
			sum_exp: f32 = 0
			for c in 0..<C {
				e := math.exp(logits[row + c] - max_val)
				softmax.data[row + c] = e
				sum_exp += e
			}
			inv := 1.0 / sum_exp
			for c in 0..<C do softmax.data[row + c] *= inv
		}
		total: f32 = 0
		for b in 0..<B {
			total += -math.ln(softmax.data[b * C + i32(ctx.labels[b])] + 1e-12)
		}
		out.data[0] = total / f32(B)

	case .Conv2d:
		x, w := contig_data(p[0]), contig_data(p[1])
		N, Ci, H, W := p[0].shape[0], p[0].shape[1], p[0].shape[2], p[0].shape[3]
		Co := p[1].shape[0]
		conv2d_f32(
			out.data, x, w,
			N, Ci, H, W, Co, ctx.kH, ctx.kW, ctx.sH, ctx.sW, ctx.pH, ctx.pW,
		)

	case .MaxPool2d:
		x := contig_data(p[0])
		N, C, H, W := p[0].shape[0], p[0].shape[1], p[0].shape[2], p[0].shape[3]
		if ctx.indices == nil {
			ctx.indices = make([]i32, len(out.data))
		}
		maxpool2d_f32(
			out.data, x, ctx.indices,
			N, C, H, W, ctx.kH, ctx.kW, ctx.sH, ctx.sW, ctx.pH, ctx.pW,
		)
	}
}
