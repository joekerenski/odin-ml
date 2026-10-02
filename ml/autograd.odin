package ml

// ============================================================================
// Autograd — reverse mode on the UOp graph, emitting UOps.
//
// backward(loss):
//   1. topo-sort every node that requires grad,
//   2. walk it in reverse; each op's grad rule builds new UOps from the
//      incoming grad g (chain rule): d(src) = g * d(out)/d(src),
//   3. leaves (.Input with requires_grad) collect their grad in .grad,
//   4. realize loss + all leaf grads in ONE schedule, so the backward graph
//      is fused and scheduled exactly like the forward graph.
//
// Broadcasting: a src broadcast up in the forward gets its grad summed back
// down to its own shape (unbroadcast).
// ============================================================================

import "core:fmt"
import "core:time"

// Sum g down to `shape` over the axes broadcasting expanded.
unbroadcast :: proc(g: ^UOp, shape: []i32) -> ^UOp {
	if shapes_equal(g.shape, shape) do return g
	lead := len(g.shape) - len(shape)
	axes: [MAX_DIMS]i32
	n := 0
	for d in 0 ..< len(g.shape) {
		if d < lead || (shape[d - lead] == 1 && g.shape[d] != 1) {
			axes[n] = i32(d)
			n += 1
		}
	}
	return reshape(sum_axes(g, axes[:n]), shape)
}

// Grad contribution of `u`'s i-th src, given the grad g of u.
grad_rule :: proc(u: ^UOp, g: ^UOp, i: int) -> ^UOp {
	a := u.src[0]
	b := len(u.src) > 1 ? u.src[1] : nil
	s := u.src[i]
	#partial switch u.op {
	case .Add:
		return unbroadcast(g, s.shape)
	case .Sub:
		return unbroadcast(i == 0 ? g : neg(g), s.shape)
	case .Mul:
		return unbroadcast(mul(g, i == 0 ? b : a), s.shape)
	case .Div:
		// d/da a/b = 1/b ;  d/db a/b = -a/b² = -out/b
		if i == 0 do return unbroadcast(div(g, b), s.shape)
		return unbroadcast(neg(div(mul(g, u), b)), s.shape)
	case .Max:
		// ties go to b; relu(x) = max(x, 0) → grad flows where x > 0
		ga := mul(g, cmplt(b, a))
		return unbroadcast(i == 0 ? ga : sub(g, ga), s.shape)
	case .Neg:
		return neg(g)
	case .Exp:
		return mul(g, u)
	case .Log:
		return div(g, a)
	case .Sqrt:
		// d sqrt(a) = 1 / (2 sqrt(a))
		return div(g, mul(u, scalar(2)))
	case .Expand:
		return unbroadcast(g, s.shape)
	case .Sum:
		return expand(g, s.shape)
	case .ReduceMax:
		// grad to the max; ties split it evenly (a <= max, so eq = 1 - (a < max))
		is_max := sub(scalar(1), cmplt(a, u))
		return mul(is_max, div(g, sum_axes(is_max, u.arg.([]i32))))
	case .Reshape:
		return reshape(g, s.shape)
	case .Permute:
		order := u.arg.([]i32)
		inv: [MAX_DIMS]i32
		for o, j in order do inv[o] = i32(j)
		return permute(g, inv[:len(order)])
	case .MatMul:
		if i == 0 do return matmul(g, mT(b))
		return matmul(mT(a), g)
	case .Conv2d:
		win := u.arg.(Window)
		if i == 0 do return new_node(.Conv2dBwdInput, a.shape, win, g, b)
		return new_node(.Conv2dBwdWeight, b.shape, win, g, a)
	case .MaxPool2d:
		return new_node(.MaxPool2dBwd, a.shape, u.arg, g, a)
	}
	fmt.panicf("backward: no grad rule for %v", u.op)
}

backward :: proc(loss: ^Tensor) {
	assert(loss.requires_grad, "backward: tensor does not require grad")

	t0: time.Tick
	if debug_level >= 1 do t0 = time.tick_now()

	topo := make([dynamic]^UOp, scratch())
	visited := make(map[^UOp]bool, scratch())
	defer delete(topo)
	defer delete(visited)
	toposort(loss, &topo, &visited)

	grads := make(map[^UOp]^UOp, scratch())
	defer delete(grads)
	building_grad = true
	grads[loss] = expand(scalar(1), loss.shape)

	sinks := make([dynamic]^UOp, scratch())
	defer delete(sinks)
	append(&sinks, loss)

	for k := len(topo) - 1; k >= 0; k -= 1 {
		u := topo[k]
		g, ok := grads[u]
		if !ok || !u.requires_grad do continue
		if u.op == .Input {
			u.grad = u.grad == nil ? g : add(u.grad, g)
			append(&sinks, u.grad)
			continue
		}
		for s, i in u.src {
			if !s.requires_grad do continue
			gs := grad_rule(u, g, i)
			if prev, has := grads[s]; has {
				grads[s] = add(prev, gs)
			} else {
				grads[s] = gs
			}
		}
		counters.bwd_ops += 1
	}
	building_grad = false

	realize_all(sinks[:])

	if debug_level == 1 {
		ms := f64(time.tick_since(t0)) / 1e6
		fmt.printfln("  backward: %d grad rules, fwd+bwd scheduled together  %.3f ms", counters.bwd_ops, ms)
	}
}
