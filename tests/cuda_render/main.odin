package main

// ============================================================================
// Render the CUDA source of every kernel shape the scheduler emits, for a
// syntax/type check by clang on machines without CUDA:
//
//   make check-cuda-render    (needs a clang with CUDA support: Homebrew llvm on macOS)
//
// nvrtc_stub.h declares what NVRTC provides implicitly. The Metal side needs no
// such check: Metal programs compile at test time on any Mac.
// ============================================================================

import "core:fmt"
import "core:os"
import ml "../../ml"

out_dir: string
n_files: int

emit :: proc(name: string, k: ^ml.Kernel, variants: []ml.GPU_Variant) {
	for v in variants {
		src := ml.gpu_source(k, v, .CUDA)
		full := fmt.tprintf("struct P {{ unsigned int v[%d]; }};\n%s", ml.GPU_PARAMS, src)
		path := fmt.tprintf("%s/%s_%v.cu", out_dir, name, v)
		_ = os.write_entire_file(path, transmute([]u8)full)
		n_files += 1
	}
}

main :: proc() {
	out_dir = os.args[1]
	x := ml.randn({4, 6, 5, 7}, 0, 1)
	row := ml.randn({7}, 0, 1)
	col := ml.randn({4, 6, 5, 1}, 0, 1)
	blk := ml.randn({1, 6, 1, 1}, 0, 1)
	gen := ml.randn({4, 1, 5, 1}, 0, 1)
	pos := ml.add(ml.square(x), ml.scalar(0.5))
	ml.realize(pos)
	// every load mode + constants + exp/log/sqrt/max/cmplt/neg/div
	one := ml.randn({1}, 0, 1)
	a0 := ml.add(x, row)
	a := ml.mul(a0, one)
	b := ml.sub(a, col)
	c := ml.div(b, ml.sqrt(pos))
	d := ml.maximum(ml.add(c, blk), ml.neg(ml.exp(gen)))
	e := ml.add(ml.cmplt(d, ml.log(pos)), d)
	group := []^ml.UOp{a0, a, b, c.src[1], c, d.src[0], d.src[1].src[0], d.src[1], d, e.src[0].src[1], e.src[0], e}
	k, ok := ml.kernel_from_group(group, {d, e})
	assert(ok)
	emit("fused_all_modes", &k, {.Elementwise})
	// reductions and their plans
	src := make([]f32, 64 * 5000)
	dst := make([]f32, 64 * 5000)
	kr := ml.kernel_reduce_run(.Sum, dst, src, 4, 5000, 1)
	emit("reduce_sum", &kr, {.Reduce_Thread, .Reduce_Group})
	km := ml.kernel_reduce_run(.ReduceMax, dst, src, 64, 50, 7)
	emit("reduce_max", &km, {.Reduce_Thread, .Reduce_Group})
	// permute / copy through a generic view
	kc := ml.kernel_copy(dst[:840], src[:840], {5, 4, 7, 6}, {42, 210, 1, 7})
	emit("copy_generic", &kc, {.Elementwise})
	kd := ml.kernel_copy(dst[:100], src[:100], {100}, {1})
	emit("copy_direct", &kd, {.Elementwise})
	// a fused reduction: prologue (exp, a broadcast load), a stored prologue value,
	// epilogue (sqrt) — under every plan, plus both passes of a split
	cx := ml.randn({256, 64}, 0, 1)
	cw := ml.randn({64}, 0, 1)
	ce := ml.exp(ml.mul(cx, cw))
	cs := ml.sum(ce, 0)
	cq := ml.sqrt(cs)
	kf, fok := ml.kernel_from_reduce_group({ce.src[0], ce, cs, cq}, {ce, cq})
	assert(fok)
	emit("fused_reduce", &kf, {.Reduce_Thread, .Reduce_Group, .Reduce_Simd})
	pa, pb, _, _ := ml.gpu_split(&kf, 8)
	emit("fused_split_a", &pa, {.Reduce_Thread})
	emit("fused_split_b", &pb, {.Reduce_Thread})
	emit("reduce_sum_simd", &kr, {.Reduce_Simd})
	fmt.printfln("%d CUDA sources written", n_files)
}
