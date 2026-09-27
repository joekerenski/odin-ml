package main

// ============================================================================
// Backend parity: every Metal kernel path vs the CPU backend.
//
//   odin run tests/metal -o:speed
//
// Each case builds the same graph (same inputs) once per device, realizes it
// (with backward where there are grads) and compares all outputs and grads.
// Shapes are picked to hit each kernel variant: hardware 8×8 GEMM, split-K,
// tiled fallback, tiny batched GEMM, both reduce kernels, permute, every
// broadcast load mode — plus the models built from them.
// ============================================================================

import "core:fmt"
import "core:math"
import "core:math/rand"
import ml "../../ml"

failed, passed: int

Case :: struct {
	name:   string,
	shapes: [][]i32, // leaf inputs (random, requires grad)
	build:  proc(x: []^ml.Tensor) -> ^ml.Tensor,
}

// Build + realize (+ backward) on one device. Returns output and grads, flat.
run :: proc(c: Case, dev: ml.Device) -> (out: []f32, grads: [][]f32) {
	ml.set_device(dev)
	rand.reset(7)
	xs := make([]^ml.Tensor, len(c.shapes))
	for s, i in c.shapes do xs[i] = ml.randn(s, 0, 1, requires_grad = true)
	y := c.build(xs)
	ml.backward(ml.sum(ml.mul(y, ml.randn(y.shape, 0, 1))))
	out = y.data
	grads = make([][]f32, len(xs))
	for x, i in xs do grads[i] = x.grad != nil ? x.grad.data : nil
	return
}

close :: proc(a, b: []f32) -> (ok: bool, worst: f32) {
	if len(a) != len(b) do return false, math.inf_f32(1)
	ok = true
	for i in 0 ..< len(a) {
		d := abs(a[i] - b[i])
		if d > 1e-3 + 1e-3 * abs(b[i]) do ok = false
		worst = max(worst, d / (1e-3 + abs(b[i])))
	}
	return
}

check :: proc(c: Case) {
	cpu_out, cpu_g := run(c, .CPU)
	gpu_out, gpu_g := run(c, .Metal)
	ok, worst := close(gpu_out, cpu_out)
	for g, i in cpu_g {
		if g == nil do continue
		gok, gw := close(gpu_g[i], g)
		ok &&= gok
		worst = max(worst, gw)
	}
	if ok {
		passed += 1
		fmt.printfln("  ok    %-36s (worst rel %.1e)", c.name, worst)
	} else {
		failed += 1
		fmt.printfln("  FAIL  %-36s (worst rel %.1e)", c.name, worst)
	}
}

main :: proc() {
	fmt.println("=== Metal vs CPU backend parity ===")
	if !ml.set_device(.Metal) {
		fmt.println("no Metal device")
		return
	}

	mm :: proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.matmul(x[0], x[1]) }
	mm_ta :: proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.matmul(ml.mT(x[0]), x[1]) }
	mm_tb :: proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.matmul(x[0], ml.mT(x[1])) }
	mm_tt :: proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.matmul(ml.mT(x[0]), ml.mT(x[1])) }
	cases := []Case {
		// GEMM paths (backward adds the transposed / split-K variants)
		{"gemm hw 8x8   5120x64 @ 64x64", {{5120, 64}, {64, 64}}, mm},
		{"gemm hw 8x8   64x256 @ 256x64", {{64, 256}, {256, 64}}, mm},
		{"gemm hw ta    (64x5120)^T @ 64x64", {{64, 5120}, {64, 64}}, mm_ta},
		{"gemm hw tb    48x32 @ (40x32)^T", {{48, 32}, {40, 32}}, mm_tb},
		{"gemm hw tt    (32x48)^T @ (40x32)^T", {{32, 48}, {40, 32}}, mm_tt},
		{"gemm split-K  64x4096 @ 4096x64", {{64, 4096}, {4096, 64}}, mm},
		{"gemm tiled    37x50 @ 50x29", {{37, 50}, {50, 29}}, mm},
		{"gemm tiled tb 37x50 @ (29x50)^T", {{37, 50}, {29, 50}}, mm_tb},
		{"bmm hw        [3,16,24] @ [3,24,40]", {{3, 16, 24}, {3, 24, 40}}, mm},
		{"bmm tiny      [64,5,8] @ [64,8,10]", {{64, 5, 8}, {64, 8, 10}}, mm},
		{"bmm tiny tb   q @ k^T [2,4,5,8]", {{2, 4, 5, 8}, {2, 4, 5, 8}}, mm_tb},
		{"bmm bcast     [4,6,8] @ [8,16]", {{4, 6, 8}, {8, 16}}, mm},
		// reductions: thread-per-output and group variants, multi-axis
		{"sum long rows [4,5000] axis 1", {{4, 5000}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.sum(x[0], 1) }},
		{"sum columns   [5120,64] axis 0", {{5120, 64}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.sum(x[0], 0) }},
		{"sum all       [300,257]", {{300, 257}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.sum(x[0]) }},
		{"sum axes {0,2} [8,6,10]", {{8, 6, 10}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.sum_axes(x[0], {0, 2}) }},
		{"max axis -1   [64,5,10]", {{64, 5, 10}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.max_axis(x[0], -1) }},
		{"max long      [3,4096]", {{3, 4096}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.max_axis(x[0], 1) }},
		// movement
		{"permute 4D    [2,3,4,5] (0,2,1,3)", {{2, 3, 4, 5}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.permute(x[0], {0, 2, 1, 3}) }},
		{"permute 3D    [4,5,6] (2,0,1)", {{4, 5, 6}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.permute(x[0], {2, 0, 1}) }},
		// fused ewise, every load mode
		{"ewise row+col [33,17]+[17]+[33,1]", {{33, 17}, {17}, {33, 1}}, proc(x: []^ml.Tensor) -> ^ml.Tensor {
			return ml.relu(ml.mul(ml.add(x[0], x[1]), ml.exp(ml.mul(x[2], ml.scalar(0.1)))))
		}},
		{"ewise block   [4,6,5,5]+[1,6,1,1]", {{4, 6, 5, 5}, {1, 6, 1, 1}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.sigmoid(ml.add(x[0], x[1])) }},
		{"ewise generic [3,1,5]*[1,4,1]", {{3, 1, 5}, {1, 4, 1}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.mul(x[0], x[1]) }},
		{"ewise log/sqrt/div", {{257}}, proc(x: []^ml.Tensor) -> ^ml.Tensor {
			p := ml.add(ml.square(x[0]), ml.scalar(0.5))
			return ml.div(ml.log(p), ml.sqrt(p))
		}},
		// compositions / model pieces
		{"softmax + logsumexp [16,5,10]", {{16, 5, 10}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.add(ml.softmax(x[0], -1), ml.logsumexp(x[0], -1)) }},
		{"layer_norm    [64,5,64]", {{64, 5, 64}}, proc(x: []^ml.Tensor) -> ^ml.Tensor { return ml.layer_norm(x[0]) }},
		{"attention     [32,5,64] x [32,10,64]", {{32, 5, 64}, {32, 10, 64}, {64, 64}, {64, 64}}, proc(x: []^ml.Tensor) -> ^ml.Tensor {
			q := ml.reshape(ml.matmul(x[0], x[2]), {32, 5, 8, 8})
			k := ml.reshape(ml.matmul(x[1], x[3]), {32, 10, 8, 8})
			q = ml.permute(q, {0, 2, 1, 3})
			k = ml.permute(k, {0, 2, 1, 3})
			att := ml.softmax(ml.mul(ml.matmul(q, ml.mT(k)), ml.scalar(0.35)), -1)
			return ml.matmul(att, k)
		}},
		{"cross_entropy [256,10]", {{256, 10}}, proc(x: []^ml.Tensor) -> ^ml.Tensor {
			labels := make([]u8, 256)
			for &l, i in labels do l = u8(i % 10)
			return ml.cross_entropy(x[0], labels)
		}},
		{"conv + pool (CPU fallback)", {{2, 3, 12, 12}, {4, 3, 3, 3}}, proc(x: []^ml.Tensor) -> ^ml.Tensor {
			return ml.max_pool2d(ml.relu(ml.conv2d(x[0], x[1], 1, 1)), 2)
		}},
	}
	for c in cases do check(c)
	fmt.printfln("\n=== %d passed, %d failed ===", passed, failed)
}
