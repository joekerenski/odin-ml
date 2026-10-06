package main

// ============================================================================
// make perf — the performance plan's yardstick (STATUS.md).
//
// Fixed workloads, each: step time (median, p10, p90 over timed steps, no
// profiling), kernels per step, and one profiled step broken down by kernel
// type (count, device ms, achieved GB/s and GFLOP/s). Compared against
// bench/perf/baseline-<machine>.json when present; every run is appended to
// build/perf.jsonl.
//
//   make perf                   ML_DEVICE picks the device (cpu|metal|cuda)
//   make perf-baseline          write this machine's baseline (commit it)
//   ... -- dt_fusion_step       run only workloads whose name contains this
// ============================================================================

import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:time"
import dt "../../dt"
import ml "../../ml"

GIT_REV :: #config(GIT_REV, "unknown")

Workload :: struct {
	name:   string,
	note:   string,
	steps:  int,
	setup:  proc(),
	step:   proc(), // one step; allocations go to the current allocator (an arena, reset per step)
}

Kind_Stat :: struct {
	kind:    string,
	kernels: int,
	ms:      f64,
	bytes:   i64,
	flops:   i64,
}

Result :: struct {
	name:      string,
	median_ms: f64,
	p10_ms:    f64,
	p90_ms:    f64,
	kernels:   int,
	profile:   []Kind_Stat,
}

Record :: struct {
	rev:       string,
	date:      string,
	machine:   string,
	workloads: []Result,
}

// ---- workloads ---------------------------------------------------------------

// M4: sample a batch of problems, forward + backward, Adam.
fusion: struct {
	m:      dt.Dt_Full,
	params: [dynamic]^ml.Tensor,
	opt:    ^ml.Adam,
}

fusion_setup :: proc() {
	ml.seed(0)
	fusion.m = dt.dt_full_model(dt.FUSION_MODEL, dt.OBS_FEAT)
	fusion.params = dt.dt_full_params(fusion.m)
	fusion.opt = ml.new_adam(fusion.params[:], lr = 1e-4)
}

fusion_step :: proc() {
	B :: 1024
	K :: dt.K_FUSION
	m := &fusion.m
	comps := ml.new_tensor({B, K, dt.N_FEAT})
	z := ml.new_tensor({B, dt.N_OBS, dt.OBS_FEAT})
	x := ml.new_tensor({B, 1, dt.D})
	prior := make([]dt.Component, K)
	mu_L := dt.mu_chol()
	for i in 0 ..< B {
		dt.sample_prior(prior, mu_L)
		xs := dt.sample_mixture(prior)
		dt.fusion_inputs(comps, z, i, prior, dt.sample_range(xs), dt.sample_bearing(xs))
		for j in 0 ..< dt.D do x.data[i * dt.D + j] = f32(xs[j])
	}
	qp, qq := dt.dt_full_forward(m, comps, z)
	ml.backward(ml.add(dt.gmm_full_nll(qq, x, m.maps), dt.gmm_full_nll(qp, x, m.maps)))
	ml.adam_step(fusion.opt)
	ml.clear_grads(..fusion.params[:])
}

// M3: the Table 1 transformer.
table1: struct {
	m:      dt.Dt_Model,
	params: [dynamic]^ml.Tensor,
	opt:    ^ml.Adam,
}

table1_setup :: proc() {
	ml.seed(0)
	table1.m = dt.dt_model(dt.DT_PAPER)
	table1.params = dt.dt_params(table1.m)
	table1.opt = ml.new_adam(table1.params[:], lr = 1e-4)
}

table1_step :: proc() {
	b := dt.make_batch(1024, 10)
	qp, qq := dt.dt_forward(&table1.m, b.prior, b.z)
	ml.backward(ml.add(dt.gmm_nll(qq, b.y), dt.gmm_nll(qp, b.y)))
	ml.adam_step(table1.opt)
	ml.clear_grads(..table1.params[:])
}

// MNIST-shaped MLP and CNN (examples/mnist, examples/mnist_cnn) on synthetic data.
small: struct {
	l1, l2, fc: ml.Linear,
	c1, c2:     ml.Conv2d,
	mlp, cnn:   [dynamic]^ml.Tensor,
	mlp_opt:    ^ml.SGD,
	cnn_opt:    ^ml.SGD,
	labels:     []u8,
}

synthetic_batch :: proc(shape: []i32) -> ^ml.Tensor {
	return ml.uniform(shape, 0, 1)
}

mlp_setup :: proc() {
	ml.seed(1)
	small.l1, small.l2 = ml.linear(784, 128, .He), ml.linear(128, 10, .Xavier)
	ml.linear_params(&small.mlp, small.l1)
	ml.linear_params(&small.mlp, small.l2)
	small.mlp_opt = ml.new_sgd_list(0.05, 0.9, small.mlp[:])
	small.labels = make([]u8, 128)
	for &l, i in small.labels do l = u8(i % 10)
}

mlp_step :: proc() {
	h := ml.relu(ml.linear_forward(&small.l1, synthetic_batch({128, 784})))
	ml.backward(ml.cross_entropy(ml.linear_forward(&small.l2, h), small.labels))
	ml.sgd_step(small.mlp_opt)
	ml.clear_grads(..small.mlp[:])
}

cnn_setup :: proc() {
	ml.seed(2)
	small.c1 = ml.conv2d_layer(1, 8, 3, stride = 1, padding = 1, init = .He)
	small.c2 = ml.conv2d_layer(8, 16, 3, stride = 1, padding = 1, init = .He)
	small.fc = ml.linear(16 * 7 * 7, 10, .Xavier)
	ml.conv_params(&small.cnn, small.c1)
	ml.conv_params(&small.cnn, small.c2)
	ml.linear_params(&small.cnn, small.fc)
	small.cnn_opt = ml.new_sgd_list(0.05, 0.9, small.cnn[:])
	if small.labels == nil {
		small.labels = make([]u8, 128)
		for &l, i in small.labels do l = u8(i % 10)
	}
}

cnn_step :: proc() {
	h := ml.relu(ml.conv2d_forward(&small.c1, synthetic_batch({128, 1, 28, 28})))
	h = ml.max_pool2d(h, 2)
	h = ml.max_pool2d(ml.relu(ml.conv2d_forward(&small.c2, h)), 2)
	ml.backward(ml.cross_entropy(ml.linear_forward(&small.fc, ml.flatten(h)), small.labels))
	ml.sgd_step(small.cnn_opt)
	ml.clear_grads(..small.cnn[:])
}

// Micro: the device's memory bandwidth and GEMM rate as seen through the pipeline,
// plus the two row-reduction patterns the transformers are full of.
micro: struct {
	big, a, b, rows, ln: ^ml.Tensor,
}

// Inputs live in device memory (a persistent arena), so a micro measures its
// kernel, not the staging of a heap buffer. (The training workloads keep their
// params on the heap like the real programs do; stage 8 moves them.)
persist: mem.Dynamic_Arena

micro_setup :: proc() {
	ml.seed(3)
	ml.arena_init(&persist, 256 * mem.Megabyte)
	context.allocator = mem.dynamic_arena_allocator(&persist)
	micro.big = ml.randn({16 * 1024 * 1024}, 0, 1) // 64 MB in, 64 MB out
	micro.a = ml.randn({2048, 2048}, 0, 1)
	micro.b = ml.randn({2048, 2048}, 0, 1)
	micro.rows = ml.randn({4096, 1024}, 0, 1)
	micro.ln = ml.randn({16384, 256}, 0, 1)
}

stream_step :: proc() {ml.realize(ml.add(ml.mul(micro.big, ml.scalar(2)), ml.scalar(1)))}
gemm_step :: proc() {ml.realize(ml.matmul(micro.a, micro.b))}
softmax_step :: proc() {ml.realize(ml.softmax(micro.rows, -1))}
layernorm_step :: proc() {ml.realize(ml.layer_norm(micro.ln))}

WORKLOADS := []Workload {
	{"dt_fusion_step", "M4 train step, batch 1024 (sample + fwd + bwd + Adam)", 60, fusion_setup, fusion_step},
	{"dt_table1_step", "M3 train step, batch 1024", 60, table1_setup, table1_step},
	{"mlp_step", "MNIST MLP 784-128-10, batch 128, SGD", 300, mlp_setup, mlp_step},
	{"cnn_step", "MNIST CNN, batch 128, SGD", 40, cnn_setup, cnn_step},
	{"stream_64MB", "y = 2x + 1 over 16M floats", 30, micro_setup, stream_step},
	{"gemm_2048", "2048³ fp32 GEMM", 20, nil, gemm_step},
	{"softmax_4096x1024", "row softmax, fwd", 30, nil, softmax_step},
	{"layernorm_16384x256", "row LayerNorm (no affine), fwd", 30, nil, layernorm_step},
}

// ---- running -------------------------------------------------------------------

arena: mem.Dynamic_Arena

run_step :: proc(w: Workload) -> f64 {
	heap := context.allocator
	context.allocator = mem.dynamic_arena_allocator(&arena)
	t0 := time.tick_now()
	w.step()
	ml.backend.sync()
	ms := time.duration_milliseconds(time.tick_since(t0))
	context.allocator = heap
	mem.dynamic_arena_reset(&arena)
	return ms
}

run :: proc(w: Workload) -> (r: Result) {
	r.name = w.name
	warm := max(3, w.steps / 10)
	for _ in 0 ..< warm do run_step(w)
	ts := make([]f64, w.steps)
	for i in 0 ..< w.steps do ts[i] = run_step(w)
	slice.sort(ts)
	r.median_ms, r.p10_ms, r.p90_ms = ts[len(ts) / 2], ts[len(ts) / 10], ts[len(ts) * 9 / 10]

	ml.counters_reset()
	run_step(w)
	r.kernels = ml.counters.kernels

	ml.profile_begin()
	run_step(w)
	stats := ml.profile_end()
	by_kind: [ml.Kernel_Kind]Kind_Stat
	for s in stats {
		k := &by_kind[s.kind]
		k.kernels += 1
		k.ms += s.ms
		k.bytes += s.bytes
		k.flops += s.flops
	}
	prof := make([dynamic]Kind_Stat)
	for &k, kind in by_kind {
		if k.kernels == 0 do continue
		k.kind = fmt.aprintf("%v", kind)
		append(&prof, k)
	}
	slice.sort_by(prof[:], proc(a, b: Kind_Stat) -> bool { return a.ms > b.ms })
	r.profile = prof[:]
	return
}

machine_name :: proc() -> string {
	return fmt.aprintf("%s-%s-%s", strings.to_lower(fmt.tprintf("%v", ml.get_device())), ODIN_OS_STRING, ODIN_ARCH_STRING)
}

load_baseline :: proc(path: string) -> (rec: Record, ok: bool) {
	data, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil do return
	ok = json.unmarshal(data, &rec) == nil // before returning rec: return values are evaluated left to right
	return
}

find :: proc(rec: Record, name: string) -> (Result, bool) {
	for w in rec.workloads do if w.name == name do return w, true
	return {}, false
}

delta :: proc(now, before: f64) -> string {
	if before <= 0 do return ""
	return fmt.tprintf("%+.0f%%", 100 * (now - before) / before)
}

main :: proc() {
	ml.setup_from_env()
	ml.warn_if_unoptimized()
	ml.arena_init(&arena, 256 * mem.Megabyte)
	write_baseline := false
	filter := ""
	for a in os.args[1:] {
		if a == "baseline" do write_baseline = true
		else do filter = a
	}
	machine := machine_name()
	baseline_path := fmt.tprintf("bench/perf/baseline-%s.json", machine)
	base, have_base := load_baseline(baseline_path)
	now := time.now()
	y, mo, d := time.date(now)
	rec := Record{rev = GIT_REV, date = fmt.aprintf("%04d-%02d-%02d", y, int(mo), d), machine = machine}
	fmt.printfln("=== perf: %s, rev %s%s ===", machine, GIT_REV,
		have_base ? fmt.tprintf(" (vs baseline rev %s, %s)", base.rev, base.date) : " (no baseline for this machine)")

	results := make([dynamic]Result)
	for w in WORKLOADS {
		if w.setup != nil do w.setup() // micro setups are shared by the following micros
		if filter != "" && !strings.contains(w.name, filter) do continue
		r := run(w)
		append(&results, r)
		b, has := find(base, r.name)
		fmt.printfln("\n%-20s %s", r.name, w.note)
		fmt.printfln("  step   median % 8.3f ms %-5s  p10 % 8.3f  p90 % 8.3f   kernels %d %s",
			r.median_ms, has ? delta(r.median_ms, b.median_ms) : "", r.p10_ms, r.p90_ms, r.kernels,
			has && b.kernels != r.kernels ? fmt.tprintf("(was %d)", b.kernels) : "")
		total: f64
		for k in r.profile do total += k.ms
		for k in r.profile {
			gbs := k.ms > 0 ? f64(k.bytes) / k.ms / 1e6 : 0
			gfs := k.ms > 0 ? f64(k.flops) / k.ms / 1e6 : 0
			fmt.printfln("    %-10s % 5d kernels % 8.3f ms % 4.0f%%   % 7.1f GB/s % 9.1f GFLOP/s",
				k.kind, k.kernels, k.ms, total > 0 ? 100 * k.ms / total : 0, gbs, gfs)
		}
	}
	rec.workloads = results[:]

	line, merr := json.marshal(rec)
	if merr == nil {
		os.make_directory_all("build")
		f, ferr := os.open("build/perf.jsonl", {.Write, .Create, .Append})
		if ferr == nil {
			os.write(f, line)
			os.write(f, transmute([]u8)string("\n"))
			os.close(f)
		}
		if write_baseline {
			pretty, _ := json.marshal(rec, {pretty = true})
			if os.write_entire_file(baseline_path, pretty) == nil do fmt.printfln("\nbaseline written: %s", baseline_path)
		}
	}
	fmt.println("\n(profile ms: each kernel timed alone on the device; the step median is the real number)")
}
