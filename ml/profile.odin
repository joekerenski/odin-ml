package ml

// ============================================================================
// Per-kernel profile: what every kernel of a realize did and how long it took.
//
//   ml.profile_begin()
//   ml.backward(loss)                // any realizes
//   stats := ml.profile_end()        // one Kernel_Stat per kernel, launch order
//
// The scheduler fills in kind, bytes and FLOPs; the backend supplies the time.
// While profiling, GPU backends run each kernel alone (own command buffer /
// between events), like ML_DEBUG=2: right for the breakdown, too slow for the
// step time — time steps without profiling.
// ============================================================================

import "core:strings"

Kernel_Kind :: enum u8 {
	Fused,        // generated elementwise group
	Reduce,
	Permute,
	GEMM,
	Conv_Pool,    // conv / pool and their backward (on GPU devices: host code after a sync)
}

Kernel_Stat :: struct {
	kind:    Kernel_Kind,
	variant: string, // the backend's kernel (e.g. "matmul_sg"), if it says
	ops:     int,    // fused: nodes in the group
	bytes:   i64,    // buffers read + written (broadcast inputs at their own size)
	flops:   i64,
	ms:      f64,    // device time; CPU: wall time
}

profiling: bool

@(private)
profile_stats: [dynamic]Kernel_Stat

// Time every kernel: profiling, or ML_DEBUG ≥ 2 (which also prints them).
kernel_timing :: #force_inline proc "contextless" () -> bool {
	return profiling || debug_level >= 2
}

profile_begin :: proc() {
	backend.sync()
	for s in profile_stats do delete(s.variant, scratch())
	if profile_stats == nil do profile_stats = make([dynamic]Kernel_Stat, scratch())
	clear(&profile_stats)
	profiling = true
}

// The kernels since profile_begin; valid until the next profile_begin.
profile_end :: proc() -> []Kernel_Stat {
	backend.sync()
	profiling = false
	return profile_stats[:]
}

// Scheduler: a kernel is about to run.
@(private)
profile_open :: proc(kind: Kernel_Kind, bytes, flops: i64, ops := 1) {
	if !profiling do return
	append(&profile_stats, Kernel_Stat{kind = kind, ops = ops, bytes = bytes, flops = flops})
}

// Scheduler: wall time of a kernel that ran on the host.
@(private)
profile_set_wall :: proc(ms: f64) {
	if !profiling || len(profile_stats) == 0 do return
	profile_stats[len(profile_stats) - 1].ms = ms
}

// Backend: device time of a dispatch belonging to the open kernel (a kernel may
// take several dispatches, e.g. split-K + combine). label's first word names it.
profile_add_ms :: proc(ms: f64, label: string) {
	if !profiling || len(profile_stats) == 0 do return
	s := &profile_stats[len(profile_stats) - 1]
	s.ms += ms
	if s.variant == "" && label != "" {
		name := label
		if i := strings.index_byte(label, ' '); i >= 0 do name = label[:i]
		s.variant = strings.clone(name, scratch())
	}
}
