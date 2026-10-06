package ml

// ============================================================================
// Kernel IR → GPU source, one renderer for Metal (MSL) and CUDA (NVRTC C++).
// The dialects differ only in spelling: signatures, thread indices, shared
// memory, barriers, math names. Both backends launch the same plans with the
// same parameter block, so a change here is exercised by Metal parity tests
// and lands on CUDA as the same program.
//
// Plans (same heuristics as the fixed kernels they replace):
//   Elementwise    one thread per index point
//   Reduce_Thread  one thread per output point, loops over the reduced run
//   Reduce_Group   one 256-thread group per output point (long runs, few rows)
//   split          long column reductions: Reduce_Thread over [O·S, R/S, I]
//                  into a partial, then a plain reduction over S
//
// Parameter block (u32), shared by every program:
//   p[0]             points: elementwise index points / reduce output points
//   p[1]             nd
//   p[2 + d]         dims[d]
//   LOAD_BASE + L·LOAD_WORDS:  offset, n, inner, strides[MAX_DIMS]
//   CONST_BASE + c   the c-th Const node's f32 bits
// ============================================================================

import "core:fmt"
import "core:strings"

GPU_PARAMS :: 160
GPU_GROUP :: 256
@(private)
LOAD_BASE :: 2 + MAX_DIMS
@(private)
LOAD_WORDS :: 3 + MAX_DIMS

GPU_Dialect :: enum u8 {
	Metal,
	CUDA,
}

GPU_Variant :: enum u8 {
	Elementwise,
	Reduce_Thread,
	Reduce_Group,
}

GPU_Plan :: struct {
	variant: GPU_Variant,
	split:   int, // > 1: two-pass column reduction
	threads: int, // threads to launch (all variants: total; groups of GPU_GROUP)
}

gpu_plan :: proc(k: ^Kernel) -> (p: GPU_Plan) {
	if !kernel_has_reduce(k) {
		p.variant, p.threads = .Elementwise, kernel_points(k)
		return
	}
	assert(k.nd == 3 && k.red_lo == 1 && k.red_hi == 2, "gpu: reductions are [outer, r, inner] for now")
	rows, r, inner := k.dims[0] * k.dims[2], k.dims[1], k.dims[2]
	single := k.n_nodes == 2 // Load → Reduce: the partial pass may re-tile it
	if single && inner >= 16 && r >= 512 && rows < 8192 {
		for s in ([]int{64, 32, 16, 8}) do if r % s == 0 && p.split == 0 do p.split = s
	}
	switch {
	case p.split > 1:
		p.variant, p.threads = .Reduce_Thread, k.dims[0] * p.split * inner
	case r >= 128 && rows < 8192:
		p.variant, p.threads = .Reduce_Group, rows * GPU_GROUP
	case:
		p.variant, p.threads = .Reduce_Thread, rows
	}
	return
}

// Fill the parameter block; returns the words used.
gpu_params :: proc(k: ^Kernel, p: ^[GPU_PARAMS]u32) -> int {
	p[0] = u32(kernel_has_reduce(k) ? kernel_out_points(k) : kernel_points(k))
	p[1] = u32(k.nd)
	for d in 0 ..< k.nd do p[2 + d] = u32(k.dims[d])
	for v, l in k.loads[:k.n_loads] {
		b := LOAD_BASE + l * LOAD_WORDS
		p[b], p[b + 1], p[b + 2] = u32(v.offset), u32(v.n), u32(v.inner)
		for d in 0 ..< k.nd do p[b + 3 + d] = u32(v.strides[d])
	}
	c := LOAD_BASE + k.n_loads * LOAD_WORDS
	for n in k.nodes[:k.n_nodes] {
		if n.kind != .Const do continue
		p[c] = transmute(u32)n.value
		c += 1
	}
	assert(c <= GPU_PARAMS)
	return c
}

// Parameters for the two passes of a split reduction (plan.split = S):
// pass A runs k's own program over [O·S, R/S, I] into a dense partial; pass B
// reduces the partial [O, S, I] (program: kernel_reduce_run's shape).
gpu_split_params :: proc(k: ^Kernel, S: int) -> (pa, pb: [GPU_PARAMS]u32, na, nb: int) {
	O, R, I := k.dims[0], k.dims[1], k.dims[2]
	a := k^
	a.dims[0], a.dims[1] = O * S, R / S
	a.loads[0].strides[0], a.loads[0].strides[1], a.loads[0].strides[2] = (R / S) * I, I, 1
	na = gpu_params(&a, &pa)
	b := kernel_reduce_run(k.nodes[kernel_reduce_node(k)].op, nil, nil, O, S, I)
	nb = gpu_params(&b, &pb)
	return
}

// ---- source ---------------------------------------------------------------------

@(private = "file")
Render :: struct {
	b:       strings.Builder,
	d:       GPU_Dialect,
	k:       ^Kernel,
	consts:  [MAX_KERNEL_NODES]int, // node → const index
}

@(private = "file")
P :: proc(r: ^Render, i: int) -> string {
	return r.d == .Metal ? fmt.tprintf("p[%d]", i) : fmt.tprintf("p.v[%d]", i)
}

@(private = "file")
alu :: proc(r: ^Render, op: Op, a, b: string) -> string {
	cuda := r.d == .CUDA
	#partial switch op {
	case .Add: return fmt.tprintf("%s + %s", a, b)
	case .Sub: return fmt.tprintf("%s - %s", a, b)
	case .Mul: return fmt.tprintf("%s * %s", a, b)
	case .Div: return fmt.tprintf("%s / %s", a, b)
	case .Max: return fmt.tprintf("(%s > %s ? %s : %s)", a, b, a, b)
	case .CmpLt: return fmt.tprintf("(%s < %s ? 1.0f : 0.0f)", a, b)
	case .Neg: return fmt.tprintf("-%s", a)
	case .Exp: return fmt.tprintf("%s(%s)", cuda ? "expf" : "exp", a)
	case .Log: return fmt.tprintf("%s(%s)", cuda ? "logf" : "log", a)
	case .Sqrt: return fmt.tprintf("%s(%s)", cuda ? "sqrtf" : "sqrt", a)
	case .Expand: return a
	}
	fmt.panicf("gpu render: %v is not elementwise", op)
}

// Index expression of load l. Elementwise: by its mode over point i.
// Reductions: [o, r, k] coordinates (r = "0" in the epilogue).
@(private = "file")
load_index :: proc(r: ^Render, l: int, variant: GPU_Variant, rr: string) -> string {
	v := r.k.loads[l]
	B := LOAD_BASE + l * LOAD_WORDS
	off, n, inner := P(r, B), P(r, B + 1), P(r, B + 2)
	if variant != .Elementwise {
		return fmt.tprintf("%s + o * %s + %s * %s + kk * %s", off, P(r, B + 3), rr, P(r, B + 4), P(r, B + 5))
	}
	switch v.mode {
	case .Direct: return fmt.tprintf("%s + i", off)
	case .Scalar: return off
	case .Row: return fmt.tprintf("%s + i %% %s", off, n)
	case .Col: return fmt.tprintf("%s + i / %s", off, inner)
	case .Block: return fmt.tprintf("%s + (i / %s) %% %s", off, inner, n)
	case .Generic: return fmt.tprintf("g%d", l) // computed by an unravel block
	}
	return ""
}

@(private = "file")
emit_nodes :: proc(r: ^Render, from, to: int, variant: GPU_Variant, rr: string, indent: string) {
	k := r.k
	for i in from ..< to {
		n := k.nodes[i]
		switch n.kind {
		case .Load:
			v := k.loads[n.load]
			if variant == .Elementwise && v.mode == .Generic {
				B := LOAD_BASE + n.load * LOAD_WORDS
				fmt.sbprintf(&r.b, "%suint g%d = %s; {{ uint rem = i;\n", indent, n.load, P(r, B))
				fmt.sbprintf(&r.b, "%s  for (int d = %d; d >= 0; d--) {{ uint c = rem %% %s; rem /= %s; g%d += c * %s; }} }}\n",
					indent, k.nd - 1, fmt.tprintf(r.d == .Metal ? "p[2 + d]" : "p.v[2 + d]"), fmt.tprintf(r.d == .Metal ? "p[2 + d]" : "p.v[2 + d]"),
					n.load, fmt.tprintf(r.d == .Metal ? "p[%d + d]" : "p.v[%d + d]", B + 3))
			}
			fmt.sbprintf(&r.b, "%sfloat v%d = b%d[%s];\n", indent, i, v.buf, load_index(r, n.load, variant, rr))
		case .Const:
			c := LOAD_BASE + k.n_loads * LOAD_WORDS + r.consts[i]
			fmt.sbprintf(&r.b, "%sfloat v%d = %s(%s);\n", indent, i, r.d == .Metal ? "as_type<float>" : "__uint_as_float", P(r, c))
		case .ALU:
			a, b := fmt.tprintf("v%d", n.a), fmt.tprintf("v%d", n.b)
			fmt.sbprintf(&r.b, "%sfloat v%d = %s;\n", indent, i, alu(r, n.op, a, b))
		case .Reduce:
		}
	}
}

@(private = "file")
reduce_init :: proc(r: ^Render, op: Op) -> string {
	if op == .Sum do return "0.0f"
	return r.d == .Metal ? "-INFINITY" : "NEG_INF"
}

@(private = "file")
reduce_combine :: proc(op: Op, acc, v: string) -> string {
	if op == .Sum do return fmt.tprintf("%s + %s", acc, v)
	return fmt.tprintf("(%s > %s ? %s : %s)", v, acc, v, acc)
}

// Source of k under a plan variant; the entry point is "k_main".
gpu_source :: proc(k: ^Kernel, variant: GPU_Variant, d: GPU_Dialect) -> string {
	r := Render{b = strings.builder_make(context.temp_allocator), d = d, k = k}
	c := 0
	for n, i in k.nodes[:k.n_nodes] do if n.kind == .Const {
		r.consts[i] = c
		c += 1
	}
	b := &r.b
	metal := d == .Metal
	if metal {
		strings.write_string(b, "#include <metal_stdlib>\nusing namespace metal;\nkernel void k_main(\n")
		for j in 0 ..< k.n_bufs {
			fmt.sbprintf(b, "    device %sfloat* b%d [[buffer(%d)]],\n", j < k.n_in ? "const " : "", j, j)
		}
		strings.write_string(b, "    constant uint* p [[buffer(30)]],\n")
		switch variant {
		case .Elementwise: strings.write_string(b, "    uint i [[thread_position_in_grid]]) {\n")
		case .Reduce_Thread: strings.write_string(b, "    uint t [[thread_position_in_grid]]) {\n")
		case .Reduce_Group: strings.write_string(b, "    uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {\n")
		}
	} else {
		strings.write_string(b, "typedef unsigned int uint;\n#define NEG_INF __int_as_float(0xff800000)\nextern \"C\" __global__ void k_main(\n")
		for j in 0 ..< k.n_bufs do fmt.sbprintf(b, "    %sfloat* b%d,\n", j < k.n_in ? "const " : "", j)
		strings.write_string(b, "    const P p) {\n")
		switch variant {
		case .Elementwise: fmt.sbprintf(b, "    uint i = blockIdx.x * %du + threadIdx.x;\n", GPU_GROUP)
		case .Reduce_Thread: fmt.sbprintf(b, "    uint t = blockIdx.x * %du + threadIdx.x;\n", GPU_GROUP)
		case .Reduce_Group: strings.write_string(b, "    uint g = blockIdx.x, t = threadIdx.x;\n")
		}
	}

	switch variant {
	case .Elementwise:
		fmt.sbprintf(b, "    if (i >= %s) return;\n", P(&r, 0))
		emit_nodes(&r, 0, k.n_nodes, variant, "", "    ")
		for s in k.stores[:k.n_stores] do fmt.sbprintf(b, "    b%d[i] = v%d;\n", s.buf, s.node)

	case .Reduce_Thread, .Reduce_Group:
		red := kernel_reduce_node(k)
		op := k.nodes[red].op
		group := variant == .Reduce_Group
		R, I := P(&r, 3), P(&r, 4) // dims[1], dims[2]
		if group {
			fmt.sbprintf(b, "    uint o = g / %s, kk = g %% %s;\n", I, I)
			strings.write_string(b, metal ? "    threadgroup float sh[256];\n" : "    __shared__ float sh[256];\n")
			fmt.sbprintf(b, "    float acc = %s;\n    for (uint r = t; r < %s; r += 256) {{\n", reduce_init(&r, op), R)
		} else {
			fmt.sbprintf(b, "    if (t >= %s) return;\n    uint o = t / %s, kk = t %% %s;\n", P(&r, 0), I, I)
			fmt.sbprintf(b, "    float acc = %s;\n    for (uint r = 0; r < %s; r++) {{\n", reduce_init(&r, op), R)
		}
		emit_nodes(&r, 0, red, variant, "r", "        ")
		fmt.sbprintf(b, "        acc = %s;\n    }}\n", reduce_combine(op, "acc", fmt.tprintf("v%d", k.nodes[red].a)))
		out := "t"
		indent := "    "
		if group {
			barrier := metal ? "threadgroup_barrier(mem_flags::mem_threadgroup);" : "__syncthreads();"
			fmt.sbprintf(b, "    sh[t] = acc;\n    %s\n", barrier)
			fmt.sbprintf(b, "    for (uint s = 128; s > 0; s >>= 1) {{\n        if (t < s) sh[t] = %s;\n        %s\n    }}\n",
				reduce_combine(op, "sh[t]", "sh[t + s]"), barrier)
			strings.write_string(b, "    if (t != 0) return;\n    acc = sh[0];\n")
			out = "g"
		}
		fmt.sbprintf(b, "%sfloat v%d = acc;\n", indent, red)
		emit_nodes(&r, red + 1, k.n_nodes, variant, "0", indent)
		for s in k.stores[:k.n_stores] do fmt.sbprintf(b, "%sb%d[%s] = v%d;\n", indent, s.buf, out, s.node)
	}
	strings.write_string(b, "}\n")
	return strings.to_string(r.b)
}
