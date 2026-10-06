package ml

// ============================================================================
// CPU elementwise kernels (kernel_cpu.odin builds the job from a Kernel). The
// program (insns over slots) runs over the output in CHUNKs:
//
//   slots[0..n_in)      each input's values for this chunk (broadcast by load
//                       mode; Direct inputs are just pointers, no copy)
//   slots[n_in + j]     result of insn j (one per group node, topo order)
//   stores              results someone still needs go straight to their buffer
//
// Per chunk, each insn is one tight SIMD loop with the op fixed at compile
// time, so dispatch costs once per CHUNK elements, not once per vector.
// ============================================================================

import "base:intrinsics"
import "core:math"
import "core:simd"

MAX_FUSED_INSNS :: 16
MAX_FUSED_INPUTS :: 12 // buffers + constants (Metal: inputs + stores must stay below buffer index 30)
MAX_FUSED_SLOTS :: MAX_FUSED_INPUTS + MAX_FUSED_INSNS

// How input element i of the output maps into an input buffer.
Load_Mode :: enum {
	Direct,  // data[i]
	Scalar,  // data[0]
	Row,     // data[i % n]            [B,N] ← [N]
	Col,     // data[i / inner]        [B,N] ← [B,1]
	Block,   // data[(i / inner) % n]  [N,C,H,W] ← [1,C,1,1]
	Generic, // unravel + broadcast index
	Const,   // a scalar constant (Const node): passed as a value, no buffer
}

Fused_In :: struct {
	mode:    Load_Mode,
	data:    []f32,
	n:       int,
	inner:   int,
	strides: [MAX_DIMS]int, // .Generic: per output dim (0 = broadcast)
	value:   f32,           // .Const
}

Fused_Insn :: struct {
	op:   Op,
	a, b: int, // slot indices
}

Fused_Store :: struct {
	slot: int,
	data: []f32,
}

CHUNK :: 256
PAR_GRAIN :: 8 * 1024 // elements per thread part (20 threads: a [1024, 4, 64] tensor in 32 parts), below this: one thread

Fused_Job :: struct {
	inputs:    []Fused_In,
	insns:     []Fused_Insn,
	stores:    []Fused_Store,
	out_shape: []int,
}

// CPU backend: split the output across cores.
run_fused_kernel :: proc(job: ^Fused_Job) {
	n := 1
	for d in job.out_shape do n *= d
	parallel_for(n, PAR_GRAIN, run_fused_range, job)
}

// Output elements [lo, hi), CHUNK at a time.
run_fused_range :: proc(data: rawptr, lo, hi: int) {
	using job := (^Fused_Job)(data)
	n_in := len(inputs)
	buf: [MAX_FUSED_SLOTS][CHUNK]f32 // chunk scratch (L1-resident)
	ptr: [MAX_FUSED_SLOTS][^]f32 // where each slot's chunk values live
	store_of: [MAX_FUSED_SLOTS]int // slot → store index + 1 (0: not stored)
	for st, k in stores do store_of[st.slot] = k + 1

	for inp, k in inputs {
		if inp.mode == .Scalar || inp.mode == .Const {
			v := inp.mode == .Const ? inp.value : inp.data[0]
			for &x in buf[k] do x = v
			ptr[k] = &buf[k][0]
		}
	}
	// Generic inputs: walk the output with an odometer, each input's offset
	// following its broadcast strides (0 on broadcast dims). No divisions per
	// element.
	nd := len(out_shape)
	gstride: [MAX_FUSED_SLOTS][MAX_DIMS]int
	goff: [MAX_FUSED_SLOTS]int
	has_generic := false
	for inp, k in inputs {
		if inp.mode != .Generic do continue
		has_generic = true
		gstride[k] = inp.strides
	}
	idx: [MAX_DIMS]int
	if has_generic {
		r := lo
		for d := nd - 1; d >= 0; d -= 1 {
			idx[d] = r % out_shape[d]
			r /= out_shape[d]
		}
		for inp, k in inputs {
			if inp.mode != .Generic do continue
			for d in 0 ..< nd do goff[k] += idx[d] * gstride[k][d]
		}
	}
	for base := lo; base < hi; base += CHUNK {
		m := min(CHUNK, hi - base)
		for inp, k in inputs {
			b := &buf[k]
			switch inp.mode {
			case .Scalar, .Const:
			case .Direct: ptr[k] = &inp.data[base]
			case .Row:
				j := base % inp.n
				for i in 0 ..< m {
					b[i] = inp.data[j]
					j += 1
					if j == inp.n do j = 0
				}
				ptr[k] = &b[0]
			case .Col, .Block:
				// runs of `inner` equal values: one division per chunk
				q, r := base / inp.inner, base % inp.inner
				if inp.mode == .Block do q %= inp.n
				for i := 0; i < m; {
					run := min(inp.inner - r, m - i)
					v := inp.data[q]
					for x in i ..< i + run do b[x] = v
					i += run
					r = 0
					q += 1
					if inp.mode == .Block && q == inp.n do q = 0
				}
				ptr[k] = &b[0]
			case .Generic:
				// filled below, all Generic inputs in one odometer pass
			}
		}
		if has_generic {
			for i in 0 ..< m {
				for inp, k in inputs do if inp.mode == .Generic do buf[k][i] = inp.data[goff[k]]
				for d := nd - 1; d >= 0; d -= 1 {
					idx[d] += 1
					for inp, k in inputs do if inp.mode == .Generic do goff[k] += gstride[k][d]
					if idx[d] < out_shape[d] do break
					for inp, k in inputs do if inp.mode == .Generic do goff[k] -= gstride[k][d] * out_shape[d]
					idx[d] = 0
				}
			}
			for inp, k in inputs do if inp.mode == .Generic do ptr[k] = &buf[k][0]
		}
		for insn, j in insns {
			slot := n_in + j
			dst: [^]f32 = &buf[slot][0]
			if st := store_of[slot]; st > 0 do dst = &stores[st - 1].data[base]
			run_chunk(insn.op, dst, ptr[insn.a], ptr[insn.b], m)
			ptr[slot] = dst
		}
	}
}

// dst[i] = op(a[i], b[i]) for one chunk. b is ignored by unary ops.
run_chunk :: proc(op: Op, dst, a, b: [^]f32, m: int) {
	#partial switch op {
	case .Add: chunk_loop(dst, a, b, m, .Add)
	case .Sub: chunk_loop(dst, a, b, m, .Sub)
	case .Mul: chunk_loop(dst, a, b, m, .Mul)
	case .Div: chunk_loop(dst, a, b, m, .Div)
	case .Max: chunk_loop(dst, a, b, m, .Max)
	case .CmpLt: chunk_loop(dst, a, b, m, .CmpLt)
	case .Neg: chunk_loop(dst, a, b, m, .Neg)
	case .Sqrt: chunk_loop(dst, a, b, m, .Sqrt)
	case .Expand: chunk_loop(dst, a, b, m, .Expand)
	case .Exp: for i in 0 ..< m do dst[i] = math.exp(a[i])
	case .Log: for i in 0 ..< m do dst[i] = math.ln(a[i])
	case: panic("run_chunk: not an elementwise op")
	}
}

chunk_loop :: #force_inline proc(dst, a, b: [^]f32, m: int, $op: Op) {
	i := 0
	for ; i + 4 <= m; i += 4 {
		va := intrinsics.unaligned_load((^simd.f32x4)(&a[i]))
		vb := intrinsics.unaligned_load((^simd.f32x4)(&b[i]))
		intrinsics.unaligned_store((^simd.f32x4)(&dst[i]), fused_eval_simd(op, va, vb))
	}
	for ; i < m; i += 1 do dst[i] = fused_eval(op, a[i], b[i])
}

fused_eval :: #force_inline proc(op: Op, a, b: f32) -> f32 {
	#partial switch op {
	case .Add: return a + b
	case .Sub: return a - b
	case .Mul: return a * b
	case .Div: return a / b
	case .Max: return a > b ? a : b
	case .CmpLt: return a < b ? 1 : 0
	case .Neg: return -a
	case .Sqrt: return math.sqrt(a)
	case .Expand: return a
	}
	panic("fused_eval: not a simd ewise op")
}

fused_eval_simd :: #force_inline proc(op: Op, a, b: simd.f32x4) -> simd.f32x4 {
	#partial switch op {
	case .Add: return simd.add(a, b)
	case .Sub: return simd.sub(a, b)
	case .Mul: return simd.mul(a, b)
	case .Div: return simd.div(a, b)
	case .Max: return simd.max(a, b)
	case .CmpLt: return simd.select(simd.lanes_lt(a, b), simd.f32x4(1), simd.f32x4(0))
	case .Neg: return simd.neg(a)
	case .Sqrt: return simd.sqrt(a)
	case .Expand: return a
	}
	panic("fused_eval_simd: not a simd ewise op")
}
