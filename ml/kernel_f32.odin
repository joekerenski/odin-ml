package ml

// ============================================================================
// CPU kernels for non-fused primitives: reduce, permute.
// All buffers dense row-major. Elementwise math lives in fuse.odin.
// ============================================================================

import "core:math"

// ---- reduce (Sum / ReduceMax) --------------------------------------------

Reduce_Job :: struct {
	dst, src:           []f32,
	outer, red, inner:  int,
	by_outer:           bool, // split rows (else columns)
}

// dst[o, k] = reduce_r src[o, r, k]   (src viewed as [outer, red, inner])
reduce_block :: proc(dst, src: []f32, outer, red, inner: int, $op: Op) {
	// Split rows across threads. Few rows (e.g. a bias grad [5120,64] → [1,64]):
	// one thread; splitting columns there costs more in cache misses than it saves.
	job := Reduce_Job{dst, src, outer, red, inner, true}
	if outer < 8 {
		if red >= 64 && red * inner * outer >= 2 * PAR_GRAIN {
			reduce_split(dst, src, outer, red, inner, op)
			return
		}
		reduce_rows(dst, src, outer, red, inner, 0, inner, op)
		return
	}
	parallel_for(outer, max(1, PAR_GRAIN / max(red * inner, 1)), reduce_range_proc(op), &job)
}

@(private)
reduce_range_proc :: proc($op: Op) -> Range_Proc {
	return proc(data: rawptr, lo, hi: int) {
		j := (^Reduce_Job)(data)
		if j.by_outer {
			reduce_rows(j.dst[lo * j.inner:hi * j.inner], j.src[lo * j.red * j.inner:hi * j.red * j.inner], hi - lo, j.red, j.inner, 0, j.inner, op)
		} else {
			reduce_rows(j.dst, j.src, j.outer, j.red, j.inner, lo, hi, op)
		}
	}
}

// Few rows, long reduced axis (a bias grad [1, 4096, 64] → [1, 1, 64], a full
// sum): each thread reduces a contiguous chunk of the reduced axis into its
// own partial, then the partials are combined.
@(private)
Reduce_Split_Job :: struct {
	partial, src:             []f32,
	outer, red, inner, chunk: int,
}

@(private)
reduce_split :: proc(dst, src: []f32, outer, red, inner: int, $op: Op) {
	chunks := min(red / 32, 4 * thread_count())
	chunk := (red + chunks - 1) / chunks
	chunks = (red + chunk - 1) / chunk
	rows := outer * inner
	job := Reduce_Split_Job{make([]f32, chunks * rows, scratch()), src, outer, red, inner, chunk}
	defer delete(job.partial, scratch())
	parallel_for(chunks, 1, proc(data: rawptr, lo, hi: int) {
		using j := (^Reduce_Split_Job)(data)
		for c in lo ..< hi {
			r0, r1 := c * chunk, min(red, (c + 1) * chunk)
			for o in 0 ..< outer {
				part := partial[(c * outer + o) * inner:][:inner]
				reduce_rows(part, src[(o * red + r0) * inner:(o * red + r1) * inner], 1, r1 - r0, inner, 0, inner, op)
			}
		}
	}, &job)
	copy(dst[:rows], job.partial[:rows])
	for c in 1 ..< chunks {
		p := job.partial[c * rows:][:rows]
		for i in 0 ..< rows {
			when op == .Sum do dst[i] += p[i]
			else do dst[i] = max(dst[i], p[i])
		}
	}
}

// Rows [0, outer), columns [k0, k1) of the [outer, red, inner] view.
@(private)
reduce_rows :: proc(dst, src: []f32, outer, red, inner, k0, k1: int, $op: Op) {
	init: f32 = op == .Sum ? 0 : math.inf_f32(-1)
	for o in 0 ..< outer do for k in k0 ..< k1 do dst[o * inner + k] = init
	if inner == 1 && k0 == 0 && k1 == 1 {
		for o in 0 ..< outer {
			acc := init
			for v in src[o * red:(o + 1) * red] {
				when op == .Sum do acc += v
				else do acc = max(acc, v)
			}
			dst[o] = acc
		}
		return
	}
	for o in 0 ..< outer {
		d := dst[o * inner:(o + 1) * inner]
		for r in 0 ..< red {
			row := src[(o * red + r) * inner:][:inner]
			for k in k0 ..< k1 {
				when op == .Sum do d[k] += row[k]
				else do d[k] = max(d[k], row[k])
			}
		}
	}
}

// out.shape[i] = shape[order[i]]. Walks the output in order with an odometer
// (no divisions); when the last axis stays last, copies contiguous runs.
// Parts of the output run in parallel, each starting its odometer at lo.
Permute_Job :: struct {
	out, a:                []f32,
	nd, outer_nd, run:     int,
	out_shape, src_stride: [MAX_DIMS]int,
}

permute_kernel :: proc(out, a: []f32, shape: []i32, order: []i32) {
	dims, st: [MAX_DIMS]int
	for o, i in order {
		dims[i] = int(shape[o])
		st[i] = int(stride_of(shape, int(o)))
	}
	strided_copy(out, a, dims[:len(order)], st[:len(order)])
}

// out[i] = a[offset(i)] over dims, offset by per-dim strides (permutes, views).
strided_copy :: proc(out, a: []f32, dims: []int, strides: []int) {
	job := Permute_Job{out = out, a = a, nd = len(dims)}
	copy(job.out_shape[:], dims)
	copy(job.src_stride[:], strides)
	job.run, job.outer_nd = 1, job.nd
	if job.nd > 0 && strides[job.nd - 1] == 1 {
		job.run, job.outer_nd = dims[job.nd - 1], job.nd - 1
	}
	rows := len(out) / max(job.run, 1)
	parallel_for(rows, max(1, PAR_GRAIN / max(job.run, 1)), permute_range, &job)
}

permute_range :: proc(data: rawptr, lo, hi: int) {
	using job := (^Permute_Job)(data)
	idx: [MAX_DIMS]int
	off, r := 0, lo
	for d := outer_nd - 1; d >= 0; d -= 1 { // odometer position of row lo
		idx[d] = r % out_shape[d]
		r /= out_shape[d]
		off += idx[d] * src_stride[d]
	}
	for row in lo ..< hi {
		f := row * run
		for k in 0 ..< run do out[f + k] = a[off + k]
		for d := outer_nd - 1; d >= 0; d -= 1 {
			idx[d] += 1
			off += src_stride[d]
			if idx[d] < out_shape[d] do break
			off -= src_stride[d] * out_shape[d]
			idx[d] = 0
		}
	}
}
