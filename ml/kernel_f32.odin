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

// Reduce over `axes` (kept as size 1). Each maximal run of adjacent reduced
// axes is one reduce_block pass, rightmost run first.
reduce_kernel :: proc(op: Op, out, a: []f32, shape: []i32, axes: []i32) {
	red: [MAX_DIMS]bool
	for ax in axes do red[ax] = true
	cur := a
	cur_shape: [MAX_DIMS]i32
	copy(cur_shape[:], shape)
	nd := len(shape)

	d := nd - 1
	for d >= 0 {
		if !red[d] || cur_shape[d] == 1 {
			d -= 1
			continue
		}
		hi := d + 1
		for d >= 0 && red[d] do d -= 1
		lo := d + 1
		outer := int(numel(cur_shape[:lo]))
		r := int(numel(cur_shape[lo:hi]))
		inner := int(numel(cur_shape[hi:nd]))
		for k in lo ..< hi do cur_shape[k] = 1
		// more runs to the left? reduce into a temp, else straight into out
		more := false
		for k in 0 ..< lo do if red[k] && cur_shape[k] != 1 do more = true
		dst := more ? make([]f32, outer * inner, scratch()) : out
		if op == .Sum {
			reduce_block(dst, cur, outer, r, inner, .Sum)
		} else {
			reduce_block(dst, cur, outer, r, inner, .ReduceMax)
		}
		if raw_data(cur) != raw_data(a) do delete(cur, scratch())
		cur = dst
	}
	if raw_data(cur) != raw_data(out) {
		copy(out, cur)
		if raw_data(cur) != raw_data(a) do delete(cur, scratch())
	}
}

// ---- permute --------------------------------------------------------------

// out.shape[i] = shape[order[i]]. Walks the output in order with an odometer
// (no divisions); when the last axis stays last, copies contiguous runs.
// Parts of the output run in parallel, each starting its odometer at lo.
Permute_Job :: struct {
	out, a:                []f32,
	nd, outer_nd, run:     int,
	out_shape, src_stride: [MAX_DIMS]int,
}

permute_kernel :: proc(out, a: []f32, shape: []i32, order: []i32) {
	job := Permute_Job{out = out, a = a, nd = len(shape)}
	for o, i in order {
		job.out_shape[i] = int(shape[o])
		job.src_stride[i] = int(stride_of(shape, int(o)))
	}
	job.run, job.outer_nd = 1, job.nd
	if int(order[job.nd - 1]) == job.nd - 1 {
		job.run, job.outer_nd = job.out_shape[job.nd - 1], job.nd - 1
	}
	rows := len(out) / job.run
	parallel_for(rows, max(1, PAR_GRAIN / job.run), permute_range, &job)
}

@(private)
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
