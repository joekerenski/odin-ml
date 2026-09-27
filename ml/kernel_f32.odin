package ml

// ============================================================================
// CPU kernels for non-fused primitives: reduce, permute, cross-entropy.
// All buffers dense row-major. Elementwise math lives in fuse.odin.
// ============================================================================

import "core:math"

// ---- reduce ---------------------------------------------------------------

// dst[o, k] = Σ_r src[o, r, k]   (src viewed as [outer, red, inner])
reduce_block :: proc(dst, src: []f32, outer, red, inner: int) {
	for i in 0 ..< outer * inner do dst[i] = 0
	if inner == 1 {
		for o in 0 ..< outer {
			s: f32 = 0
			for v in src[o * red:(o + 1) * red] do s += v
			dst[o] = s
		}
		return
	}
	for o in 0 ..< outer {
		d := dst[o * inner:(o + 1) * inner]
		for r in 0 ..< red {
			row := src[(o * red + r) * inner:][:inner]
			for k in 0 ..< inner do d[k] += row[k]
		}
	}
}

// Sum over `axes` (kept as size 1). Each maximal run of adjacent reduced axes
// is one reduce_block pass, rightmost run first.
sum_kernel :: proc(out, a: []f32, shape: []i32, axes: []i32) {
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
		dst := more ? make([]f32, outer * inner) : out
		reduce_block(dst, cur, outer, r, inner)
		cur = dst
	}
	if raw_data(cur) != raw_data(out) do copy(out, cur)
}

// ---- permute --------------------------------------------------------------

// out.shape[i] = shape[order[i]]
permute_kernel :: proc(out, a: []f32, shape: []i32, order: []i32) {
	nd := len(shape)
	if nd == 2 && order[0] == 1 {
		M, N := int(shape[0]), int(shape[1])
		for i in 0 ..< M do for j in 0 ..< N do out[j * M + i] = a[i * N + j]
		return
	}
	out_shape, src_stride: [MAX_DIMS]i32
	for o, i in order {
		out_shape[i] = shape[o]
		src_stride[i] = stride_of(shape, int(o))
	}
	idx: [MAX_DIMS]i32
	for f in 0 ..< len(out) {
		unravel_index(i32(f), out_shape[:nd], idx[:])
		off: i32 = 0
		for i in 0 ..< nd do off += idx[i] * src_stride[i]
		out[f] = a[off]
	}
}

// ---- cross-entropy --------------------------------------------------------

// Softmax of one row into p, returns log Σ exp(row).
softmax_row :: proc(p, row: []f32) -> f32 {
	m := row[0]
	for v in row do m = max(m, v)
	s: f32 = 0
	for v, c in row {
		p[c] = math.exp(v - m)
		s += p[c]
	}
	for c in 0 ..< len(p) do p[c] /= s
	return m + math.ln(s)
}

// mean_b( logsumexp(logits[b]) - logits[b, label_b] )
cross_entropy_f32 :: proc(logits: []f32, B, C: i32, labels: []u8) -> f32 {
	p := make([]f32, C)
	defer delete(p)
	total: f32 = 0
	for b in 0 ..< int(B) {
		row := logits[b * int(C):][:C]
		total += softmax_row(p, row) - row[labels[b]]
	}
	return total / f32(B)
}

// dlogits = g/B * (softmax - onehot)
cross_entropy_backward :: proc(dx: []f32, g: f32, logits: []f32, B, C: i32, labels: []u8) {
	scale := g / f32(B)
	for b in 0 ..< int(B) {
		row := logits[b * int(C):][:C]
		d := dx[b * int(C):][:C]
		softmax_row(d, row)
		d[labels[b]] -= 1
		for c in 0 ..< int(C) do d[c] *= scale
	}
}
