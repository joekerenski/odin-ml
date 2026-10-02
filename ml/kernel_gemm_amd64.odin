package ml

// ============================================================================
// x86-64 GEMM: AVX2 + FMA on all cores. C[M,N] = op(A) @ op(B).
//
// The kernels are compiled for AVX2/FMA per procedure (enable_target_feature)
// and picked at runtime from CPUID, so a plain generic x86-64 build is fast
// too (without FMA, LLVM lowers each fused_mul_add lane to a libm call).
//
// Packing: op(A) into 6-row panels laid out [k][6], op(B) into 16-column
// panels [k][16], edges zero-padded. Transposes cost nothing extra: they are
// just the packing's read order. Micro-kernel: a 6×16 block of C in 12 ymm
// accumulators; per k, 6 broadcasts of A and 2 loads of B feed 12 FMAs.
//
// Few tiles and a long K (weight grads: [64, 4096] @ [4096, 64]): K is split
// too, each split writing a partial C that's summed at the end.
//
// Work: op(B) is packed up front (on the calling thread when it's small, e.g.
// a weight), then C is cut into tiles of whole panels, run in parallel. A tile
// walks K in KC blocks: it packs its rows of op(A) for the block into a
// thread-local buffer (L2), then sweeps each B panel slice (KC×16, L1) down
// them. Packing A inside the tile reads A once, straight into cache.
// ============================================================================

import "base:intrinsics"
import "base:runtime"
import "core:simd"
import "core:sys/info"

@(private = "file")
MR :: 6
@(private = "file")
NR :: 16
@(private = "file")
KC :: 256

// GEMMs smaller than this (multiply-adds) skip packing: attention heads etc.
@(private = "file")
SMALL_GEMM :: 16 * 16 * 64

gemm_x86_available: bool

@(init, private = "file")
gemm_x86_init :: proc "contextless" () {
	f := info.cpu_features()
	gemm_x86_available = .avx2 in f && .fma in f
}

// Packing buffers, per thread: B for the GEMM this thread runs (serial GEMMs
// run concurrently in batched matmul), A for the tile it's working on.
@(thread_local, private = "file")
pack_a_buf: []f32
@(thread_local, private = "file")
pack_b_buf: []f32

// Pack B on the calling thread below this many floats (a weight matrix).
@(private = "file")
SMALL_PACK :: 64 * 1024

// Partial outputs of split-K (the caller's thread only).
@(private = "file")
partial_buf: []f32

@(private = "file")
pack_space :: proc(buf: ^[]f32, n: int) -> [^]f32 {
	if len(buf^) < n {
		if buf^ != nil do runtime.mem_free(raw_data(buf^), scratch())
		m := n + n / 4
		bytes, err := runtime.mem_alloc_non_zeroed(m * size_of(f32), 64, scratch())
		assert(err == nil, "gemm: out of memory")
		buf^ = ([^]f32)(raw_data(bytes))[:m]
	}
	return raw_data(buf^)
}

@(private = "file")
Gemm_Job :: struct {
	C, A, B:          [^]f32,
	M, K, N:          int,
	trans_a, trans_b: bool,
	bp:               [^]f32, // packed op(B) panels
	mp, np:           int, // panel counts
	tile_m, tile_n:   int, // panels per tile
	tiles_n:          int,
	splits, kspan:    int, // split-K: K ranges of kspan, split s > 0 writes partial[s - 1]
	partial:          [^]f32,
}

// C = op(A) @ op(B), overwriting C. parallel: may use the worker pool (only
// from the thread that owns it — not from inside a parallel_for).
gemm_x86 :: proc(C, A, B: []f32, M, K, N: int, trans_a, trans_b: bool, parallel: bool) {
	if M == 0 || N == 0 do return
	if K == 0 {
		for i in 0 ..< M * N do C[i] = 0
		return
	}
	if M * N * K <= SMALL_GEMM {
		gemm_small_avx2(raw_data(C), raw_data(A), raw_data(B), M, K, N, trans_a, trans_b)
		return
	}
	job := Gemm_Job {
		C = raw_data(C), A = raw_data(A), B = raw_data(B),
		M = M, K = K, N = N, trans_a = trans_a, trans_b = trans_b,
		mp = (M + MR - 1) / MR, np = (N + NR - 1) / NR,
	}
	job.bp = pack_space(&pack_b_buf, job.np * K * NR)

	// Tiles: start big (16 × 6 rows, all of N up to 256 columns), halve until
	// there are enough to keep every thread busy. With a long K, columns stay
	// whole (a tile packs its A rows once) and split-K adds the parallelism.
	// Mid-size GEMMs want fewer, fatter units: at least ~512K multiply-adds each.
	threads := parallel ? thread_count() : 1
	want_units := max(1, min(3 * threads, M * N * K / (1 << 19)))
	long_k := K >= 4 * KC
	job.tile_m, job.tile_n = min(job.mp, 16), min(job.np, 16)
	for {
		units := ((job.mp + job.tile_m - 1) / job.tile_m) * ((job.np + job.tile_n - 1) / job.tile_n)
		if units >= want_units || (job.tile_m == 1 && (job.tile_n == 1 || long_k)) do break
		if (job.tile_m >= job.tile_n || long_k) && job.tile_m > 1 do job.tile_m = (job.tile_m + 1) / 2
		else do job.tile_n = (job.tile_n + 1) / 2
	}
	job.tiles_n = (job.np + job.tile_n - 1) / job.tile_n
	tiles := ((job.mp + job.tile_m - 1) / job.tile_m) * job.tiles_n
	job.splits, job.kspan = 1, K
	if parallel && tiles < want_units && long_k {
		want := min((want_units + tiles - 1) / tiles, K / (2 * KC))
		job.kspan = (K + want - 1) / want
		job.kspan = (job.kspan + KC - 1) / KC * KC // whole KC blocks
		job.splits = (K + job.kspan - 1) / job.kspan
		if job.splits > 1 do job.partial = pack_space(&partial_buf, (job.splits - 1) * M * N)
	}

	kblocks := (K + KC - 1) / KC
	if parallel && K * N > SMALL_PACK {
		parallel_for(job.np * kblocks, 1, gemm_pack_b_range, &job)
	} else {
		gemm_pack_b_range(&job, 0, job.np * kblocks)
	}
	if parallel {
		parallel_for(tiles * job.splits, 1, gemm_tile_range, &job)
	} else {
		gemm_tile_range(&job, 0, tiles)
	}
	if job.splits > 1 {
		parallel_for(M * N, 16 * 1024, proc(data: rawptr, lo, hi: int) {
			using j := (^Gemm_Job)(data)
			for s in 1 ..< splits {
				p := partial[(s - 1) * M * N:]
				for i in lo ..< hi do C[i] += p[i]
			}
		}, &job)
	}
}

// Row panels [p0, p1) of op(A), k in [k0, k0 + kc), into dst laid out
// [panel][k][6]: dst[((p - p0)·kc + k)·6 + r] = op(A)[6p + r, k0 + k].
@(private = "file")
gemm_pack_a :: proc(job: ^Gemm_Job, dst: [^]f32, p0, p1, k0, kc: int) {
	using j := job
	for p in p0 ..< p1 {
		d := dst[(p - p0) * kc * MR:]
		rows := min(MR, M - p * MR)
		if trans_a { // A stored [K, M]: the 6 values for one k are adjacent
			for k in 0 ..< kc {
				src := A[(k0 + k) * M + p * MR:]
				for r in 0 ..< rows do d[k * MR + r] = src[r]
				for r in rows ..< MR do d[k * MR + r] = 0
			}
		} else {
			for r in 0 ..< rows {
				src := A[(p * MR + r) * K + k0:]
				for k in 0 ..< kc do d[k * MR + r] = src[k]
			}
			for r in rows ..< MR do for k in 0 ..< kc do d[k * MR + r] = 0
		}
	}
}

// Units [lo, hi) of (panel q, K block): bp[(q·K + k)·16 + c] = op(B)[k, 16q + c].
@(private = "file")
gemm_pack_b_range :: proc(data: rawptr, lo, hi: int) {
	using j := (^Gemm_Job)(data)
	kblocks := (K + KC - 1) / KC
	for u in lo ..< hi {
		q, k0 := u / kblocks, (u % kblocks) * KC
		k1 := min(K, k0 + KC)
		dst := bp[q * K * NR:]
		cols := min(NR, N - q * NR)
		if trans_b { // B stored [N, K]
			for c in 0 ..< cols {
				src := B[(q * NR + c) * K:]
				for k in k0 ..< k1 do dst[k * NR + c] = src[k]
			}
			for c in cols ..< NR do for k in k0 ..< k1 do dst[k * NR + c] = 0
		} else {
			for k in k0 ..< k1 {
				src := B[k * N + q * NR:]
				for c in 0 ..< cols do dst[k * NR + c] = src[c]
				for c in cols ..< NR do dst[k * NR + c] = 0
			}
		}
	}
}

@(private = "file")
gemm_tile_range :: proc(data: rawptr, lo, hi: int) {
	using j := (^Gemm_Job)(data)
	edge: [MR * NR]f32
	ap := pack_space(&pack_a_buf, tile_m * KC * MR)
	for u in lo ..< hi {
		t, s := u / splits, u % splits
		k_lo, k_hi := s * kspan, min(K, (s + 1) * kspan)
		out := s == 0 ? C : partial[(s - 1) * M * N:]
		p0 := (t / tiles_n) * tile_m
		q0 := (t % tiles_n) * tile_n
		p1 := min(p0 + tile_m, mp)
		q1 := min(q0 + tile_n, np)
		for k0 := k_lo; k0 < k_hi; k0 += KC {
			kc := min(KC, k_hi - k0)
			acc := k0 > k_lo
			gemm_pack_a(j, ap, p0, p1, k0, kc)
			for q in q0 ..< q1 {
				b := bp[(q * K + k0) * NR:]
				cols := min(NR, N - q * NR)
				for p in p0 ..< p1 {
					a := ap[(p - p0) * kc * MR:]
					rows := min(MR, M - p * MR)
					c := out[(p * MR) * N + q * NR:]
					if rows == MR && cols == NR {
						gemm_micro_6x16(kc, a, b, c, N, acc)
						continue
					}
					// edge: run on a full 6×16 scratch block, copy the valid part
					if acc do for r in 0 ..< rows do for x in 0 ..< cols do edge[r * NR + x] = c[r * N + x]
					gemm_micro_6x16(kc, a, b, &edge[0], NR, acc)
					for r in 0 ..< rows do for x in 0 ..< cols do c[r * N + x] = edge[r * NR + x]
				}
			}
		}
	}
}

// C[6×16] (+)= A panel [kc][6] · B panel [kc][16].
@(private = "file", enable_target_feature = "avx2,fma")
gemm_micro_6x16 :: proc "contextless" (kc: int, a, b, c: [^]f32, ldc: int, accumulate: bool) {
	v8 :: simd.f32x8
	c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51: v8
	if accumulate {
		c00 = intrinsics.unaligned_load((^v8)(&c[0 * ldc]))
		c01 = intrinsics.unaligned_load((^v8)(&c[0 * ldc + 8]))
		c10 = intrinsics.unaligned_load((^v8)(&c[1 * ldc]))
		c11 = intrinsics.unaligned_load((^v8)(&c[1 * ldc + 8]))
		c20 = intrinsics.unaligned_load((^v8)(&c[2 * ldc]))
		c21 = intrinsics.unaligned_load((^v8)(&c[2 * ldc + 8]))
		c30 = intrinsics.unaligned_load((^v8)(&c[3 * ldc]))
		c31 = intrinsics.unaligned_load((^v8)(&c[3 * ldc + 8]))
		c40 = intrinsics.unaligned_load((^v8)(&c[4 * ldc]))
		c41 = intrinsics.unaligned_load((^v8)(&c[4 * ldc + 8]))
		c50 = intrinsics.unaligned_load((^v8)(&c[5 * ldc]))
		c51 = intrinsics.unaligned_load((^v8)(&c[5 * ldc + 8]))
	}
	pa, pb := a, b
	for _ in 0 ..< kc {
		b0 := intrinsics.unaligned_load((^v8)(&pb[0]))
		b1 := intrinsics.unaligned_load((^v8)(&pb[8]))
		a0 := v8(pa[0])
		c00 = intrinsics.fused_mul_add(a0, b0, c00)
		c01 = intrinsics.fused_mul_add(a0, b1, c01)
		a1 := v8(pa[1])
		c10 = intrinsics.fused_mul_add(a1, b0, c10)
		c11 = intrinsics.fused_mul_add(a1, b1, c11)
		a2 := v8(pa[2])
		c20 = intrinsics.fused_mul_add(a2, b0, c20)
		c21 = intrinsics.fused_mul_add(a2, b1, c21)
		a3 := v8(pa[3])
		c30 = intrinsics.fused_mul_add(a3, b0, c30)
		c31 = intrinsics.fused_mul_add(a3, b1, c31)
		a4 := v8(pa[4])
		c40 = intrinsics.fused_mul_add(a4, b0, c40)
		c41 = intrinsics.fused_mul_add(a4, b1, c41)
		a5 := v8(pa[5])
		c50 = intrinsics.fused_mul_add(a5, b0, c50)
		c51 = intrinsics.fused_mul_add(a5, b1, c51)
		pa = pa[MR:]
		pb = pb[NR:]
	}
	intrinsics.unaligned_store((^v8)(&c[0 * ldc]), c00)
	intrinsics.unaligned_store((^v8)(&c[0 * ldc + 8]), c01)
	intrinsics.unaligned_store((^v8)(&c[1 * ldc]), c10)
	intrinsics.unaligned_store((^v8)(&c[1 * ldc + 8]), c11)
	intrinsics.unaligned_store((^v8)(&c[2 * ldc]), c20)
	intrinsics.unaligned_store((^v8)(&c[2 * ldc + 8]), c21)
	intrinsics.unaligned_store((^v8)(&c[3 * ldc]), c30)
	intrinsics.unaligned_store((^v8)(&c[3 * ldc + 8]), c31)
	intrinsics.unaligned_store((^v8)(&c[4 * ldc]), c40)
	intrinsics.unaligned_store((^v8)(&c[4 * ldc + 8]), c41)
	intrinsics.unaligned_store((^v8)(&c[5 * ldc]), c50)
	intrinsics.unaligned_store((^v8)(&c[5 * ldc + 8]), c51)
}

// Tiny GEMMs, no packing: rows of C accumulate rows of op(B) (contiguous when
// B isn't transposed), else dot products along K.
@(private = "file", enable_target_feature = "avx2,fma")
gemm_small_avx2 :: proc "contextless" (C, A, B: [^]f32, M, K, N: int, trans_a, trans_b: bool) {
	for i in 0 ..< M {
		c := C[i * N:]
		if trans_b {
			for jj in 0 ..< N {
				s: f32 = 0
				for k in 0 ..< K do s += (trans_a ? A[k * M + i] : A[i * K + k]) * B[jj * K + k]
				c[jj] = s
			}
			continue
		}
		for jj in 0 ..< N do c[jj] = 0
		for k in 0 ..< K {
			av := trans_a ? A[k * M + i] : A[i * K + k]
			brow := B[k * N:]
			for jj in 0 ..< N do c[jj] += av * brow[jj]
		}
	}
}

// Many tiny GEMMs (attention heads, e.g. 8192 × [5×8]·[8×5]): batch items
// [lo, hi), C[b] = op(A[b]) @ op(B[b]), in one loop per transpose combination
// instead of one dispatch per item.
gemm_x86_batched_small :: proc(C, A, B: []f32, lo, hi, M, K, N: int, trans_a, trans_b: bool) {
	c, a, b := raw_data(C), raw_data(A), raw_data(B)
	switch {
	case !trans_a && !trans_b: bmm_small(c, a, b, lo, hi, M, K, N, false, false)
	case trans_a && !trans_b: bmm_small(c, a, b, lo, hi, M, K, N, true, false)
	case !trans_a && trans_b: bmm_small(c, a, b, lo, hi, M, K, N, false, true)
	case: bmm_small(c, a, b, lo, hi, M, K, N, true, true)
	}
}

@(private = "file", enable_target_feature = "avx2,fma")
bmm_small :: proc "contextless" (C, A, B: [^]f32, lo, hi, M, K, N: int, $TA, $TB: bool) {
	for z in lo ..< hi {
		c, a, b := C[z * M * N:], A[z * M * K:], B[z * K * N:]
		for i in 0 ..< M {
			row := c[i * N:]
			when TB { // B stored [N, K]: dot products along K
				for j in 0 ..< N {
					s: f32 = 0
					for k in 0 ..< K do s += (TA ? a[k * M + i] : a[i * K + k]) * b[j * K + k]
					row[j] = s
				}
			} else { // rows of B, contiguous: c[i, :] += a[i, k] · b[k, :]
				for j in 0 ..< N do row[j] = 0
				for k in 0 ..< K {
					av := TA ? a[k * M + i] : a[i * K + k]
					br := b[k * N:]
					for j in 0 ..< N do row[j] += av * br[j]
				}
			}
		}
	}
}
