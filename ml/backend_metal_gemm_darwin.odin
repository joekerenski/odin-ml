package ml

// ============================================================================
// Metal GEMM: generated kernels on the hardware 8×8 matrix units
// (simdgroup_matrix, what tinygrad's Metal tensor cores use), one per tile
// shape, plus a one-thread-per-output kernel for tiny batched matrices
// (attention heads). Which one runs, and whether K is split, is picked per
// problem by timing the candidates on the device (search.odin caches the
// choice on disk); ML_SEARCH=0 uses a fixed heuristic.
//
// Operands are strided (views read in place): element (z0, z1, i, j) of X at
// X[z0·b0 + z1·b1 + i·rs + j·cs]. Grid z = z0·Z1 + z1. Split-K: all z share A
// and B, z covers K range [z·kc, (z+1)·kc) and writes a dense partial C[z],
// summed by a second kernel.
// ============================================================================

import "core:fmt"
import "core:strings"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

@(private)
GEMM_P :: `// p = [M, K, N, kc, Z1, then rs, cs, b0, b1 of A, B, C]
struct Gemm_P {
    uint M, K, N, kc, Z1;
    uint a_rs, a_cs, a_b0, a_b1, b_rs, b_cs, b_b0, b_b1, c_rs, c_cs, c_b0, c_b1;
};`

// Tile shapes (BM × BN of C per 128-thread group), each as a staged kernel
// (threadgroup memory) and a direct one (fragments straight from device
// memory), then the tiny kernel.
@(private = "file")
GEMM_TILES := [?][2]int{{32, 32}, {64, 32}, {32, 64}, {64, 64}}
@(private = "file")
GEMM_DIRECT :: len(GEMM_TILES) // tile index t + GEMM_DIRECT: the direct kernel
@(private = "file")
GEMM_SMALL :: 2 * len(GEMM_TILES)

// A choice: tile index (or GEMM_SMALL) and K splits, as one int for the cache.
@(private = "file")
Gemm_Choice :: struct {
	tile, splits: int,
}

@(private = "file")
encode_choice :: proc(c: Gemm_Choice) -> int { return c.tile * 1000 + c.splits }
@(private = "file")
decode_choice :: proc(v: int) -> Gemm_Choice { return {v / 1000, v % 1000} }

// A 128-thread group (4 SIMD groups, 2 × 2) computes a BM × BN tile of C. Per
// K step of 32 all threads load A[BM × 32] and B[32 × BN] into threadgroup
// memory with coalesced reads (consecutive threads walk the operand's
// unit-stride axis), zero-padding the edges; each SIMD group then multiplies
// its (BM/2) × (BN/2) corner as 8×8 tiles. C goes out through threadgroup
// memory, so edges and any C layout are written with plain bounds checks.
// epi: an epilogue kernel (gpu_epilogue_metal) applied to each element of C
// instead of storing it; nil: plain.
@(private = "file")
gemm_source :: proc(bm, bn: int, direct: bool, epi: ^Kernel = nil) -> string {
	fn, params, args: string
	if epi != nil do fn, params, args = gpu_epilogue_metal(epi, GEMM_EPI_SLOT, 29)
	src := strings.concatenate({"#include <metal_stdlib>\nusing namespace metal;\n", GEMM_P, "\n", fn, direct ? GEMM_DIRECT_TEMPLATE : GEMM_TEMPLATE}, context.temp_allocator)
	store := "c[(i0 + r) * p.c_rs + (j0 + q) * p.c_cs] = sh[r * BN + q];"
	direct_out := `for (uint r = 0; r < TM; r++) for (uint q = 0; q < TN; q++) {
        uint i = i0 + 8 * r, j = j0 + 8 * q;
        if (i >= M || j >= N) continue;
        if (tc) simdgroup_store(acc[r][q], c, ldc, ulong2(i, j), true);
        else simdgroup_store(acc[r][q], c, ldc, ulong2(j, i), false);
    }`
	if epi != nil {
		store = fmt.tprintf("epi((tg.z * M + i0 + r) * N + j0 + q, sh[r * BN + q], %s);", args)
		// through threadgroup memory: each element's index for the epilogue
		direct_out = fmt.tprintf(`threadgroup float sh[BM * BN];
    uint ti = tg.y * BM, tj = tg.x * BN;
    for (uint r = 0; r < TM; r++) for (uint q = 0; q < TN; q++)
        simdgroup_store(acc[r][q], &sh[(i0 - ti + 8 * r) * BN + j0 - tj + 8 * q], BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {{
        uint r = e / BN, q = e %% BN;
        if (ti + r < M && tj + q < N) epi((tg.z * M + ti + r) * N + tj + q, sh[e], %s);
    }}`, args)
	}
	src, _ = strings.replace_all(src, "@EPI_PARAMS@", params, context.temp_allocator)
	src, _ = strings.replace_all(src, "@STORE@", store, context.temp_allocator)
	src, _ = strings.replace_all(src, "@DIRECT_OUT@", direct_out, context.temp_allocator)
	src, _ = strings.replace_all(src, "@BM@", fmt.tprint(bm), context.temp_allocator)
	src, _ = strings.replace_all(src, "@BN@", fmt.tprint(bn), context.temp_allocator)
	return src
}

// Epilogue buffers bind from this slot (0-2: A, B, C; 29: the epilogue's
// parameters, 30: the GEMM's).
@(private = "file")
GEMM_EPI_SLOT :: 3

@(private = "file")
GEMM_TEMPLATE :: `
kernel void gemm_@BM@x@BN@(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]], constant Gemm_P& p [[buffer(30)]], @EPI_PARAMS@
                      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = @BM@, BN = @BN@, BK = 32, TM = BM / 16, TN = BN / 16;
    constexpr uint SH = BM * BK + BK * BN > BM * BN ? BM * BK + BK * BN : BM * BN;
    uint M = p.M, K = p.K, N = p.N, kc = p.kc;
    bool ta = p.a_cs != 1, tb = p.b_cs != 1;
    uint z0 = tg.z / p.Z1, z1 = tg.z % p.Z1;
    device const float* a = kc ? A : A + z0 * p.a_b0 + z1 * p.a_b1;
    device const float* b = kc ? B : B + z0 * p.b_b0 + z1 * p.b_b1;
    device float* c = C + z0 * p.c_b0 + z1 * p.c_b1;
    uint k_lo = kc ? tg.z * kc : 0, k_hi = kc ? min(K, k_lo + kc) : K;
    uint i0 = tg.y * BM, j0 = tg.x * BN;
    threadgroup float sh[SH];
    threadgroup float* As = sh;          // [BM][BK]
    threadgroup float* Bs = sh + BM * BK; // [BK][BN]
    uint si = (sg / 2) * (BM / 2), sj = (sg % 2) * (BN / 2);
    simdgroup_float8x8 acc[TM][TN];
    for (uint r = 0; r < TM; r++) for (uint q = 0; q < TN; q++) acc[r][q] = simdgroup_float8x8(0.0f);
    for (uint k0 = k_lo; k0 < k_hi; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += 128) {
            uint r = ta ? e % BM : e / BK, kk = ta ? e / BM : e % BK;
            uint gi = i0 + r, gk = k0 + kk;
            As[r * BK + kk] = (gi < M && gk < k_hi) ? a[gi * p.a_rs + gk * p.a_cs] : 0.0f;
        }
        for (uint e = tid; e < BK * BN; e += 128) {
            uint q = tb ? e / BK : e % BN, kb = tb ? e % BK : e / BN;
            uint gj = j0 + q, gk = k0 + kb;
            Bs[kb * BN + q] = (gj < N && gk < k_hi) ? b[gk * p.b_rs + gj * p.b_cs] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 am[TM], bm[TN];
            for (uint r = 0; r < TM; r++) simdgroup_load(am[r], &As[(si + 8 * r) * BK + kk], BK);
            for (uint q = 0; q < TN; q++) simdgroup_load(bm[q], &Bs[kk * BN + sj + 8 * q], BN);
            for (uint r = 0; r < TM; r++) for (uint q = 0; q < TN; q++) simdgroup_multiply_accumulate(acc[r][q], am[r], bm[q], acc[r][q]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < TM; r++) for (uint q = 0; q < TN; q++) simdgroup_store(acc[r][q], &sh[(si + 8 * r) * BN + sj + 8 * q], BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool tc = p.c_cs != 1;
    for (uint e = tid; e < BM * BN; e += 128) {
        uint r = tc ? e % BM : e / BN, q = tc ? e / BM : e % BN;
        if (i0 + r < M && j0 + q < N) @STORE@
    }
}
`

// Direct: each SIMD group loads its 8×8 fragments straight from device memory
// (simdgroup_load with the operand's leading dim; a column-major operand loads
// transposed) and stores C the same way: no threadgroup memory, no barriers.
// Needs a unit stride per operand and M, N, K multiples of 8 (whole fragments).
@(private = "file")
GEMM_DIRECT_TEMPLATE :: `
kernel void gemmd_@BM@x@BN@(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]], constant Gemm_P& p [[buffer(30)]], @EPI_PARAMS@
                      uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint BM = @BM@, BN = @BN@, TM = BM / 16, TN = BN / 16;
    uint M = p.M, K = p.K, N = p.N, kc = p.kc;
    bool ta = p.a_cs != 1, tb = p.b_cs != 1, tc = p.c_cs != 1;
    ulong lda = ta ? p.a_cs : p.a_rs, ldb = tb ? p.b_cs : p.b_rs, ldc = tc ? p.c_cs : p.c_rs;
    uint z0 = tg.z / p.Z1, z1 = tg.z % p.Z1;
    device const float* a = kc ? A : A + z0 * p.a_b0 + z1 * p.a_b1;
    device const float* b = kc ? B : B + z0 * p.b_b0 + z1 * p.b_b1;
    device float* c = C + z0 * p.c_b0 + z1 * p.c_b1;
    uint k_lo = kc ? tg.z * kc : 0, k_hi = kc ? min(K, k_lo + kc) : K;
    uint i0 = tg.y * BM + (sg / 2) * (BM / 2), j0 = tg.x * BN + (sg % 2) * (BN / 2);
    simdgroup_float8x8 acc[TM][TN];
    for (uint r = 0; r < TM; r++) for (uint q = 0; q < TN; q++) acc[r][q] = simdgroup_float8x8(0.0f);
    for (uint k = k_lo; k < k_hi; k += 8) {
        simdgroup_float8x8 am[TM], bm[TN];
        for (uint r = 0; r < TM; r++) {
            uint i = min(i0 + 8 * r, M - 8); // past the edge: a valid fragment, result discarded
            if (ta) simdgroup_load(am[r], a, lda, ulong2(i, k), true);
            else simdgroup_load(am[r], a, lda, ulong2(k, i), false);
        }
        for (uint q = 0; q < TN; q++) {
            uint j = min(j0 + 8 * q, N - 8);
            if (tb) simdgroup_load(bm[q], b, ldb, ulong2(k, j), true);
            else simdgroup_load(bm[q], b, ldb, ulong2(j, k), false);
        }
        for (uint r = 0; r < TM; r++) for (uint q = 0; q < TN; q++) simdgroup_multiply_accumulate(acc[r][q], am[r], bm[q], acc[r][q]);
    }
    @DIRECT_OUT@
}
`

@(private = "file")
gemm_pso :: proc(tile: int) -> ^MTL.ComputePipelineState {
	@(static) psos: [GEMM_SMALL + 1]^MTL.ComputePipelineState
	if psos[tile] != nil do return psos[tile]
	if tile == GEMM_SMALL {
		psos[tile] = metal_get_kernel(KERNELS, "matmul_small")
	} else {
		t := GEMM_TILES[tile % GEMM_DIRECT]
		psos[tile] = metal_get_kernel(gemm_source(t[0], t[1], tile >= GEMM_DIRECT), gemm_name(tile))
	}
	return psos[tile]
}

@(private = "file")
gemm_name :: proc(tile: int) -> string {
	if tile == GEMM_SMALL do return "matmul_small"
	t := GEMM_TILES[tile % GEMM_DIRECT]
	return fmt.tprintf("%s_%dx%d", tile >= GEMM_DIRECT ? "gemmd" : "gemm", t[0], t[1])
}

// One choice as dispatches: the GEMM (into C, or a dense partial per split)
// and, when K is split, the combine.
@(private = "file")
Gemm_Launch :: struct {
	params:      [17]u32,
	grid, group: [3]int,
	splits:      int,
}

@(private = "file")
gemm_launch :: proc(g: ^Gemm, ch: Gemm_Choice) -> (l: Gemm_Launch) {
	l.params = gemm_params(g)
	Z := g.Z0 * g.Z1
	if ch.tile == GEMM_SMALL {
		l.grid, l.group = {g.N, g.M, Z}, {g.N, g.M, 1}
		return
	}
	t := GEMM_TILES[ch.tile % GEMM_DIRECT]
	l.group = {128, 1, 1}
	l.grid = {(g.N + t[1] - 1) / t[1] * 128, (g.M + t[0] - 1) / t[0], Z}
	if ch.splits > 1 {
		kc := round_up((g.K + ch.splits - 1) / ch.splits, 32) // whole K tiles per split
		l.splits = (g.K + kc - 1) / kc
		l.params[3], l.params[4] = u32(kc), 1
		l.params[13], l.params[14], l.params[15] = u32(g.N), 1, u32(g.M * g.N) // dense partial C[z] at z·m·n
		l.grid[2] = l.splits
	}
	return
}

// The choices worth trying for a problem.
@(private = "file")
gemm_candidates :: proc(g: ^Gemm, out: ^[dynamic]Gemm_Choice) {
	Z := g.Z0 * g.Z1
	if g.M <= 32 && g.N <= 32 && g.K <= 256 do append(out, Gemm_Choice{GEMM_SMALL, 1})
	c_dense := g.c.cs == 1 && (g.M == 1 || g.c.rs == g.N)
	direct := gemm_direct_ok(g)
	for t in 0 ..< 2 * len(GEMM_TILES) {
		if t >= GEMM_DIRECT && !direct do break
		append(out, Gemm_Choice{t, 1})
		if Z == 1 && c_dense && g.K >= 512 && GEMM_TILES[t % GEMM_DIRECT][0] == 32 { // split-K is for few outputs
			// direct split-K: each split must be whole fragments (kc is a multiple of 32)
			for s := 2; s <= 64 && g.K / s >= 64; s *= 2 do append(out, Gemm_Choice{t, s})
		}
	}
}

// The direct kernels need whole 8×8 fragments and a unit stride per operand.
@(private = "file")
gemm_direct_ok :: proc(g: ^Gemm) -> bool {
	unit :: proc(o: Gemm_Operand) -> bool { return o.rs == 1 || o.cs == 1 }
	return g.M % 8 == 0 && g.N % 8 == 0 && g.K % 8 == 0 && unit(g.a) && unit(g.b) && unit(g.c)
}

// Without a search: what the backend did before searching existed.
@(private = "file")
gemm_default :: proc(g: ^Gemm) -> Gemm_Choice {
	Z := g.Z0 * g.Z1
	c_dense := g.c.cs == 1 && (g.M == 1 || g.c.rs == g.N)
	if g.M <= 16 && g.N <= 16 && g.K <= 64 do return {GEMM_SMALL, 1}
	if Z == 1 && c_dense && g.K >= 1024 && g.M * g.N <= 64 * 1024 do return {0, min(g.K / 256, 64)}
	return {0, 1}
}

@(private = "file")
gemm_key :: proc(g: ^Gemm) -> string {
	return fmt.tprintf("gemm2 m%d k%d n%d z%d a%d b%d c%d", g.M, g.K, g.N, g.Z0 * g.Z1,
		int(g.a.cs == 1), int(g.b.cs == 1), int(g.c.cs == 1 && g.c.rs == g.N))
}

metal_matmul :: proc(g: ^Gemm) {
	key := gemm_key(g)
	ch: Gemm_Choice
	if gemm_force >= 0 {
		cands := make([dynamic]Gemm_Choice, context.temp_allocator)
		gemm_candidates(g, &cands)
		ch = gemm_default(g)
		for c in cands do if c.tile == gemm_force % (GEMM_SMALL + 1) && (c.splits == 1) == (gemm_force <= GEMM_SMALL) do ch = c
	} else if v, ok := choice_get(key); ok {
		ch = decode_choice(v)
	} else if search_enabled {
		ch = gemm_search(g)
		choice_put(key, encode_choice(ch))
	} else {
		ch = gemm_default(g)
	}
	l := gemm_launch(g, ch)
	a, b, c := resolve(g.a.data), resolve(g.b.data), resolve(g.c.data, output = true)
	label := ""
	if kernel_timing() {
		name := gemm_name(ch.tile)
		label = fmt.tprintf("%s%s %dx%dx%d b%d", name, l.splits > 1 ? fmt.tprintf(" splitk s%d", l.splits) : "", g.M, g.K, g.N, g.Z0 * g.Z1)
	}
	if l.splits > 1 {
		partial := scratch_alloc(l.splits * g.M * g.N * size_of(f32))
		dispatch(gemm_pso(ch.tile), {a, b, partial}, l.params[:], l.grid, l.group, 2, label)
		dispatch(metal_get_kernel(KERNELS, "reduce_sum_thread"), {partial, c}, []u32{1, u32(l.splits), u32(g.M * g.N)},
			{g.M * g.N, 1, 1}, {256, 1, 1}, 1, kernel_timing() ? "splitk_sum" : "")
		return
	}
	dispatch(gemm_pso(ch.tile), {a, b, c}, l.params[:], l.grid, l.group, 2, label)
}

// Time every candidate on scratch operands of the same extents (values don't
// matter for speed). R copies of the operands, R runs per command buffer each
// on the next copy: the footprint exceeds the system cache, so runs read
// memory about as cold as the real step's GEMMs do.
@(private = "file")
gemm_search :: proc(g: ^Gemm) -> Gemm_Choice {
	extent :: proc(g: ^Gemm, o: Gemm_Operand, R, C: int) -> int {
		return (g.Z0 - 1) * o.bs[0] + (g.Z1 - 1) * o.bs[1] + (R - 1) * o.rs + (C - 1) * o.cs + 1
	}
	pool := NS.AutoreleasePool_alloc()->init()
	defer NS.AutoreleasePool_drain(pool)
	ext := [3]int{extent(g, g.a, g.M, g.K), extent(g, g.b, g.K, g.N), extent(g, g.c, g.M, g.N)}
	bytes := 4 * (ext[0] + ext[1] + ext[2])
	R := clamp((32 * 1024 * 1024 + bytes - 1) / bytes, 2, 8)
	cands := make([dynamic]Gemm_Choice, context.temp_allocator)
	gemm_candidates(g, &cands)
	max_splits := 1
	for ch in cands do max_splits = max(max_splits, gemm_launch(g, ch).splits)
	// one scratch buffer for every search, grown as needed: R operand copies, then the partials
	@(static) arena: ^MTL.Buffer
	need := R * bytes + 4 * max_splits * g.M * g.N + 1024 * (3 * R + 1)
	if arena == nil || int(MTL.Buffer_length(arena)) < need {
		if arena != nil do arena->release()
		arena = metal_new_buffer_empty(need)
	}
	copies := make([][3]Dev_Ref, R, context.temp_allocator)
	off := 0
	for &c in copies do for j in 0 ..< 3 {
		c[j] = Dev_Ref{arena, off, 4 * ext[j]}
		off = (off + 4 * ext[j] + 255) &~ 255
	}
	partial := Dev_Ref{arena, off, 0}
	best, best_ms := gemm_default(g), max(f64)
	combine := metal_get_kernel(KERNELS, "reduce_sum_thread")
	for ch in cands {
		l := gemm_launch(g, ch)
		pso := gemm_pso(ch.tile)
		ms := max(f64)
		{
			cmd := MTL.CommandQueue_commandBuffer(metal_ctx.queue)
			enc := MTL.CommandBuffer_computeCommandEncoder(cmd)
			for c in copies {
				out := l.splits > 1 ? partial : c[2]
				encode_one(enc, pso, {c[0], c[1], out}, l.params[:], l.grid, l.group)
				if l.splits > 1 {
					p := []u32{1, u32(l.splits), u32(g.M * g.N)}
					encode_one(enc, combine, {partial, c[2]}, p, {g.M * g.N, 1, 1}, {256, 1, 1})
				}
			}
			MTL.CommandEncoder_endEncoding(enc)
			MTL.CommandBuffer_commit(cmd)
			MTL.CommandBuffer_waitUntilCompleted(cmd)
			ms = min(ms, f64(MTL.CommandBuffer_GPUEndTime(cmd) - MTL.CommandBuffer_GPUStartTime(cmd)) * 1000 / f64(R))
		}
		if ms < best_ms do best, best_ms = ch, ms
	}
	if debug_level >= 1 {
		fmt.printfln("  search  %s → %s s%d (%.4f ms, %d candidates)", gemm_key(g), gemm_name(best.tile), best.splits, best_ms, len(cands))
	}
	return best
}

@(private = "file")
encode_one :: proc(enc: ^MTL.ComputeCommandEncoder, pso: ^MTL.ComputePipelineState, bufs: []Dev_Ref, params: []u32, grid, group: [3]int) {
	MTL.ComputeCommandEncoder_setComputePipelineState(enc, pso)
	for b, i in bufs do MTL.ComputeCommandEncoder_setBuffer(enc, b.buf, NS.UInteger(b.off), NS.UInteger(i))
	MTL.ComputeCommandEncoder_setBytes(enc, ([^]byte)(raw_data(params))[:len(params) * 4], 30)
	MTL.ComputeCommandEncoder_dispatchThreads(enc,
		{NS.Integer(grid[0]), NS.Integer(grid[1]), NS.Integer(grid[2])},
		{NS.Integer(group[0]), NS.Integer(group[1]), NS.Integer(group[2])})
}

// GEMM + elementwise epilogue in one kernel: C is never stored, each element
// goes through k. Tiled kernels only (not the tiny one, not split-K).
metal_matmul_epi :: proc(g: ^Gemm, k: ^Kernel) -> bool {
	if GEMM_EPI_SLOT + k.n_bufs > 29 do return false
	key := gemm_key(g)
	ch: Gemm_Choice
	if gemm_force >= 0 {
		ch = {gemm_force % (GEMM_SMALL + 1), 1}
		if ch.tile == GEMM_SMALL || ch.tile >= GEMM_DIRECT && !gemm_direct_ok(g) do return false
	} else if v, ok := choice_get(key); ok {
		ch = decode_choice(v)
	} else if search_enabled {
		ch = gemm_search(g)
		choice_put(key, encode_choice(ch))
	} else {
		ch = gemm_default(g)
	}
	if ch.tile == GEMM_SMALL || ch.splits > 1 do return false
	@(static) psos: map[u64]^MTL.ComputePipelineState
	if psos == nil do psos = make(map[u64]^MTL.ComputePipelineState, scratch())
	h := kernel_hash(k, 1000 + ch.tile)
	pso, ok := psos[h]
	if !ok {
		t := GEMM_TILES[ch.tile % GEMM_DIRECT]
		pso = metal_compile(gemm_source(t[0], t[1], ch.tile >= GEMM_DIRECT, k), gemm_name(ch.tile))
		psos[h] = pso
	}
	l := gemm_launch(g, ch)
	bufs: [GEMM_EPI_SLOT + MAX_KERNEL_BUFS]Dev_Ref
	bufs[0], bufs[1] = resolve(g.a.data), resolve(g.b.data)
	bufs[2] = bufs[0] // C is not used
	for j in 0 ..< k.n_bufs do bufs[GEMM_EPI_SLOT + j] = resolve(k.bufs[j], output = j >= k.n_in)
	ep: [GPU_PARAMS]u32
	n := gpu_params(k, &ep)
	label := ""
	if kernel_timing() do label = fmt.tprintf("%s+epi %dx%dx%d b%d %s", gemm_name(ch.tile), g.M, g.K, g.N, g.Z0 * g.Z1, kernel_describe(k))
	dispatch(pso, bufs[:GEMM_EPI_SLOT + k.n_bufs], l.params[:], l.grid, l.group, GEMM_EPI_SLOT + k.n_in, label, ep[:n])
	return true
}
