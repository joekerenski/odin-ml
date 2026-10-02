package ml

// x86 handles subnormal floats in microcode, ~100× slower than normal ones,
// and smooth activations make plenty of them in backward passes (σ(1−σ) far
// from 0, exp of very negative values). Flush them to zero (FTZ) and read
// them as zero (DAZ), per thread — the same as GPUs do. ARM cores (the M5)
// handle subnormals at full speed, so this is x86 only.

import "core:simd/x86"

@(enable_target_feature = "sse")
flush_denormals :: proc "contextless" () {
	x86._mm_setcsr(x86._mm_getcsr() | x86._MM_FLUSH_ZERO_ON | 0x0040) // FTZ | DAZ
}
