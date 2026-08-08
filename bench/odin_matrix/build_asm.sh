#!/bin/sh
# Build this bench to assembly / LLVM IR and show what the study procs compile to.
# Usage (from bench/odin_matrix/): sh build_asm.sh

set -e
cd "$(dirname "$0")"

FLAGS="-o:speed -no-bounds-check -disable-assert"

echo "== assembly (study_* procs) =="
odin build . -build-mode:asm $FLAGS -out:study.s
for p in study_mat_add study_mat_mul study_plain_add study_plain_mul study_slice_add study_slice_mul; do
	echo
	echo "--- $p ---"
	awk "/^_$p:/,/\tret/" study.s | rg -v "^\s*\.cfi|^\s*\.p2align|^\s*\.data|^\s*\.text|^\s*\.globl|L__unnamed|^\s*\.section|^\s*\.subsections|^\s*\.byte|^\s*\.asciz|^\s*\.long|^\s*\.quad" | head -60
done

echo
echo "== LLVM IR (look for <4 x float>, fmul/fadd on vectors) =="
rm -rf study_ir
mkdir -p study_ir
odin build . -build-mode:llvm-ir $FLAGS -out:study_ir/
IR=$(find study_ir -name "*.ll" | head -1)
echo "IR file: $IR"
for p in study_mat_add study_mat_mul study_plain_add study_plain_mul study_slice_add study_slice_mul; do
	echo
	echo "--- $p ---"
	rg -n -A 40 "define.*$p\(" "$IR" | rg "define|fadd|fmul|fneg|<4 x float>|<2 x float>|load|store|ptrtoint|bitcast" | head -40
done
