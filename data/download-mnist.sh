#!/bin/sh
# Fetch MNIST into data/mnist/ (IDX files, as ml.load_idx_* expects).
# Mirror: the TensorFlow datasets bucket. Run from anywhere.
set -eu
cd "$(dirname "$0")"
mkdir -p mnist
base=https://storage.googleapis.com/cvdf-datasets/mnist
fetch() { # $1 = name, $2 = idx kind
	out="mnist/$1.$2-ubyte"
	[ -f "$out" ] && { echo "have $out"; return; }
	echo "get  $out"
	curl -fsSL "$base/$1-$2-ubyte.gz" | gunzip > "$out"
}
fetch train-images idx3
fetch train-labels idx1
fetch t10k-images  idx3
fetch t10k-labels  idx1
