# Optimized builds by default: -o:none is ~10x slower for the training examples.
ODIN  ?= odin
FAST  := -o:speed -no-bounds-check
BUILD := build

PHONY_PERF := perf perf-baseline perf-tiny
.PHONY: test test-gpu test-metal test-cuda oracle oracle-tiny mnist cnn regression tour bench data clean dt-conjugate dt-table1 dt-fusion models

test:        ; $(ODIN) run tests/tensor_ops -out:$(BUILD)/tensor_ops
test-gpu:    ; $(ODIN) run tests/parity -o:speed -out:$(BUILD)/parity   # GPU (Metal or CUDA) vs CPU, every kernel path
test-metal:  ; ML_DEVICE=metal $(MAKE) test-gpu
test-cuda:   ; ML_DEVICE=cuda $(MAKE) test-gpu
# CUDA codegen check without a GPU: render every kernel shape, parse it with clang
CLANG ?= $(shell test -x /opt/homebrew/opt/llvm/bin/clang && echo /opt/homebrew/opt/llvm/bin/clang || echo clang)
check-cuda-render:
	@mkdir -p $(BUILD)/cuda_render && rm -f $(BUILD)/cuda_render/*.cu
	@$(ODIN) run tests/cuda_render -out:$(BUILD)/cuda_render_gen -- $(BUILD)/cuda_render
	@for f in $(BUILD)/cuda_render/*.cu; do \
		$(CLANG) -x cuda --cuda-gpu-arch=sm_89 -nocudainc -nocudalib --cuda-device-only -fsyntax-only \
			-include tests/cuda_render/nvrtc_stub.h $$f || exit 1; echo "  ok $$(basename $$f)"; done
# tinygrad + MLX oracles; ML_DEVICE=metal|cuda make oracle checks the GPU path
oracle:
	$(ODIN) build bench/odin_vs_tiny -o:speed -out:$(BUILD)/odin_vs_tiny
	$(BUILD)/odin_vs_tiny > bench/odin_ref.txt
	cd bench && uv run tinygrad_compare.py odin_ref.txt && uv run mlx_compare.py odin_ref.txt
# tinygrad only, from a local checkout (Linux: no MLX); DEV=CUDA runs tinygrad on the GPU
TINYGRAD_PATH ?= $(HOME)/code/repos/tinygrad
oracle-tiny:
	$(ODIN) build bench/odin_vs_tiny -o:speed -out:$(BUILD)/odin_vs_tiny
	$(BUILD)/odin_vs_tiny > bench/odin_ref.txt
	cd bench && TINYGRAD_PATH=$(TINYGRAD_PATH) uv run --no-project --with numpy python tinygrad_compare.py odin_ref.txt
mnist:       ; $(ODIN) run examples/mnist $(FAST) -out:$(BUILD)/mnist
cnn:         ; $(ODIN) run examples/mnist_cnn $(FAST) -out:$(BUILD)/mnist_cnn
regression:  ; $(ODIN) run examples/regression $(FAST) -out:$(BUILD)/regression
bench:       ; $(ODIN) run bench/loop $(FAST) -disable-assert -out:$(BUILD)/loop
# performance plan yardstick (STATUS.md): ML_DEVICE=metal|cuda|cpu make perf; ARGS=<name> runs matching workloads
GIT_REV := $(shell git rev-parse --short HEAD 2>/dev/null)
perf:          ; $(ODIN) run bench/perf $(FAST) -define:GIT_REV='"$(GIT_REV)"' -out:$(BUILD)/perf -- $(ARGS)
perf-baseline: ; $(ODIN) run bench/perf $(FAST) -define:GIT_REV='"$(GIT_REV)"' -out:$(BUILD)/perf -- baseline
# the bar: the same M4 step in tinygrad (local checkout), default and BEAM=2 (first BEAM run searches ~6 min)
perf-tiny:
	cd dt/tinygrad && TINYGRAD_PATH=$(TINYGRAD_PATH) uv run --no-project --with numpy python fusion.py bench 30
	cd dt/tinygrad && BEAM=2 TINYGRAD_PATH=$(TINYGRAD_PATH) uv run --no-project --with numpy python fusion.py bench 30
tour:
	@for f in examples/tour/0*.odin; do $(ODIN) run $$f -file -out:$(BUILD)/tour || exit 1; done
data:        ; sh data/download-mnist.sh

# Distribution Transformers (dt/)
dt-conjugate: ; $(ODIN) run dt/conjugate $(FAST) -out:$(BUILD)/dt_conjugate
dt-table1:    ; $(ODIN) run dt/table1 $(FAST) -out:$(BUILD)/dt_table1   # DT-5; DT-2: add -- 2
dt-fusion:    ; $(ODIN) run dt/fusion $(FAST) -out:$(BUILD)/dt_fusion   # M4; retrain: add -- train
# trained weights live in models/ (safetensors); experiments load them instead of retraining
models:       ; @$(ODIN) run dt/models -out:$(BUILD)/models
clean:       ; rm -rf $(BUILD)

$(shell mkdir -p $(BUILD))
