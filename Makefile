# Optimized builds by default: -o:none is ~10x slower for the training examples.
ODIN  ?= odin
FAST  := -o:speed -no-bounds-check
BUILD := build

.PHONY: test test-gpu test-metal test-cuda oracle oracle-tiny mnist cnn regression tour bench data clean dt-conjugate dt-table1 dt-fusion models

test:        ; $(ODIN) run tests/tensor_ops -out:$(BUILD)/tensor_ops
test-gpu:    ; $(ODIN) run tests/parity -o:speed -out:$(BUILD)/parity   # GPU (Metal or CUDA) vs CPU, every kernel path
test-metal:  ; ML_DEVICE=metal $(MAKE) test-gpu
test-cuda:   ; ML_DEVICE=cuda $(MAKE) test-gpu
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
