# Optimized builds by default: -o:none is ~10x slower for the training examples.
ODIN  ?= odin
FAST  := -o:speed -no-bounds-check
BUILD := build

.PHONY: test mnist cnn regression tour bench data clean dt-conjugate

test:        ; $(ODIN) run tests/tensor_ops -out:$(BUILD)/tensor_ops
mnist:       ; $(ODIN) run examples/mnist $(FAST) -out:$(BUILD)/mnist
cnn:         ; $(ODIN) run examples/mnist_cnn $(FAST) -out:$(BUILD)/mnist_cnn
regression:  ; $(ODIN) run examples/regression $(FAST) -out:$(BUILD)/regression
bench:       ; $(ODIN) run bench/loop $(FAST) -disable-assert -out:$(BUILD)/loop
tour:
	@for f in examples/tour/0*.odin; do $(ODIN) run $$f -file -out:$(BUILD)/tour || exit 1; done
data:        ; sh data/download-mnist.sh

# Distribution Transformers (dt/)
dt-conjugate: ; $(ODIN) run dt/conjugate $(FAST) -out:$(BUILD)/dt_conjugate
clean:       ; rm -rf $(BUILD)

$(shell mkdir -p $(BUILD))
