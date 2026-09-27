package dt

// ============================================================================
// Model library: every experiment saves its trained weights to models/ as
// safetensors (with its config and results as metadata) and loads them on the
// next run instead of retraining. Pass `train` on the command line to retrain.
//
//   make models      lists what's there (dt/models)
// ============================================================================

import "core:fmt"
import "core:os"
import ml "../ml"

MODELS_DIR :: "models"

// Command line without the `train` flag, and whether it was given.
args :: proc() -> (positional: []string, retrain: bool) {
	out := make([dynamic]string, 0, len(os.args))
	for a in os.args[1:] {
		if a == "train" do retrain = true
		else do append(&out, a)
	}
	return out[:], retrain
}

// Load weights from path unless retraining; prints the stored metadata.
try_load :: proc(path: string, params: []^ml.Tensor, retrain: bool) -> bool {
	if retrain || !os.exists(path) do return false
	meta, ok := ml.load(path, params)
	if !ok do return false
	fmt.printfln("loaded %s (retrain with `train`)", path)
	for kv in meta do fmt.printfln("  %-20s %s", kv.key, kv.value)
	return true
}

// Where a training run saves: the experiment's checkpoint for a full-length run,
// a step-suffixed sibling for shorter/longer ones (a quick test never clobbers it).
run_path :: proc(path: string, steps, default_steps: int) -> string {
	if steps == default_steps do return path
	return fmt.tprintf("%s_%dsteps.safetensors", path[:len(path) - len(".safetensors")], steps)
}

save_model :: proc(path: string, params: []^ml.Tensor, meta: []ml.Meta) {
	os.make_directory_all(MODELS_DIR)
	if ml.save(path, params, meta = meta) do fmt.printfln("saved %s", path)
}
