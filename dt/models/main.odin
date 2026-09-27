package main

// Lists the model library: every checkpoint in models/ with its metadata.
//
//   make models

import "core:fmt"
import "core:os"
import "core:strings"
import dt ".."
import ml "../../ml"

main :: proc() {
	entries, err := os.read_all_directory_by_path(dt.MODELS_DIR, context.allocator)
	if err != nil {
		fmt.printfln("no %s/ yet: run an experiment (make dt-table1, make dt-fusion, ...)", dt.MODELS_DIR)
		return
	}
	for e in entries {
		if !strings.has_suffix(e.name, ".safetensors") do continue
		meta, ok := ml.read_meta(e.fullpath)
		fmt.printfln("%s  (%.1f MB)", e.fullpath, f64(e.size) / 1e6)
		if !ok {
			fmt.println("  (unreadable)")
			continue
		}
		for kv in meta do fmt.printfln("  %-20s %s", kv.key, kv.value)
	}
}
