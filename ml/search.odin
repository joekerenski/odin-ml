package ml

// ============================================================================
// Kernel choices found by timing on the device (GEMM tiles and split-K today),
// cached in memory and on disk so only the first run of a shape pays:
//
//   ~/.cache/odin-ml/<device>.txt      one "key value" line per choice
//
// ML_SEARCH=0 skips searching (heuristics, plus whatever the cache already
// has). Keys name the kernel kind and everything its speed depends on.
// ============================================================================

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"

search_enabled := true

// Tests: run every GEMM with this backend variant (≥ 0; its own numbering,
// e.g. Metal: tiles, direct tiles, small) where it applies, instead of the
// chosen one. -1: off.
gemm_force := -1

@(private)
Choices :: struct {
	device: string, // cache file name; "" until a backend sets it
	loaded: bool,
	table:  map[string]int,
}

@(private)
choices: Choices

// Set by a GPU backend at init: the device's name names its cache file.
choices_open :: proc(device: string) {
	if choices.device == device do return
	choices.device = strings.clone(device, scratch())
	choices.loaded = false
	if choices.table == nil do choices.table = make(map[string]int, scratch())
	clear(&choices.table)
}

@(private)
choices_path :: proc() -> string {
	home, _ := os.lookup_env_alloc("HOME", context.temp_allocator)
	name, _ := strings.replace_all(choices.device, " ", "-", context.temp_allocator)
	return fmt.tprintf("%s/.cache/odin-ml/%s.txt", home, name)
}

@(private)
choices_load :: proc() {
	choices.loaded = true
	data, err := os.read_entire_file_from_path(choices_path(), context.temp_allocator)
	if err != nil do return
	text := string(data)
	for line in strings.split_lines_iterator(&text) {
		sp := strings.last_index_byte(line, ' ')
		if sp < 0 do continue
		if v, ok := strconv.parse_int(line[sp + 1:]); ok do choices.table[strings.clone(line[:sp], scratch())] = v
	}
}

choice_get :: proc(key: string) -> (int, bool) {
	if choices.device == "" do return 0, false
	if !choices.loaded do choices_load()
	v, ok := choices.table[key]
	return v, ok
}

choice_put :: proc(key: string, v: int) {
	if choices.device == "" do return
	choices.table[strings.clone(key, scratch())] = v
	path := choices_path()
	os.make_directory_all(path[:strings.last_index_byte(path, '/')])
	f, err := os.open(path, {.Write, .Create, .Append})
	if err != nil do return
	defer os.close(f)
	os.write_string(f, fmt.tprintf("%s %d\n", key, v))
}
