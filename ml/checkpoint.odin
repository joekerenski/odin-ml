package ml

// ============================================================================
// Checkpoints as safetensors: an 8-byte header length, a JSON header naming
// each tensor (dtype, shape, byte range) plus free-form string metadata, then
// the raw little-endian f32 data. The same file loads in numpy, torch, MLX and
// tinygrad, so a trained model is never locked into this library.
//
//   ml.save("models/dt5.safetensors", params[:], meta = {{"k", "5"}, {"steps", "10000"}})
//   meta, ok := ml.load("models/dt5.safetensors", params[:])
//
// Tensors are matched by name; without names they are "000", "001", ... in
// param order, so a model loads back into the same architecture built the same
// way. load writes into the existing param buffers (optimizers keep working)
// and refuses files whose names or shapes don't match.
// ============================================================================

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:slice"
import "core:strings"

Meta :: struct {
	key, value: string,
}

save :: proc(path: string, params: []^Tensor, names: []string = nil, meta: []Meta = nil) -> bool {
	assert(names == nil || len(names) == len(params))
	backend.sync()
	h := strings.builder_make(scratch())
	defer strings.builder_destroy(&h)
	strings.write_byte(&h, '{')
	if len(meta) > 0 {
		strings.write_string(&h, `"__metadata__":{`)
		for m, i in meta {
			if i > 0 do strings.write_byte(&h, ',')
			write_json_string(&h, m.key)
			strings.write_byte(&h, ':')
			write_json_string(&h, m.value)
		}
		strings.write_byte(&h, '}')
	}
	offset := 0
	for p, i in params {
		if i > 0 || len(meta) > 0 do strings.write_byte(&h, ',')
		write_json_string(&h, param_name(names, i))
		strings.write_string(&h, `:{"dtype":"F32","shape":[`)
		for d, j in p.shape {
			if j > 0 do strings.write_byte(&h, ',')
			strings.write_int(&h, int(d))
		}
		size := len(p.data) * size_of(f32)
		fmt.sbprintf(&h, `],"data_offsets":[%d,%d]}}`, offset, offset + size)
		offset += size
	}
	strings.write_byte(&h, '}')
	for strings.builder_len(h) % 8 != 0 do strings.write_byte(&h, ' ') // keeps the data 8-byte aligned

	header := strings.to_string(h)
	buf := make([]byte, 8 + len(header) + offset, scratch())
	defer delete(buf, scratch())
	(^u64le)(&buf[0])^ = u64le(len(header))
	copy(buf[8:], header)
	at := 8 + len(header)
	for p in params {
		copy(buf[at:], slice.to_bytes(p.data))
		at += len(p.data) * size_of(f32)
	}
	if err := os.write_entire_file(path, buf); err != nil {
		fmt.eprintfln("ml.save %s: %v", path, err)
		return false
	}
	return true
}

// Load tensors into params (same names/shapes). meta is allocated with
// context.allocator.
load :: proc(path: string, params: []^Tensor, names: []string = nil) -> (meta: []Meta, ok: bool) {
	assert(names == nil || len(names) == len(params))
	buf, err := os.read_entire_file_from_path(path, scratch())
	if err != nil {
		fmt.eprintfln("ml.load %s: %v", path, err)
		return
	}
	defer delete(buf, scratch())
	header, data, entries, hok := parse_safetensors(buf)
	if !hok {
		fmt.eprintfln("ml.load %s: not a safetensors file", path)
		return
	}
	defer json.destroy_value(header, scratch())

	backend.sync()
	for p, i in params {
		name := param_name(names, i)
		e, found := entries[name]
		if !found {
			fmt.eprintfln("ml.load %s: no tensor %q", path, name)
			return
		}
		info := e.(json.Object)
		if info["dtype"].(json.String) != "F32" || !shape_matches(info["shape"].(json.Array), p.shape) {
			fmt.eprintfln("ml.load %s: tensor %q is %v %v, model wants F32 %v", path, name, info["dtype"], info["shape"], p.shape)
			return
		}
		offs := info["data_offsets"].(json.Array)
		lo, hi := int(offs[0].(json.Integer)), int(offs[1].(json.Integer))
		if hi - lo != len(p.data) * size_of(f32) || hi > len(data) {
			fmt.eprintfln("ml.load %s: tensor %q has a bad byte range", path, name)
			return
		}
		copy(slice.to_bytes(p.data), data[lo:hi])
	}
	return read_meta_from(entries), true
}

// Metadata only (for listing a folder of models without loading them).
read_meta :: proc(path: string) -> (meta: []Meta, ok: bool) {
	buf, err := os.read_entire_file_from_path(path, scratch())
	if err != nil do return
	defer delete(buf, scratch())
	header, _, entries, hok := parse_safetensors(buf)
	if !hok do return
	defer json.destroy_value(header, scratch())
	return read_meta_from(entries), true
}

meta_get :: proc(meta: []Meta, key: string) -> (string, bool) {
	for m in meta do if m.key == key do return m.value, true
	return "", false
}

@(private = "file")
parse_safetensors :: proc(buf: []byte) -> (header: json.Value, data: []byte, entries: json.Object, ok: bool) {
	if len(buf) < 8 do return
	n := int((^u64le)(&buf[0])^)
	if n <= 0 || 8 + n > len(buf) do return
	v, jerr := json.parse(buf[8:][:n], parse_integers = true, allocator = scratch())
	if jerr != nil do return
	obj, is_obj := v.(json.Object)
	if !is_obj {
		json.destroy_value(v, scratch())
		return
	}
	return v, buf[8 + n:], obj, true
}

@(private = "file")
read_meta_from :: proc(entries: json.Object) -> []Meta {
	m, has := entries["__metadata__"].(json.Object)
	if !has do return nil
	out := make([dynamic]Meta, 0, len(m))
	for k, v in m {
		if s, is_str := v.(json.String); is_str {
			append(&out, Meta{strings.clone(k), strings.clone(s)})
		}
	}
	slice.sort_by(out[:], proc(a, b: Meta) -> bool { return a.key < b.key })
	return out[:]
}

@(private = "file")
param_name :: proc(names: []string, i: int) -> string {
	if names != nil do return names[i]
	return fmt.tprintf("%03d", i)
}

@(private = "file")
shape_matches :: proc(a: json.Array, shape: []i32) -> bool {
	if len(a) != len(shape) do return false
	for v, i in a {
		n, is_int := v.(json.Integer)
		if !is_int || n != i64(shape[i]) do return false
	}
	return true
}

@(private = "file")
write_json_string :: proc(b: ^strings.Builder, s: string) {
	strings.write_byte(b, '"')
	for c in transmute([]byte)s {
		switch c {
		case '"', '\\':
			strings.write_byte(b, '\\')
			strings.write_byte(b, c)
		case '\n':
			strings.write_string(b, `\n`)
		case 0 ..< 0x20:
			fmt.sbprintf(b, `\u%04x`, c)
		case:
			strings.write_byte(b, c)
		}
	}
	strings.write_byte(b, '"')
}

