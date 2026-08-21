package main

import "core:fmt"
import "core:math/rand"
import "core:os"
import "core:strconv"
import "core:time"

Bag :: struct {
	next:  map[u32]int,
	total: int,
}

main :: proc() {
	path := "data/shakespeare.txt"
	vocab_size := 8192
	n_gen := 200
	if len(os.args) > 1 do path = os.args[1]
	if len(os.args) > 2 {
		if v, ok := strconv.parse_int(os.args[2]); ok do vocab_size = v
	}

	data, err := os.read_entire_file_from_path(path, context.allocator)
	assert(err == nil, "failed to read text")
	fmt.printfln("text: %s  %d bytes  vocab=%d", path, len(data), vocab_size)

	t0 := time.tick_now()
	b, ids := bpe_train(data, vocab_size)
	dt := time.duration_seconds(time.tick_since(t0))
	fmt.printfln("trained %d merges  vocab=%d  %.2fs", len(b.merges), bpe_vocab_size(&b), dt)
	fmt.printfln("tokens: %d  compression=%.3fx", len(ids), f64(len(data)) / f64(max(len(ids), 1)))

	uni: Bag
	bi: map[u32]Bag
	tri: map[Pair]Bag
	uni.next = make(map[u32]int)
	for i in 0 ..< len(ids) {
		bag_add(&uni, ids[i])
		if i >= 1 {
			bag := bi[ids[i - 1]]
			if bag.next == nil do bag.next = make(map[u32]int)
			bag_add(&bag, ids[i])
			bi[ids[i - 1]] = bag
		}
		if i >= 2 {
			ctx := Pair{ids[i - 2], ids[i - 1]}
			bag := tri[ctx]
			if bag.next == nil do bag.next = make(map[u32]int)
			bag_add(&bag, ids[i])
			tri[ctx] = bag
		}
	}

	prompt := transmute([]u8)string("To be")
	ctx_ids := bpe_encode(&b, prompt)
	rand.reset(42)
	out := make([dynamic]u32, 0, len(ctx_ids) + n_gen)
	append(&out, ..ctx_ids)
	for _ in 0 ..< n_gen {
		n := len(out)
		next: u32
		if n >= 2 {
			bag := tri[Pair{out[n - 2], out[n - 1]}]
			if bag.total > 0 {
				next = bag_sample(bag)
				append(&out, next)
				continue
			}
		}
		if n >= 1 {
			bag := bi[out[n - 1]]
			if bag.total > 0 {
				next = bag_sample(bag)
				append(&out, next)
				continue
			}
		}
		append(&out, bag_sample(uni))
	}

	text := bpe_decode(&b, out[:])
	fmt.println("--- sample ---")
	fmt.println(string(text))
}

bag_add :: proc(bag: ^Bag, id: u32) {
	bag.next[id] += 1
	bag.total += 1
}

bag_sample :: proc(bag: Bag) -> u32 {
	assert(bag.total > 0)
	r := int(rand.int31_max(i32(bag.total)))
	for id, c in bag.next {
		r -= c
		if r < 0 do return id
	}
	return 0
}
