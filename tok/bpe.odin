package main

Pair :: struct {
	a, b: u32,
}

Node :: struct {
	id:      u32,
	prev:    i32,
	next:    i32,
	deleted: bool,
}

BPE :: struct {
	merges: [dynamic]Pair,
	vocab:  [dynamic][]u8,
}

bpe_init :: proc(b: ^BPE) {
	b.merges = make([dynamic]Pair, 0)
	b.vocab = make([dynamic][]u8, 256)
	for i in 0 ..< 256 {
		b.vocab[i] = make([]u8, 1)
		b.vocab[i][0] = u8(i)
	}
}

bpe_vocab_size :: proc(b: ^BPE) -> int {
	return 256 + len(b.merges)
}

add_pos :: proc(pos: ^map[Pair][dynamic]i32, p: Pair, idx: i32) {
	arr := pos[p]
	append(&arr, idx)
	pos[p] = arr
}

inc_pair :: proc(counts: ^map[Pair]int, pos: ^map[Pair][dynamic]i32, p: Pair, idx: i32) {
	counts[p] += 1
	add_pos(pos, p, idx)
}

dec_pair :: proc(counts: ^map[Pair]int, p: Pair) {
	n := counts[p] - 1
	if n <= 0 {
		delete_key(counts, p)
	} else {
		counts[p] = n
	}
}

best_from_counts :: proc(counts: map[Pair]int) -> (Pair, int, bool) {
	best: Pair
	best_n := -1
	for p, n in counts {
		if n > best_n ||
		   (n == best_n && (p.a < best.a || (p.a == best.a && p.b < best.b))) {
			best = p
			best_n = n
		}
	}
	return best, best_n, best_n >= 2
}

bpe_train :: proc(data: []u8, vocab_size: int) -> (BPE, []u32) {
	assert(vocab_size >= 256, "bpe_train: vocab_size < 256")
	b: BPE
	bpe_init(&b)
	n := len(data)
	if n == 0 do return b, {}

	nodes := make([]Node, n)
	defer delete(nodes)
	for i in 0 ..< n {
		prv: i32 = i == 0 ? -1 : i32(i - 1)
		nxt: i32 = i == n - 1 ? -1 : i32(i + 1)
		nodes[i] = Node{id = u32(data[i]), prev = prv, next = nxt}
	}

	counts: map[Pair]int
	pos: map[Pair][dynamic]i32
	defer delete(counts)
	defer {
		for _, arr in pos do delete(arr)
		delete(pos)
	}
	for i in 0 ..< n - 1 {
		p := Pair{nodes[i].id, nodes[i + 1].id}
		inc_pair(&counts, &pos, p, i32(i))
	}

	for 256 + len(b.merges) < vocab_size {
		p, _, ok := best_from_counts(counts)
		if !ok do break
		z := u32(256 + len(b.merges))
		append(&b.merges, p)
		va, vb := b.vocab[p.a], b.vocab[p.b]
		nz := make([]u8, len(va) + len(vb))
		copy(nz, va)
		copy(nz[len(va):], vb)
		append(&b.vocab, nz)

		occ := pos[p]
		for idx in occ {
			if idx < 0 || int(idx) >= n do continue
			node := &nodes[idx]
			if node.deleted || node.id != p.a do continue
			nxt := node.next
			if nxt < 0 do continue
			nb := &nodes[nxt]
			if nb.deleted || nb.id != p.b do continue

			prv := node.prev
			aft := nb.next
			if prv >= 0 do dec_pair(&counts, Pair{nodes[prv].id, p.a})
			dec_pair(&counts, p)
			if aft >= 0 do dec_pair(&counts, Pair{p.b, nodes[aft].id})

			node.id = z
			nb.deleted = true
			node.next = aft
			if aft >= 0 do nodes[aft].prev = idx

			if prv >= 0 do inc_pair(&counts, &pos, Pair{nodes[prv].id, z}, prv)
			if aft >= 0 do inc_pair(&counts, &pos, Pair{z, nodes[aft].id}, idx)
		}
	}

	packed := make([dynamic]u32, 0, n)
	for i: i32 = 0; i >= 0; i = nodes[i].next {
		append(&packed, nodes[i].id)
	}
	return b, packed[:]
}

merge_in_place :: proc(ids: []u32, p: Pair, z: u32) -> []u32 {
	n := len(ids)
	j := 0
	i := 0
	for i < n {
		if i + 1 < n && ids[i] == p.a && ids[i + 1] == p.b {
			ids[j] = z
			j += 1
			i += 2
		} else {
			ids[j] = ids[i]
			j += 1
			i += 1
		}
	}
	return ids[:j]
}

bpe_encode :: proc(b: ^BPE, data: []u8) -> []u32 {
	ids := make([dynamic]u32, len(data))
	for i in 0 ..< len(data) do ids[i] = u32(data[i])
	n := len(ids)
	for i in 0 ..< len(b.merges) {
		z := u32(256 + i)
		merged := merge_in_place(ids[:n], b.merges[i], z)
		n = len(merged)
	}
	out := make([]u32, n)
	copy(out, ids[:n])
	return out
}

bpe_decode :: proc(b: ^BPE, ids: []u32) -> []u8 {
	total := 0
	for id in ids do total += len(b.vocab[id])
	out := make([]u8, total)
	o := 0
	for id in ids {
		v := b.vocab[id]
		copy(out[o:], v)
		o += len(v)
	}
	return out
}
