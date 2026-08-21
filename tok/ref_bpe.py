#!/usr/bin/env python3
"""Industrial BPE ref: HuggingFace `tokenizers` (ByteLevel BPE).

  uv run --with tokenizers tok/ref_bpe.py
  uv run --with tokenizers tok/ref_bpe.py --vocab 2048
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = ROOT / "data" / "shakespeare.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    ap.add_argument("--vocab", type=int, default=1024)
    args = ap.parse_args()

    data = args.text.read_bytes()
    print(f"text: {args.text}  {len(data):,} bytes  vocab={args.vocab}")

    tok = ByteLevelBPETokenizer()
    t0 = time.perf_counter()
    tok.train(files=[str(args.text)], vocab_size=args.vocab, min_frequency=2)
    dt = time.perf_counter() - t0

    ids = tok.encode(data.decode("utf-8", errors="replace")).ids
    sample = "To be, or not to be, that is the question:"
    enc = tok.encode(sample)
    back = tok.decode(enc.ids)

    print(f"trained in {dt:.3f}s  vocab={tok.get_vocab_size()}")
    print(f"tokens: {len(ids):,}  compression={len(data)/max(len(ids),1):.3f}x")
    print("sample tokens:")
    for t in enc.tokens:
        print(f"  {t!r}")
    print(f"roundtrip: {back!r}")
    print(f"ok={back.replace(' ', '') == sample.replace(' ', '') or sample in back or back.strip()==sample}")


if __name__ == "__main__":
    main()
