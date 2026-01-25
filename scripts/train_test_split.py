#!/usr/bin/env python3
"""Prepare Alpaca-style train/validation splits from an expanded JSONL.

Input: JSONL where each line contains {"instruction":..., "input":"", "output":...}
or objects with `variants` arrays; this script will produce `train.jsonl` and `val.jsonl`.
"""
import json
import random
from pathlib import Path
import argparse


def load_examples(in_path):
    examples = []
    decoder = json.JSONDecoder()
    with open(in_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                objs = [obj]
            except json.JSONDecodeError:
                # Try to handle concatenated JSON objects on a single line
                objs = []
                text = line
                pos = 0
                length = len(text)
                while pos < length:
                    try:
                        obj, next_pos = decoder.raw_decode(text, pos)
                        objs.append(obj)
                        pos = next_pos
                        # skip whitespace between objects
                        while pos < length and text[pos].isspace():
                            pos += 1
                    except Exception:
                        # give up parsing this line
                        print(f"Warning: failed to parse JSON chunk starting at pos {pos} in line: {line[:80]}...")
                        break

            for obj in objs:
                # If Alpaca output already: instruction/input/output
                if "instruction" in obj and "output" in obj:
                    examples.append({"instruction": obj.get("instruction", ""), "input": obj.get("input", ""), "output": obj.get("output", "")})
                    continue
                # If object with variants
                code = obj.get("code") or obj.get("output") or ""
                # include original explanation as one example if present
                expl = obj.get("explanation") or obj.get("instruction") or ""
                if expl:
                    examples.append({"instruction": expl, "input": "", "output": code})
                for v in obj.get("variants", []):
                    examples.append({"instruction": v, "input": "", "output": code})
    return examples


def split_and_write(examples, out_dir: Path, val_frac=0.05, seed=42):
    random.Random(seed).shuffle(examples)
    n_val = max(1, int(len(examples) * val_frac))
    val = examples[:n_val]
    train = examples[n_val:]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as t, open(out_dir / "val.jsonl", "w", encoding="utf-8") as v:
        for ex in train:
            t.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in val:
            v.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(train), len(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out-dir", dest="out_dir", default="data/unsloth")
    parser.add_argument("--val-frac", dest="val_frac", type=float, default=0.05)
    args = parser.parse_args()

    ex = load_examples(args.in_path)
    tr, va = split_and_write(ex, Path(args.out_dir), val_frac=args.val_frac)
    print(f"Wrote {tr} train and {va} val examples to {args.out_dir}")


if __name__ == "__main__":
    main()
