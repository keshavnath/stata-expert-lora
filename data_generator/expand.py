"""Expand provided STATA examples by asking OpenRouter to generate multiple prompt variants.

Input: a JSONL file where each line is an object with at least the key `code`.
Optional keys: `instruction` or `explanation` (used as context).

Output: JSONL with either `variants` lists or Alpaca-format examples ready for fine-tuning.

Usage:
  python -m data_generator.expand --in data/cheatsheet.jsonl --out data/expanded.jsonl --variants 4
  python -m data_generator.expand --in data/cheatsheet.jsonl --out data/alpaca.jsonl --variants 4 --alpaca
  python -m data_generator.expand --in data/cheatsheet.jsonl --out data/alpaca_dry.jsonl --variants 4 --alpaca --dry-run
"""
from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import requests
from .prompts import REVERSE_PROMPT_SYSTEM as PROMPT_SYSTEM_HELPER

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
# Allow selecting a model hosted on OpenRouter (set to a free/OSS model id if available)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")


class OpenRouterClient:
    def __init__(self, api_key: str, url: str = None):
        self.api_key = api_key
        self.url = url or OPENROUTER_URL

    def chat_completion(self, system: str, user: str, max_tokens: int = 512):
        from requests.exceptions import RequestException
        import time as _time

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except Exception:
                    try:
                        return data["choices"][0]["text"]
                    except Exception:
                        return json.dumps(data)
            except RequestException as e:
                last_err = e
                _time.sleep(2 ** attempt)
                continue

        # If we reach here, all retries failed; raise to be handled by caller
        raise last_err


SYSTEM_PROMPT = (
    "You are an assistant that, given a STATA code snippet and optional context,"
    " produces several distinct, natural-language prompts a user might give to request"
    " that same STATA code. Return a JSON object with key `variants` mapped to an array"
    " of unique prompt strings."
)

USER_TEMPLATE = (
    "STATA CODE:\n{code}\n\n"
    "Context (optional): {context}\n\n"
    "Produce {n} distinct, short user prompts/instructions that would request the above STATA code."
    " Return ONLY valid JSON: {{\"variants\": [..]}}"
)


def generate_variants(code: str, context: str, n: int, client: Optional[OpenRouterClient], dry_run: bool = False):
    user = USER_TEMPLATE.format(code=code, context=context or "", n=n)
    if client:
        try:
            resp = client.chat_completion(SYSTEM_PROMPT, user, max_tokens=512)
            parsed = json.loads(resp)
            variants = parsed.get("variants") or []
            if isinstance(variants, str):
                # sometimes returned as newline-separated strings
                variants = [v.strip() for v in variants.splitlines() if v.strip()]
            return variants
        except Exception as e:
            print(f"OpenRouter call failed: {e}")
    # Dry-run or fallback: create simple paraphrases deterministically
    if dry_run:
        base = (context or code).strip().replace('\n', ' ')
        variants = [f"{verb} {base}" for verb in ("Run", "Execute", "Provide STATA commands to", "Show the STATA code to")] 
        return variants[:n]
    return [f"Run the following STATA code: {code[:120]}..." for _ in range(n)]


def expand_file(in_path: str, out_path: str, variants_per: int = 4, dry_run: bool = False, alpaca: bool = False):
    client = OpenRouterClient(OPENROUTER_API_KEY) if OPENROUTER_API_KEY and not dry_run else None
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    count = 0
    with open(in_path, "r", encoding="utf-8") as inf, open(out_path, "w", encoding="utf-8") as outf:
        data = inf.read()
        decoder = json.JSONDecoder()
        idx = 0
        length = len(data)
        while True:
            # skip whitespace
            while idx < length and data[idx].isspace():
                idx += 1
            if idx >= length:
                break
            try:
                obj, end = decoder.raw_decode(data, idx)
            except Exception as e:
                # Can't decode at this position; try to advance to next line and continue
                next_nl = data.find('\n', idx)
                if next_nl == -1:
                    break
                idx = next_nl + 1
                print(f"Warning: skipping malformed JSON chunk near char {idx}: {e}")
                continue

            idx = end
            code = obj.get("code") or obj.get("output") or ""
            context = obj.get("instruction") or obj.get("explanation") or ""
            variants = generate_variants(code, context, variants_per, client, dry_run=dry_run)
            if alpaca:
                # include the original example as an Alpaca pair (use explanation as instruction)
                orig_instr = obj.get("explanation") or obj.get("instruction") or ""
                orig_ex = {"instruction": orig_instr, "input": "", "output": code}
                outf.write(json.dumps(orig_ex, ensure_ascii=False) + "\n")
                # then write one Alpaca-style example per generated variant
                for v in variants:
                    ex = {"instruction": v, "input": "", "output": code}
                    outf.write(json.dumps(ex, ensure_ascii=False) + "\n")
            else:
                obj["variants"] = variants
                outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expand cheatsheet examples into multiple prompt variants using OpenRouter.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input JSONL with `code` field per line")
    parser.add_argument("--out", dest="out_path", default="data/expanded.jsonl")
    parser.add_argument("--variants", dest="variants", type=int, default=4)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Do not call OpenRouter; use deterministic paraphrases")
    parser.add_argument("--alpaca", dest="alpaca", action="store_true", help="Output Alpaca-format examples (one per variant)")
    args = parser.parse_args()
    n = expand_file(args.in_path, args.out_path, variants_per=args.variants, dry_run=args.dry_run, alpaca=args.alpaca)
    print(f"Wrote {n} expanded examples to {args.out_path}")
