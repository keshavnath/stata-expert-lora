"""Evaluation utilities: syntax validity and LLM-as-a-judge wrapper.

Usage: run `evaluate.py --base outputs/baseline.jsonl --finetuned outputs/finetuned.jsonl`
Each file is a JSONL with fields {"instruction":..., "output":...} where `output` is model-generated STATA code.
"""
import re
import json
import argparse
from pathlib import Path
from statistics import mean

STATA_KEYWORDS = [
    r"\bregress\b",
    r"\bxtreg\b",
    r"\bgen\b",
    r"\bforeach\b",
    r"\blocal\b",
    r"\breshape\b",
    r"\bmerge\b",
    r"\btsset\b",
]


def syntax_validity(code: str) -> float:
    s = code.lower()
    scores = [1 if re.search(p, s) else 0 for p in STATA_KEYWORDS]
    return mean(scores) if scores else 0.0


def read_outputs(path: Path):
    items = []
    for line in open(path, "r", encoding="utf-8"):
        obj = json.loads(line)
        items.append(obj)
    return items


def judge_via_openrouter(prompt: str, base_out: str, fine_out: str, api_key: str, model: str = None):
    # Minimal wrapper that asks OpenRouter to score base vs fine on a 1-10 scale.
    # Returns (base_score, fine_score)
    import os
    import requests

    url = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system = "You are an expert STATA programmer and judge."
    user = (
        "Rate the following two STATA code outputs for the same instruction on a scale 1-10,"
        " where 10 is perfect, focusing on correctness and idiomatic STATA usage.\n\n"
        f"Base output:\n{base_out}\n\nFinetuned output:\n{fine_out}\n\n"
        "Respond with JSON: {\"base\": <score>, \"finetuned\": <score>}"
    )
    payload = {"model": model or "gpt-4o-mini", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return {"base": None, "finetuned": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--openrouter-key", help="Optional: use OpenRouter as judge")
    args = parser.parse_args()

    base = read_outputs(Path(args.base))
    fine = read_outputs(Path(args.finetuned))
    n = min(len(base), len(fine))
    synt_scores = []
    for i in range(n):
        b = base[i].get("output", "")
        f = fine[i].get("output", "")
        synt_scores.append((syntax_validity(b), syntax_validity(f)))
    avg_base = mean([s[0] for s in synt_scores]) if synt_scores else 0
    avg_fine = mean([s[1] for s in synt_scores]) if synt_scores else 0
    print(f"Syntax validity — base: {avg_base:.3f}, finetuned: {avg_fine:.3f}")

    if args.openrouter_key:
        jud_scores = []
        for i in range(n):
            b = base[i].get("output", "")
            f = fine[i].get("output", "")
            try:
                res = judge_via_openrouter(base[i].get("instruction", ""), b, f, args.openrouter_key)
                jud_scores.append((res.get("base"), res.get("finetuned")))
            except Exception as e:
                print(f"Judge call failed for item {i}: {e}")
        print("Sample judge scores:", jud_scores[:5])


if __name__ == "__main__":
    main()
