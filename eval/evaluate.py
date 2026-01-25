"""LLM-as-a-judge evaluation using OpenRouter; writes a JSONL report.

Usage: `python eval/evaluate.py --base outputs/base_outputs.jsonl --finetuned outputs/finetuned_outputs.jsonl --openrouter-key $KEY`
Each output file is a JSONL with fields {"instruction":..., "output":...}.
This script will call OpenRouter for each pair and write a JSONL report with
fields {"instruction","base_score","finetuned_score","base_output","finetuned_output"}.
"""
import os
import json
import argparse
import logging
import re
from pathlib import Path
from statistics import mean

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
from dotenv import load_dotenv


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def read_outputs(path: Path):
    items = []
    for line in open(path, "r", encoding="utf-8"):
        obj = json.loads(line)
        items.append(obj)
    return items


def judge_via_openrouter(prompt: str, base_out: str, fine_out: str, api_key: str, model: str = None):
    import os
    import requests

    url = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system = "You are an expert STATA programmer and judge."
    user = (
        "For the following instruction, compare the two STATA code outputs.\n"
        "Rate each output on an integer scale 0-10 (0=very poor, 10=perfect), focusing on correctness and idiomatic STATA usage.\n\n"
        f"Instruction:\n{prompt}\n\n"
        f"Base output:\n{base_out}\n\nFinetuned output:\n{fine_out}\n\n"
        "Respond WITH ONLY a JSON object (no surrounding text) with exactly these three keys:\n"
        "{\"base\": <int 0-10>, \"finetuned\": <int 0-10>, \"explanation\": <string explanation>}\n"
        "The scores must be integers between 0 and 10 inclusive. The explanation should be a short plain-text justification."
    )
    payload = {
        "model": model or os.getenv("OPENROUTER_MODEL"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return {"base": None, "finetuned": None, "explanation": None}

    # strip possible markdown/code fences
    def _strip_codeblock(s: str) -> str:
        s = s.strip()
        # remove triple backticks and optional language markers
        if s.startswith("```") and s.endswith("```"):
            # remove first and last fence
            parts = s.split("\n")
            # try to remove the opening fence line and closing fence
            if parts[0].startswith("```"):
                parts = parts[1:]
            if parts and parts[-1].startswith("```"):
                parts = parts[:-1]
            s = "\n".join(parts).strip()
        # also remove single backticks
        s = s.strip('` \n')
        return s

    content = _strip_codeblock(content)

    # Expect JSON; parse and validate structure
    try:
        parsed = json.loads(content)
    except Exception:
        # If not JSON, fail gracefully with None values
        return {"base": None, "finetuned": None, "explanation": content}

    # Validate keys
    base_val = parsed.get("base")
    fine_val = parsed.get("finetuned")
    expl = parsed.get("explanation")

    def _to_int_score(x):
        try:
            xi = int(x)
        except Exception:
            try:
                xi = int(float(x))
            except Exception:
                return None
        # clamp
        if xi < 0:
            xi = 0
        if xi > 10:
            xi = 10
        return xi

    base_score = _to_int_score(base_val)
    finetuned_score = _to_int_score(fine_val)
    if expl is None:
        expl = ""

    return {"base": base_score, "finetuned": finetuned_score, "explanation": str(expl)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Base outputs JSONL", default="outputs/inference/base_outputs.jsonl")
    parser.add_argument("--finetuned", help="Finetuned outputs JSONL", default="outputs/inference/finetuned_outputs.jsonl")
    parser.add_argument("--openrouter-key", help="OpenRouter API key (optional; falls back to .env or env vars)")
    parser.add_argument("--openrouter-model", help="OpenRouter model (optional; .env or env vars fallback)")
    parser.add_argument("--out-report", help="JSONL report path", default="outputs/eval/judge_report.jsonl")
    parser.add_argument("--max-samples", type=int, help="Max number of samples to judge (for cost control)")
    args = parser.parse_args()

    openrouter_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    openrouter_model = args.openrouter_model or os.environ.get("OPENROUTER_MODEL")

    if not openrouter_key:
        raise SystemExit("OpenRouter API key not found. Provide --openrouter-key or set OPENROUTER_API_KEY in env or .env")

    base = read_outputs(Path(args.base))
    fine = read_outputs(Path(args.finetuned))
    n = min(len(base), len(fine))
    if args.max_samples:
        n = min(n, args.max_samples)

    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    jud_scores = []
    logging.basicConfig(level=logging.INFO)
    with open(report_path, "w", encoding="utf-8") as rf:
        for i in tqdm(range(n), desc="Judging"):
            b_out = base[i].get("output", "")
            f_out = fine[i].get("output", "")
            instr = base[i].get("instruction", base[i].get("prompt", ""))
            try:
                res = judge_via_openrouter(instr, b_out, f_out, openrouter_key, model=openrouter_model)
                base_score = res.get("base")
                fine_score = res.get("finetuned")
                explanation = res.get("explanation")
            except Exception as e:
                logging.exception("Judge call failed for item %s", i)
                base_score = None
                fine_score = None
                explanation = None
            jud_scores.append((base_score, fine_score))
            entry = {
                "index": i,
                "instruction": instr,
                "base_score": base_score,
                "finetuned_score": fine_score,
                "base_output": b_out,
                "finetuned_output": f_out,
                "judge_explanation": explanation,
            }
            rf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Compute averages where available
    base_vals = [s[0] for s in jud_scores if isinstance(s[0], (int, float))]
    fine_vals = [s[1] for s in jud_scores if isinstance(s[1], (int, float))]
    avg_base = mean(base_vals) if base_vals else None
    avg_fine = mean(fine_vals) if fine_vals else None
    print(f"Wrote judge report to {report_path}")
    print(f"Average judge scores — base: {avg_base}, finetuned: {avg_fine}")
    # write a short README summary next to the report
    summary_path = report_path.parent / "README.md"
    with open(summary_path, "w", encoding="utf-8") as mf:
        mf.write("# Evaluation summary\n\n")
        mf.write(f"Report path: {report_path}\n\n")
        mf.write(f"Samples evaluated: {n}\n")
        mf.write(f"Average base score: {avg_base}\n")
        mf.write(f"Average finetuned score: {avg_fine}\n\n")
        mf.write("See the JSONL report for per-sample scores and outputs.\n")
    print(f"Wrote summary README to {summary_path}")


if __name__ == "__main__":
    main()
