Stata-Expert-LoRA
=================

Compact, technical repository for synthesizing STATA-focused prompts, QLoRA-style LoRA fine-tuning, low-VRAM inference, and LLM-based evaluation.

What this repo contains
- Dataset generation: prompt templates and a small generator to turn STATA cheatsheets into JSONL training data.
- Training: QLoRA-style fine-tuning that produces a LoRA adapter (saved to `outputs/stata_lora_adapter`).
- Inference: a two-pass, 4-bit-friendly inference pipeline (base → free memory → apply LoRA → finetuned) designed for low-VRAM GPUs.
- Evaluation: LLM-as-judge evaluation that compares base vs finetuned outputs and writes structured JSONL reports.

Design goals
- Reproducible: data and training pipelines use JSONL artifacts so runs are scriptable and reviewable.
- Low-VRAM friendly: inference uses 4-bit loading and memory cleanup to work on constrained GPUs (e.g., 4GB-class).
- Auditable: evaluation produces per-sample JSON entries and explanations for downstream analysis.

Quick start
1. Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set credentials (OpenRouter judge):

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

3. Generate (optional), train, or run inference/evaluation:

Generate a small dataset (example):

Input format: a JSONL file where each line is an object with a `code` field (the STATA snippet) and an optional `explanation` or `instruction` field used as context.

```bash
# Expand an existing cheatsheet JSONL into multiple prompt variants (calls OpenRouter):
python -m data_generator.expand --in data/cheatsheet.jsonl --out data/expanded.jsonl --variants 4

# Produce Alpaca-format examples suitable for fine-tuning:
python -m data_generator.expand --in data/cheatsheet.jsonl --out data/alpaca.jsonl --variants 4 --alpaca

# Dry-run (deterministic paraphrases; no network calls):
python -m data_generator.expand --in data/cheatsheet.jsonl --out data/expanded_dry.jsonl --variants 4 --dry-run

# Optional: override OpenRouter model used by the expansion step:
python -m data_generator.expand --in data/cheatsheet.jsonl --out data/expanded.jsonl --variants 4 --model meta-llama/llama-3.3-70b-instruct:free
```

Run inference (two-pass base + LoRA). Example:

```bash
python inference/inference.py --lora outputs/stata_lora_adapter --out-dir outputs/inference
```

Run evaluation (OpenRouter judge):

```bash
python eval/evaluate.py
```

Notes and dependencies
- Key Python libs: `torch`, `transformers`, `peft` (LoRA), `unsloth` (4-bit helpers), `requests`.
- The training step produces a LoRA adapter; the base model is loaded from the HF cache or remote model hub at inference time.
- The evaluation script expects the judge key in `OPENROUTER_API_KEY` and writes `outputs/eval/judge_report.jsonl`.

Project outcomes (what's implemented)
- Synthetic dataset generator and prompt templates for STATA code.
- LoRA fine-tuning pipeline producing adapter artifacts.
- Low-VRAM, two-pass inference script producing base and finetuned JSONL outputs.
- Structured LLM-as-judge evaluation that enforces JSON scoring and preserves raw explanations for debug.
