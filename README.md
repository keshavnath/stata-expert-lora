Stata-Expert-LoRA
=================

Minimal personal project scaffold for generating a STATA-focused synthetic dataset and later QLoRA fine-tuning.

Quick start
-----------

- Create a Python venv and activate it (in WSL2 if using GPU):

```bash
python -m venv .venv
source .venv/bin/activate  # WSL/Linux
```
Dependencies & tools:

- This personal project uses `uv` for dependency management. 

- Put your OpenRouter API key in `.env` as `OPENROUTER_API_KEY`.

- Generate a small sample dataset (downloads code + reverse-prompts):

```bash
python run_data_generator.py --count 10 --source-list urls.txt
```

Files added
- `data_generator/` — minimal API wrapper + prompt templates
- `run_data_generator.py` — simple CLI to create JSONL
- `requirements.txt`, `.gitignore`

Next steps
- Review `data_generator/generate.py` and run with your key to produce the full dataset.

Cheatsheet input format
- Provide a JSONL file (one JSON object per line). Each object must include a `code` field containing the STATA code snippet.
	Optional fields: `instruction` or `explanation` which provide the original description from your cheat-sheet.

Example line:

```json
{"code": "regress y x1 x2, robust", "explanation": "Run OLS of y on x1 and x2 with robust SEs."}
```

Expansion workflow
- Use `data_generator/expand.py` to expand each `code` into multiple prompt variants using OpenRouter:

```powershell
python -m data_generator.expand --in data/cheatsheet.jsonl --out data/expanded.jsonl --variants 4
```

This will produce `data/expanded.jsonl` where each line contains the original fields plus a `variants` array with the generated prompts.
