### Stata-Expert-LoRA

Compact, technical repository for synthesizing STATA-focused prompts, QLoRA-style LoRA fine-tuning, low-VRAM inference, and LLM-based evaluation.
The fine-tuned model is available on the Hugging Face Hub: [keshavnath/Qwen2.5-STATA](https://huggingface.co/keshavnath/Qwen2.5-STATA)

---

### What this repo contains

* **Dataset generation:** Prompt templates and a small generator to turn STATA cheatsheets into JSONL training data.
* **Training:** QLoRA-style fine-tuning that produces a LoRA adapter (saved to `outputs/stata_lora_adapter`).
* **Inference:** A two-pass, 4-bit-friendly inference pipeline (base → free memory → apply LoRA → finetuned) designed for low-VRAM GPUs.
* **Evaluation:** LLM-as-judge evaluation that compares base vs finetuned outputs and writes structured JSONL reports.

### Design goals

* **Reproducible:** Data and training pipelines use JSONL artifacts so runs are scriptable and reviewable.
* **Low-VRAM friendly:** Inference uses 4-bit loading and memory cleanup to work on constrained GPUs (e.g., 4GB-class).
* **Auditable:** Evaluation produces per-sample JSON entries and explanations for downstream analysis.

### Technical Stack: Unsloth & PEFT

This project leverages the **Unsloth** library to achieve 2x faster training and 60% less memory usage.

* **QLoRA (4-bit):** We load the base model in 4-bit precision using `bitsandbytes` to stay within a 4GB VRAM budget.
* **PEFT (LoRA):** Instead of full-parameter fine-tuning, we train a lightweight adapter using **Parameter-Efficient Fine-Tuning**. This keeps the "knowledge" of the base model intact while teaching it STATA syntax.

---

### Quick start

1. **Environment Setup:** This project uses `uv` for lightning-fast dependency management.

```bash
# Sync environment and install dependencies from pyprojects.toml and uv.lock
uv sync

```

2. **Set credentials (OpenRouter judge):**

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env

```

See `.env.example` for details.

3. **Execution:** Use `uv run` to ensure scripts execute within the managed environment.

4. **Generate Data:**

```bash
uv run python -m data_generator.expand --in data/source/cheatsheet_data.jsonl --out data/unsloth/alpaca.jsonl --variants 4 --alpaca

```

For simple alpaca conversion without expansion:

```bash
uv run python -m data_generator.expand --in data/source/cheatsheet_data.jsonl --out data/unsloth/alpaca.jsonl --variants 0 --alpaca --dry-run

```

5. **Train Test Split:**

```bash
uv run scripts/train_test_split.py --in data/unsloth/alpaca.json

```

6. **Train QLoRA:**

```bash
uv run scripts/train_unsloth_qlora.py --config configs/train_config.yaml

```

7. **Run Inference:**

```bash
uv run inference/inference.py --lora outputs/stata_lora_adapter --out-dir outputs/inference

```

8. **Run Evaluation:**

```bash
uv run eval/evaluate.py

```

---

### Sample Results

* **Training data:** ~150 labeled STATA examples.
* **Approach:** LoRA via `peft` trained on Qwen2.5-Coder-1.5B; adapter saved to `outputs/stata_lora_adapter`.
* **Measured outcome:** Average base score **1.33** --> Finetuned score **8.67** on the validation set.
* See [outputs/eval/README.md](outputs/eval/README.md) and the full per-sample report at [outputs/eval/judge_report.jsonl](outputs/eval/judge_report.jsonl).

### Notes and dependencies

* Key Python libs: `torch`, `transformers`, `peft` (LoRA), `unsloth` (4-bit helpers), `requests`.
* The training step produces a LoRA adapter; the base model is loaded from the HF cache or remote model hub at inference time.

---