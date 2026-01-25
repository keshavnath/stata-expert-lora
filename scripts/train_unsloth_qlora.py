"""Train Qwen2.5-Coder with QLoRA via Unsloth (with peft fallback).

Usage:
  python scripts/train_unsloth_qlora.py --data stata_dataset.json --output stata-lora-adapter

Requirements:
  - unsloth (preferred) OR transformers + peft + bitsandbytes
  - datasets
  - accelerate (for device mapping)

This script follows your requirements: uses FastLanguageModel.from_pretrained(..., load_in_4bit=True),
LoRA r=32 alpha=32 targeting common linear proj modules, optimizer paged_adamw_8bit, batch/accum settings,
and saves adapters to `stata-lora-adapter`.
"""
import os
import sys
from pathlib import Path

os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["RAY_CHUNKS"] = "1" 
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
# Force Dynamo to ignore compilation errors and fall back to "eager" mode
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# Optional: Disable Inductor specifically (the source of the triton_key error)
import multiprocess
# multiprocess.set_start_method('spawn', force=True)

# Force the Unsloth cache and project root into the path
current_dir = os.getcwd()
cache_dir = os.path.join(current_dir, "unsloth_compiled_cache")

if cache_dir not in sys.path:
    sys.path.insert(0, cache_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Disable tokenizer multiprocessing which causes issues with uv + windows
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml

ALPACA_PROMPT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def make_prompt(example: dict) -> str:
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    return ALPACA_PROMPT.format(instruction=instr, input=inp, output=out)


def load_dataset_file(path: str):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")
    return ds


def prepare_tokenized_dataset(dataset, tokenizer, max_length=1024):
    def map_fn(ex):
        prompt = make_prompt(ex)
        tokenized = tokenizer(prompt, truncation=True, max_length=max_length)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(map_fn, remove_columns=dataset.column_names, batched=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to YAML train config")
    parser.add_argument("--data", help="Path to stata_dataset.json (Alpaca format)")
    parser.add_argument("--output", help="Where to save LoRA adapters")
    parser.add_argument("--base-model", help="Base model id")
    parser.add_argument("--max-steps", type=int, help="Max steps for quick test")
    parser.add_argument("--epochs", type=int, help="Fallback epochs if max_steps not used")
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    args = parser.parse_args()

    # Load config file and apply defaults; CLI args override config values when provided
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    def cfg_get(*keys, default=None):
        d = cfg
        for k in keys:
            if d is None:
                return default
            d = d.get(k)
        return d if d is not None else default

    out_dir = Path(args.output or cfg_get("output", "out_dir") or "stata_lora_adapter")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use unsloth FastModel + FastLanguageModel + TRL SFTTrainer API
    from unsloth import FastModel, FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # Load base model + tokenizer in 4-bit (FastModel returns model, tokenizer)
    print("Loading base model + tokenizer in 4-bit (this may take a moment)...")
    base_model = args.base_model or cfg_get("model", "base_model") or "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    max_seq = int(cfg_get("data", "max_length") or cfg_get("training", "max_seq_length") or 1024)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,
        max_seq_length = max_seq,
        load_in_4bit = True,
    )

    # Configure and apply LoRA via unsloth helper (use dropout=0 for best performance)
    lora_r = int(cfg_get("model", "lora_rank") or 32)
    lora_alpha = int(cfg_get("model", "lora_alpha") or 32)
    lora_dropout = 0

    print(f"Patching model with LoRA r={lora_r}, alpha={lora_alpha} (dropout={lora_dropout})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(cfg_get("training", "seed") or 3407),
        max_seq_length=max_seq,
        use_rslora=False,
        loftq_config=None,
    )

    # Build TRL SFTConfig
    train_cfg = cfg_get("training") or {}
    sft_args = SFTConfig(
        max_seq_length=max_seq,
        per_device_train_batch_size=int(args.per_device_batch_size or cfg_get("training", "batch_size") or 2),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps or cfg_get("training", "gradient_accumulation_steps") or 4),
        warmup_steps=int(cfg_get("training", "warmup_steps") or 0),
        max_steps=int(args.max_steps or cfg_get("training", "max_steps") or 100),
        logging_steps=int(cfg_get("training", "logging_steps") or 10),
        output_dir=str(out_dir),
        optim=cfg_get("training", "optim") or "paged_adamw_8bit",
        seed=int(cfg_get("training", "seed") or 3407),
        packing=False,
        dataset_num_proc = 1,  # This disables the multi-process "map" operation
    )

    # Load dataset and use formatting function (do NOT pre-tokenize)
    print("Loading dataset and preparing text field...")
    data_path = args.data or cfg_get("data", "train_path")
    if data_path is None:
        raise RuntimeError("No data path provided via CLI or config (data.train_path)")
    ds = load_dataset_file(data_path)

    # Use tokenizer returned by FastLanguageModel.from_pretrained
    try:
        tokenizer = tokenizer
    except NameError:
        # tokenizer should have been returned by from_pretrained earlier; if not, raise
        raise RuntimeError("Tokenizer not available from FastLanguageModel.from_pretrained")

    def formatting_prompts_func(examples):
        instructions = examples.get("instruction", [])
        inputs = examples.get("input", [])
        outputs = examples.get("output", [])
        texts = []
        for instruction, inp, out in zip(instructions, inputs, outputs):
            text = ALPACA_PROMPT.format(instruction=instruction, input=inp, output=out)
            texts.append(text)
        return {"text": texts}

    ds = ds.map(formatting_prompts_func, batched=True, num_proc=1)

    # Create trainer and train using TRL SFTTrainer (use dataset_text_field)
    print("Starting training (QLoRA) with TRL SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        args=sft_args,
    )

    trainer.train()

    # Save LoRA adapters
    print(f"Saving LoRA adapters to {out_dir}...")
    try:
        model.save_pretrained(str(out_dir))
    except Exception:
        from peft import PeftModel

        model.save_pretrained(str(out_dir))

    # Quick inference test
    print("Running short inference test with adapters applied...")
    test_prompt = (
        "Generate code to merge two datasets on the variable id and then collapse by year"
    )

    # Inference: switch to optimized inference mode (do NOT reload model)
    print("Running inference (switching to inference mode)...")
    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        # best-effort: continue if method not present
        pass

    import torch
    inputs = tokenizer(
        [ALPACA_PROMPT.format(instruction=test_prompt, input="", output="")],
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("--- Inference output ---")
    print(decoded[0])


if __name__ == "__main__":
    main()
