import os
os.environ.setdefault("PYTHONPATH", ".")

"""UnsloTh-based inference runner (two-pass: base -> finetuned LoRA).

This script uses `unsloth.FastLanguageModel.from_pretrained(..., load_in_4bit=True)`
and runs two passes over `data/unsloth/val.jsonl`: first with the base model,
then reloads and applies the LoRA adapter and runs again. Results are saved to
`outputs/inference/base_outputs.jsonl` and `outputs/inference/finetuned_outputs.jsonl`.

Designed for constrained GPUs (e.g. RTX 3050 Ti, 4GB VRAM). Uses 4-bit
loading and `FastLanguageModel.for_inference(model)` when available. Uses
`torch.float16` for inference tensors.
"""

import argparse
import json
import gc
from pathlib import Path

try:
    import torch
except Exception as e:
    raise SystemExit("torch is required for this script: %s" % e)

try:
    from unsloth import FastLanguageModel
except Exception:
    raise SystemExit("unsloth is required for this script. Install or ensure PYTHONPATH includes unsloth.")

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

ALPACA_PROMPT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def formatting_prompts_func(example: dict) -> str:
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    return ALPACA_PROMPT.format(instruction=instr, input=inp)


def read_dataset(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def write_outputs(path: Path, items, outputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for orig, out in zip(items, outputs):
            obj = {"instruction": orig.get("instruction", orig.get("prompt", "")), "output": out}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_with_model(model, tokenizer, prompts, device, max_new_tokens=256, batch_size=1):
    outputs = []
    
    # Enable Unsloth's 2x faster inference mode
    FastLanguageModel.for_inference(model) 

    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            
            # Tokenize and move ONLY the input tensors to the device
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            
            # Generate using Unsloth's optimized engine
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id # Prevents warnings
            )
            
            # Decode only the NEW tokens (removing the prompt)
            # Unsloth generate returns the full sequence (prompt + response)
            input_len = inputs.input_ids.shape[1]
            decoded = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)
            
            outputs.extend([text.strip() for text in decoded])
            
            # Explicit cleanup for 4GB VRAM
            del inputs, gen
            torch.cuda.empty_cache()
            
    return outputs


def free_cuda(model=None, tokenizer=None):
    try:
        if model is not None:
            del model
    except Exception:
        pass
    try:
        if tokenizer is not None:
            del tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Base model HF id or local path (used with unsloth).")
    parser.add_argument("--lora", help="Path to LoRA adapter directory (optional)", default="outputs/stata_lora_adapter")
    parser.add_argument("--data", help="Validation JSONL file", default="data/unsloth/val.jsonl")
    parser.add_argument("--out-dir", help="Output directory", default="outputs/inference")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size (keep 1 for 4GB VRAM)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Validation file not found: {data_path}")

    items = read_dataset(data_path)
    prompts = [formatting_prompts_func(it) for it in items]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_out_path = out_dir / "base_outputs.jsonl"
    fine_out_path = out_dir / "finetuned_outputs.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_name = args.base or (os.environ.get("BASE_MODEL") or "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    print(f"Loading base model {base_model_name} in 4-bit for inference (may take a moment)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=1024,
        load_in_4bit=True,
    )

    # Ensure half precision where possible
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=1024,
    load_in_4bit=True,
    device_map="auto", # Ensures it lands on your GPU immediately
    )
    
    FastLanguageModel.for_inference(model)

    print("Running base model generation...")
    base_outputs = generate_with_model(model, tokenizer, prompts, device=device, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    write_outputs(base_out_path, items, base_outputs)

    # Free memory completely before loading LoRA-applied model
    free_cuda(model, tokenizer)

    # Second pass: load base again and apply LoRA
    print("Reloading base model and applying LoRA adapter...")
    model2, tokenizer2 = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=1024,
        load_in_4bit=True,
    )

    if args.lora:
        if PeftModel is None:
            raise SystemExit("PEFT is required to load LoRA adapters (install peft)")
        print(f"Applying LoRA adapter from {args.lora}...")
        model2 = PeftModel.from_pretrained(model2, args.lora)

    try:
        model2.to(torch.float16)
    except Exception:
        pass

    print("Running finetuned model generation...")
    fine_outputs = generate_with_model(model2, tokenizer2, prompts, device=device, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    write_outputs(fine_out_path, items, fine_outputs)

    # Final cleanup
    free_cuda(model2, tokenizer2)

    print("Done. Wrote:")
    print(f" - Base outputs: {base_out_path}")
    print(f" - Finetuned outputs: {fine_out_path}")


if __name__ == "__main__":
    main()
