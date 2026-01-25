"""Minimal inference loader for base model + LoRA adapter.

This script is a template that uses `transformers` + `peft` to load a base model and
apply a LoRA adapter (the adapter path is the output from training). Adjust
paths and tokens for your environment.
"""
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def load_model(base_name, lora_path=None):
    tok = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(base_name, low_cpu_mem_usage=True)
    if lora_path and PeftModel is not None:
        model = PeftModel.from_pretrained(model, lora_path)
    return tok, model


def generate(tok, model, prompt, max_new_tokens=256):
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model checkpoint")
    parser.add_argument("--lora", help="Path to LoRA adapter (optional)")
    parser.add_argument("--prompt", help="Prompt to generate", default="regress y x1 x2, robust")
    args = parser.parse_args()

    tok, model = load_model(args.base, args.lora)
    out = generate(tok, model, args.prompt)
    print(out)


if __name__ == "__main__":
    main()
