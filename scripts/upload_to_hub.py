


import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    parser = argparse.ArgumentParser(description="Upload a PEFT adapter to the Hugging Face Hub.")
    parser.add_argument("--adapter_path", type=str, default="outputs/stata_lora_adapter", help="Path to the trained PEFT adapter.")
    parser.add_argument("--repo_name", type=str, required=True, help="Name of the repository on the Hugging Face Hub (e.g., 'your-username/your-model-name').")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Name of the base model from the Hugging Face Hub.")
    parser.add_argument("--hf_token", type=str, default=None, help="Your Hugging Face API token (if not already logged in).")

    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading PEFT adapter from: {args.adapter_path}")
    # Make sure we are loading the final adapter, not a checkpoint
    if "checkpoint" in os.path.basename(os.path.normpath(args.adapter_path)):
        print("Warning: You might be loading a checkpoint. It's recommended to load the final adapter.")

    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print(f"Pushing adapter to Hugging Face Hub repository: {args.repo_name}")
    # The `push_to_hub` method on a PeftModel will only upload the adapter weights and config
    model.push_to_hub(args.repo_name, use_auth_token=args.hf_token)
    tokenizer.push_to_hub(args.repo_name, use_auth_token=args.hf_token)

    print("\nAdapter uploaded successfully!")
    print(f"You can now load your adapter using: model.load_adapter('your-username/{args.repo_name}')")

if __name__ == "__main__":
    main()


