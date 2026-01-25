"""Dry-run validation for training script and config.

Checks performed:
- Load `configs/train_config.yaml` and print main fields
- Verify dataset path existence and read first JSON line if present
- Run `prepare_tokenized_dataset` from `train_unsloth_qlora.py` using a fake tokenizer
"""
import json
from pathlib import Path
import yaml
import sys

print("Dry-run validation starting...")
# Load config
cfg_path = Path("configs/train_config.yaml")
if not cfg_path.exists():
    print(f"ERROR: config file not found: {cfg_path}")
    sys.exit(2)
with cfg_path.open("r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}

print("Loaded config keys:", list(cfg.keys()))
base_model = cfg.get("model", {}).get("base_model")
print("Base model:", base_model)
print("LoRA rank:", cfg.get("model", {}).get("lora_rank"))
print("LoRA alpha:", cfg.get("model", {}).get("lora_alpha"))
print("Training settings:", cfg.get("training", {}))

# Check dataset path
data_path = cfg.get("data", {}).get("train_path")
if not data_path:
    print("No data.train_path set in config; skipping dataset checks.")
else:
    dp = Path(data_path)
    print("Configured train path:", dp)
    if not dp.exists():
        print("Dataset file does not exist; skipping dataset read.")
    else:
        print("Dataset file exists; reading first line to validate JSON...")
        with dp.open("r", encoding="utf-8") as fh:
            first = fh.readline()
            try:
                obj = json.loads(first)
                print("First line parsed OK. Example keys:", list(obj.keys()))
            except Exception as e:
                print("Failed to parse first JSON line:", e)

# Validate tokenization step using local function
try:
    import train_unsloth_qlora as trainer_mod
except Exception as e:
    print("Failed to import train_unsloth_qlora:", e)
    sys.exit(3)

from datasets import Dataset
examples = [{"instruction": "Merge datasets on id and collapse by year", "input": "", "output": "merge 1:1 id using other.dta\ncollapse (mean) x, by(year)"}]
small_ds = Dataset.from_list(examples)

class FakeTokenizer:
    def __init__(self):
        self.eos_token = ""
    def __call__(self, text, truncation=True, max_length=1024):
        # simplistic fake tokenization
        toks = text.split()[:max_length]
        ids = [len(w) for w in toks]
        return {"input_ids": ids, "attention_mask": [1]*len(ids)}
    def decode(self, ids, skip_special_tokens=True):
        return "decoded string"

fake_tok = FakeTokenizer()
try:
    toked = trainer_mod.prepare_tokenized_dataset(small_ds, fake_tok, max_length=512)
    sample = toked[0]
    print("Tokenization produced keys:", list(sample.keys()))
    if "labels" in sample:
        print("Labels present; sample label length:", len(sample["labels"]))
    print("prepare_tokenized_dataset ran successfully.")
except Exception as e:
    print("prepare_tokenized_dataset failed:", e)
    sys.exit(4)

print("Dry-run validation completed successfully.")
