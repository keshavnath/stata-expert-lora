---
library_name: transformers
tags:
- code
- stata
base_model:
- Qwen/Qwen2.5-Coder-1.5B-Instruct
pipeline_tag: text-generation
datasets:
  - keshavnath/stata-code-explanations
---

# Model Card for keshavnath/Qwen2.5-STATA (STATA-Expert-LoRA)

A LoRA fine-tuned model of `Qwen/Qwen2.5-Coder-1.5B-Instruct` for generating Stata code from natural language instructions.

## Model Details

### Model Description

This is a LoRA (Low-Rank Adaptation) fine-tuned model based on `Qwen/Qwen2.5-Coder-1.5B-Instruct`. It has been trained on a dataset of Stata commands to improve its ability to function as a Stata coding assistant. The fine-tuning was performed using QLoRA, a memory-efficient fine-tuning technique, with the `unsloth` library.

- **Developed by:** keshavnath
- **Model type:** Causal Language Model
- **Language(s) (NLP):** English, Stata
- **License:** Apache 2.0
- **Finetuned from model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`

## Uses

### Direct Use

This model is intended for generating Stata code snippets from natural language prompts. It can be used as a programming assistant for Stata users.

**Example prompt:** "Generate code to merge two datasets on the variable id and then collapse by year"

### Out-of-Scope Use

This model is not a general-purpose chatbot and has not been trained for tasks other than Stata code generation. It may produce incorrect output for prompts unrelated to Stata.

## Bias, Risks, and Limitations

The model's output reflects the data it was trained on. It does not have a true understanding of statistical theory and may generate code that is syntactically correct but statistically inappropriate. **Always review and validate the generated code before use.**

## How to Get Started with the Model

Use the code below to load the model directly from the Hugging Face Hub.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "keshavnath/Qwen2.5-STATA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# --- Inference Example ---
instruction = "Generate code to merge two datasets on the variable id and then collapse by year"
prompt_template = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n\n\n"
    "### Response:\n"
)
prompt = prompt_template.format(instruction=instruction)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(decoded_output)
```

## Training Details

### Training Data

The model was trained on a dataset derived from Stata command examples, formatted in the Alpaca instruction-following style. The dataset is available on the Hugging Face Hub: [keshavnath/stata-code-instructions](https://huggingface.co/datasets/keshavnath/stata-code-instructions).

### Training Procedure

The model was fine-tuned using QLoRA in 4-bit precision with the `unsloth` library.

#### Training Hyperparameters

- **Training regime:** QLoRA with `fp16` mixed precision
- **lora_rank:** 32
- **lora_alpha:** 32
- **optimizer:** `paged_adamw_8bit`
- **learning_rate:** 2e-4
- **epochs:** 3
- **max_steps:** 100
- **batch_size:** 1
- **gradient_accumulation_steps:** 8

## Evaluation

The model was evaluated using an LLM-as-a-judge using `llama-3.3-70b-instruct`, scoring the correctness and idiomatic usage of the generated Stata code on a scale of 0-10.

### Results

The fine-tuned model showed a substantial improvement over the base model.

| Model      | Average Score |
|------------|---------------|
| Base       | 1.33          |
| Fine-tuned | 8.67          |

## Technical Specifications

### Compute Infrastructure

- **Hardware:** Single CUDA-enabled GPU
- **Software:** `unsloth`, `peft`, `transformers`, `torch`

