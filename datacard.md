---
language: en
license: mit
---

# Dataset Card for Stata-Bot Dataset - Stata-code-explanations

## Dataset Description

This dataset contains pairs of instructions and responses for generating Stata code. It was used to fine-tune a language model to create Stata-Bot, a helpful assistant for Stata programming.

The dataset is provided in the Alpaca format.

- **`instruction`**: A natural language prompt describing a task in Stata.
- **`input`**: Additional context for the instruction (often empty).
- **`output`**: The corresponding Stata code that accomplishes the task.

## How to Use

You can load this dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("keshavnath/stata-code-explanations")

print(dataset)
```

## Dataset Curation

The data was generated from Stata cheat sheets and expanded using a language model. The goal was to create a diverse set of examples covering common Stata commands and use cases.
