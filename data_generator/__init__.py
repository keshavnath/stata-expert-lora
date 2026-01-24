
"""Utilities for generating synthetic STATA datasets.

Exports:
- `generate`: local chunking and file-to-JSONL utilities
- `expand`: expansion helpers (variant generation)
- `prompts`: prompt templates
"""

from . import generate, expand, prompts

__all__ = ["generate", "expand", "prompts"]
