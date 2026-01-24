"""Prompt templates used by data generation tools.

Two templates are included:
- `REVERSE_PROMPT_*` — used to ask an LLM to convert STATA code into an
  instruction/input/output JSON suitable for fine-tuning.
- `CHUNK_METADATA_HELPER` — short heuristics guidance for extracting headings.
"""

REVERSE_PROMPT_SYSTEM = (
    "You are an assistant that, given a STATA code snippet, writes a concise and precise "
    "natural-language instruction describing what the code does, any required inputs, and returns a JSON object."
)

REVERSE_PROMPT_USER = (
    "Given the following STATA code, produce a JSON object with keys:\n"
    "- instruction: a short natural-language instruction suitable for fine-tuning,\n"
    "- input: an example input description (or empty string),\n"
    "- output: the original STATA code (preserve formatting).\n\n"
    "Respond ONLY with the JSON object.\n\nSTATA CODE:\n{code}\n"
)

CHUNK_METADATA_HELPER = (
    "Heuristic: section headings often appear as comments starting with `//` or `*` "
    "on the line(s) immediately preceding a block. Use them as `explanation`."
)
