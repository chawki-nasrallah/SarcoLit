"""Grounded answer generation with 4-bit Qwen (Step 4 of v0.2-rag-baseline).

Loads ``Qwen/Qwen2.5-3B-Instruct`` quantized to 4-bit NF4 (fits ~2 GB VRAM, so
it runs on a 6 GB laptop GPU), then answers a question grounded in the abstracts
passed to ``answer()``. The retrieval half (which abstracts) lives in
``sarcolit.retrieval``; this module only turns (question + abstracts) into text.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sarcolit.generation.prompt import Passage, build_messages

MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _quant_config() -> BitsAndBytesConfig:
    """4-bit NF4 with double quantisation; bf16 compute (Ampere-friendly)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class Generator:
    """Loads the quantized model once; reuse across calls."""

    def __init__(self, model_name: str = MODEL) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=_quant_config(), device_map="auto"
        )

    @torch.no_grad()
    def answer(self, question: str, passages: list[Passage], max_new_tokens: int = 400) -> str:
        """Generate an answer grounded in ``passages`` (greedy / deterministic)."""
        messages = build_messages(question, passages)
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
