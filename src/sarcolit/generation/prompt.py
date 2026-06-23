"""Prompt assembly for grounded generation (pure, no model/torch).

Kept separate from ``generate.py`` so the prompt logic — the part that actually
decides answer quality and grounding — is unit-testable without loading Qwen.

The system prompt is the grounding contract: answer only from the supplied
abstracts, cite PMIDs, and decline when the context is insufficient rather than
hallucinate. For a biomedical assistant, "I don't know" is a correct answer.
"""

from __future__ import annotations

from typing import Protocol


class Passage(Protocol):
    """Structural type for anything carrying an abstract (e.g. SearchHit)."""

    pmid: str
    title: str
    abstract: str
    year: int | None


SYSTEM_PROMPT = (
    "You are a biomedical research assistant specialised in sarcopenia and "
    "age-related muscle health. Answer the user's question using ONLY the "
    "numbered abstracts provided as context.\n"
    "Rules:\n"
    "- Base every statement on the abstracts; do not use outside knowledge.\n"
    "- After each claim, cite the supporting source(s) by PMID, "
    "e.g. [PMID: 12345678].\n"
    "- If the abstracts do not contain enough information to answer, say so "
    "plainly instead of guessing.\n"
    "- Be concise and precise."
)


def format_context(passages: list[Passage]) -> str:
    """Render retrieved abstracts as a numbered context block."""
    blocks = []
    for i, p in enumerate(passages, 1):
        year = f", {p.year}" if p.year else ""
        blocks.append(f"[{i}] (PMID: {p.pmid}{year}) {p.title}\n{p.abstract}")
    return "\n\n".join(blocks)


def build_messages(question: str, passages: list[Passage]) -> list[dict[str, str]]:
    """Build the chat messages: grounding system prompt + context + question."""
    context = format_context(passages)
    user = f"Context — abstracts retrieved from the corpus:\n\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
