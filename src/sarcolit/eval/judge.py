"""Faithfulness judge for v0.3-eval generation eval (Claude Opus as judge).

Given a question, the RAG answer, and the source abstracts it was grounded in,
Claude judges whether the answer's factual claims are actually *supported* by
those sources — the metric that catches mis-attribution like the knee-extension
case. Only the evaluator uses Claude; the RAG system under test stays local.

Sources are passed as dicts ``{pmid, title, abstract}`` (the runner converts
SearchHits). Faithfulness is scored per answer; the runner aggregates.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic

MODEL = "claude-opus-4-8"

JUDGE_PROMPT = """You are evaluating a biomedical RAG assistant for FAITHFULNESS \
(groundedness).

You are given a QUESTION, the assistant's ANSWER, and the SOURCE abstracts the \
assistant was given. Judge whether the answer's factual claims are supported by \
the sources.

Rules:
- "faithful" is true only if every substantive factual claim in the answer is \
supported by at least one source.
- A claim that is unsupported, contradicted, or attributed to a source that \
does not actually support it makes the answer NOT faithful (e.g. citing a \
knee-extension value as a handgrip value).
- Judge grounding only — ignore fluency and style.
- A correct "I don't know / insufficient information" answer is faithful.

Return ONLY JSON:
{{"faithful": true or false, "score": <0.0-1.0 fraction of claims supported>, \
"unsupported_claims": ["..."], "reasoning": "<one or two sentences>"}}

QUESTION:
{question}

SOURCES:
{sources}

ANSWER:
{answer}"""


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON found in judge response: {text[:200]!r}")
    return json.loads(match.group())


def format_sources(sources: list[dict]) -> str:
    """Render the source abstracts the same way the RAG system saw them."""
    return "\n\n".join(
        f"[{i}] (PMID: {s['pmid']}) {s['title']}\n{s['abstract']}" for i, s in enumerate(sources, 1)
    )


def judge_faithfulness(client: Anthropic, question: str, answer: str, sources: list[dict]) -> dict:
    """Ask Claude to score the answer's faithfulness against its sources."""
    prompt = JUDGE_PROMPT.format(question=question, sources=format_sources(sources), answer=answer)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return _parse_json(text)
