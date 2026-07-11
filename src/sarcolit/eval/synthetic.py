"""Generate a synthetic known-item retrieval eval set (Step 1 of v0.3-eval).

For each sampled abstract, Claude (Opus) writes one realistic question the
abstract answers; that abstract's PMID is the known-relevant target. Retrieval
metrics (recall@k / MRR / nDCG) then measure whether search surfaces the source
abstract for its question.

Only the eval *tooling* uses Claude here; the RAG system itself stays local.
The generated set is written once and committed to ``evals/`` — a fixed,
versioned benchmark (generation isn't deterministic, so we freeze the output).

Run:  ``uv run --env-file .env python -m sarcolit.eval.synthetic --n 150``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from anthropic import Anthropic

MODEL = "claude-opus-4-8"
CORPUS_PATH = Path("data/sarcolit-corpus-v0.1/corpus.jsonl")
OUT_PATH = Path("evals/synthetic_queries.jsonl")

PROMPT = """You are building a retrieval-evaluation set for a search engine over \
sarcopenia and muscle-aging research abstracts.

Given ONE PubMed abstract, write ONE realistic question a clinician or \
researcher might type into the search engine, for which THIS abstract would be \
a relevant answer.

Requirements:
- Answerable from this abstract's content.
- Do NOT quote the abstract verbatim or copy its phrasing — write a natural \
information need, as a real user would ask.
- Specific enough that this abstract is a strong match.
- Also give a short subtopic tag (e.g. "diagnosis", "EMG", "nutrition", \
"imaging", "epidemiology").

Return ONLY JSON: {{"question": "...", "subtopic": "..."}}

Title: {title}
Abstract: {abstract}"""


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    """Read the JSONL corpus (kept local so this module needs no torch/qdrant)."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _parse_json(text: str) -> dict:
    """Parse the first JSON object in the response, ignoring any trailing text."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON found in response: {text[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def generate_question(client: Anthropic, record: dict) -> dict:
    """Ask Claude for one realistic question this abstract would answer."""
    prompt = PROMPT.format(title=record["title"], abstract=record["abstract"])
    msg = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    data = _parse_json(text)
    return {
        "query": data["question"],
        "relevant_pmid": record["pmid"],
        "subtopic": data.get("subtopic", ""),
    }


def build(n: int, seed: int = 0, corpus_path: Path = CORPUS_PATH, out_path: Path = OUT_PATH) -> int:
    """Sample n abstracts (deterministically) and generate a question for each."""
    records = load_corpus(corpus_path)
    sample = random.Random(seed).sample(records, min(n, len(records)))
    client = Anthropic()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(sample, 1):
            item = generate_question(client, rec)
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(sample)}] ({item['subtopic']}) {item['query'][:70]}")
    return len(sample)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic retrieval eval queries.")
    ap.add_argument("--n", type=int, default=150, help="number of abstracts to sample")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (reproducible)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    n = build(args.n, seed=args.seed, out_path=args.out)
    print(f"wrote {n} synthetic queries -> {args.out}")


if __name__ == "__main__":
    main()
