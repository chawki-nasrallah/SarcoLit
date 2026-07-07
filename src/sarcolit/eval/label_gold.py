"""Interactive labelling tool to build the gold eval set (v0.3-eval).

For each seed query it retrieves a deep candidate pool (top-N via the retriever)
and asks you — the domain expert — to mark which candidates are relevant. Writes
``evals/gold_queries.jsonl``. Resumable: already-labelled queries are skipped, so
you can label in chunks.

Pooling caveat: only the retrieved pool is judged; anything not surfaced is
assumed irrelevant. Pool is deep (top-20) to mitigate; broaden with keyword
search once v0.4 adds it.

Run:  ``uv run python -m sarcolit.eval.label_gold``
Controls per query: type the relevant numbers (e.g. ``1 4 9``); ``n`` = none
relevant; ``s`` = skip this query for now; ``q`` = save and quit.
"""

from __future__ import annotations

import json
from pathlib import Path

SEED_PATH = Path("evals/gold_queries_seed.jsonl")
OUT_PATH = Path("evals/gold_queries.jsonl")
POOL_SIZE = 20
SNIPPET = 260


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _parse_selection(raw: str, n: int) -> list[int] | None:
    """Parse '1 4 9' into [0,3,8]. Returns None on invalid; [] for 'none'."""
    raw = raw.strip().lower()
    if raw in ("n", ""):
        return []
    picks = []
    for tok in raw.replace(",", " ").split():
        if not tok.isdigit() or not (1 <= int(tok) <= n):
            return None
        picks.append(int(tok) - 1)
    return sorted(set(picks))


def label() -> None:
    seed = _load_jsonl(SEED_PATH)
    done = {r["query"] for r in _load_jsonl(OUT_PATH)}
    todo = [q for q in seed if q["query"] not in done]
    if not todo:
        print(f"All {len(seed)} queries already labelled -> {OUT_PATH}")
        return

    print(f"{len(done)} done, {len(todo)} to label. Pool size {POOL_SIZE}.")
    print("Per query: relevant numbers (e.g. '1 4 9') | 'n' none | 's' skip | 'q' quit.\n")

    from sarcolit.retrieval.search import Retriever  # lazy: heavy import

    retriever = Retriever()
    try:
        with OUT_PATH.open("a", encoding="utf-8") as fh:
            for qi, item in enumerate(todo, 1):
                hits = retriever.search(item["query"], k=POOL_SIZE)
                print("=" * 78)
                print(f"[{qi}/{len(todo)}] QUERY ({item.get('subtopic', '')}): {item['query']}\n")
                for i, h in enumerate(hits, 1):
                    print(f"  [{i:2d}] ({h.year}) {h.title}")
                    print(f"       {h.abstract[:SNIPPET].strip()}...")
                while True:
                    raw = input("\n  relevant #s: ")
                    if raw.strip().lower() == "q":
                        print(f"\nsaved; quit. -> {OUT_PATH}")
                        return
                    if raw.strip().lower() == "s":
                        print("  skipped.\n")
                        break
                    picks = _parse_selection(raw, len(hits))
                    if picks is None:
                        print("  invalid — enter numbers in range, 'n', 's', or 'q'.")
                        continue
                    relevant = [hits[i].pmid for i in picks]
                    fh.write(
                        json.dumps(
                            {
                                "query": item["query"],
                                "relevant_pmids": relevant,
                                "subtopic": item.get("subtopic", ""),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fh.flush()
                    print(f"  saved {len(relevant)} relevant.\n")
                    break
    finally:
        retriever.close()
    print(f"done -> {OUT_PATH}")


if __name__ == "__main__":
    label()
