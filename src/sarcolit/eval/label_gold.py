"""Interactive labelling tool to build the gold eval set (multi-method pool, v0.4).

For each seed query it builds a **multi-method candidate pool** — the union of
what bge (dense/semantic) retrieves and what BM25 (sparse/keyword) retrieves —
and asks you, the domain expert, to mark which candidates are relevant. Writes
``evals/gold_queries.jsonl``. Resumable: already-labelled queries are skipped.

Why the union: pooling from a single retriever is self-biased (a relevant doc it
misses never enters the pool). Pooling bge ∪ BM25 means every doc *either* method
surfaces gets judged, so the labels are a fair denominator for comparing bge vs
hybrid. (Docs no method surfaces are still assumed irrelevant — the residual,
smaller, pooling approximation.)

You judge *relevance only* (binary), not order and not any answer. The ranking
metrics get their order from the retriever under test, not from you.

Run:  ``uv run python -m sarcolit.eval.label_gold``
Controls per query: relevant numbers (e.g. ``1 4 9``); ``n`` none; ``s`` skip;
``q`` save and quit.
"""

from __future__ import annotations

import json
from pathlib import Path

SEED_PATH = Path("evals/gold_queries_seed.jsonl")
OUT_PATH = Path("evals/gold_queries.jsonl")
POOL_EACH = 15  # top-N from each method; union is the pool
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


def _pool(hybrid, query: str, pool_each: int) -> list:
    """Union of bge top-N and BM25 top-N as SearchHits (dense first, dedup)."""
    from sarcolit.retrieval.search import SearchHit

    dense = hybrid.base.search(query, k=pool_each)
    sparse_pmids = hybrid.bm25.search(query, pool_each)

    seen: set[str] = set()
    pool: list[SearchHit] = []
    for h in dense:
        if h.pmid not in seen:
            seen.add(h.pmid)
            pool.append(h)
    for pmid in sparse_pmids:
        if pmid in seen:
            continue
        seen.add(pmid)
        r = hybrid.by_pmid[pmid]
        pool.append(
            SearchHit(
                pmid=pmid,
                score=0.0,
                title=r["title"],
                abstract=r["abstract"],
                year=r.get("year"),
                journal=r.get("journal", ""),
                authors=r.get("authors", []),
                doi=r.get("doi"),
            )
        )
    return pool


def label() -> None:
    seed = _load_jsonl(SEED_PATH)
    done = {r["query"] for r in _load_jsonl(OUT_PATH)}
    todo = [q for q in seed if q["query"] not in done]
    if not todo:
        print(f"All {len(seed)} queries already labelled -> {OUT_PATH}")
        return

    print(f"{len(done)} done, {len(todo)} to label. Pool = bge + BM25 (top-{POOL_EACH} each).")
    print("Per query: relevant numbers (e.g. '1 4 9') | 'n' none | 's' skip | 'q' quit.")
    print("\nLoading retriever + building BM25 index (first launch ~30-60s)...", flush=True)

    from sarcolit.retrieval.hybrid import HybridRetriever  # lazy: heavy import

    hybrid = HybridRetriever()
    print("Ready.\n", flush=True)
    try:
        with OUT_PATH.open("a", encoding="utf-8") as fh:
            for qi, item in enumerate(todo, 1):
                pool = _pool(hybrid, item["query"], POOL_EACH)
                print("=" * 78)
                print(f"[{qi}/{len(todo)}] QUERY ({item.get('subtopic', '')}): {item['query']}")
                print(f"  ({len(pool)} candidates in pool)\n")
                for i, h in enumerate(pool, 1):
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
                    picks = _parse_selection(raw, len(pool))
                    if picks is None:
                        print("  invalid — enter numbers in range, 'n', 's', or 'q'.")
                        continue
                    relevant = [pool[i].pmid for i in picks]
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
        hybrid.close()
    print(f"done -> {OUT_PATH}")


if __name__ == "__main__":
    label()
