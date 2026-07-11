"""Run the retrieval evaluation (Step 2 of v0.3-eval).

Loads a query set (synthetic or gold), retrieves top-k for each query, and
computes recall@k / MRR / nDCG via :mod:`sarcolit.eval.metrics`. Because the
metrics are computed at several k from one top-max_k retrieval per query, this
also *is* the k-sweep.

No API needed (retrieval + metrics only). The Retriever is imported lazily so
this module can be imported (and the runner unit-tested with a fake retriever)
without pulling in torch/qdrant.

Run:  ``uv run python -m sarcolit.eval.retrieval --queries evals/synthetic_queries.jsonl``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sarcolit.eval.metrics import evaluate


def load_query_set(path: Path) -> list[dict]:
    """Load a query set. Accepts synthetic (``relevant_pmid``) or gold
    (``relevant_pmids``) rows; normalises to ``{query, relevant: [str, ...]}``."""
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            relevant = d.get("relevant_pmids") or [d["relevant_pmid"]]
            items.append({"query": d["query"], "relevant": [str(p) for p in relevant]})
    return items


def run_retrieval_eval(
    query_path: Path,
    ks: Sequence[int] = (1, 3, 5, 10),
    retriever=None,
    rerank: bool = False,
    hybrid: bool = False,
) -> tuple[dict[str, float], int]:
    """Retrieve for every query and return (metrics, n_queries)."""
    items = load_query_set(query_path)
    max_k = max(ks)

    created = retriever is None
    if created:
        if rerank:
            from sarcolit.retrieval.rerank import RerankingRetriever

            retriever = RerankingRetriever()
        elif hybrid:
            from sarcolit.retrieval.hybrid import HybridRetriever

            retriever = HybridRetriever()
        else:
            from sarcolit.retrieval.search import Retriever  # lazy: avoids torch

            retriever = Retriever()

    try:
        results = []
        for it in items:
            hits = retriever.search(it["query"], k=max_k)
            ranked = [h.pmid for h in hits]
            results.append((ranked, it["relevant"]))
    finally:
        if created:
            retriever.close()

    return evaluate(results, ks=ks), len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run retrieval eval over a query set.")
    ap.add_argument("--queries", type=Path, default=Path("evals/synthetic_queries.jsonl"))
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--rerank", action="store_true", help="add cross-encoder reranking")
    ap.add_argument("--hybrid", action="store_true", help="dense + BM25 (RRF fusion)")
    args = ap.parse_args()

    metrics, n = run_retrieval_eval(
        args.queries, ks=tuple(args.ks), rerank=args.rerank, hybrid=args.hybrid
    )
    mode = "bge + rerank" if args.rerank else "bge + BM25 (hybrid)" if args.hybrid else "bge only"
    print(f"retrieval eval over {n} queries ({mode}):\n")
    for key, val in metrics.items():
        print(f"  {key:12s} {val:.3f}")


if __name__ == "__main__":
    main()
