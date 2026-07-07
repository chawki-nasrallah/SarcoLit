"""Retrieval metrics for v0.3-eval (pure, no API/GPU).

Each function scores one query given its ranked retrieved PMIDs and the set of
relevant PMIDs. ``evaluate`` aggregates over a whole query set at several k.

- recall@k : fraction of the relevant docs that appear in the top-k.
             (For known-item queries with a single relevant doc, this is 1 if
             found in the top-k else 0.)
- MRR      : mean reciprocal rank — 1/rank of the *first* relevant doc (0 if
             none retrieved). Rewards ranking the answer high.
- nDCG@k   : normalised discounted cumulative gain — ranking quality with a
             log discount, normalised so a perfect ranking scores 1.0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    found = len(set(ranked[:k]) & relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    for i, pmid in enumerate(ranked, 1):
        if pmid in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    # Binary relevance; discount by log2(rank+1).
    return sum(
        (1.0 if pmid in relevant else 0.0) / math.log2(i + 1)
        for i, pmid in enumerate(ranked[:k], 1)
    )


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    dcg = dcg_at_k(ranked, relevant, k)
    # Ideal DCG: all relevant docs packed at the top.
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    results: list[tuple[Sequence[str], Iterable[str]]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Aggregate metrics over a query set.

    ``results`` is a list of ``(ranked_pmids, relevant_pmids)`` per query.
    Returns recall@k and nDCG@k for each k, plus a single MRR.
    """
    pairs = [(ranked, set(rel)) for ranked, rel in results]

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = _mean([recall_at_k(r, rel, k) for r, rel in pairs])
        metrics[f"ndcg@{k}"] = _mean([ndcg_at_k(r, rel, k) for r, rel in pairs])
    metrics["mrr"] = _mean([reciprocal_rank(r, rel) for r, rel in pairs])
    return metrics
