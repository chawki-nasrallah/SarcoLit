"""Hybrid retrieval: dense (bge) + sparse (BM25), fused with RRF (v0.4).

Dense (bge) captures *semantic* matches (paraphrases, synonyms); sparse (BM25)
captures *exact-term* matches (acronyms, rare technical terms) the embedder can
blur. They're complementary, so fusing them can beat either alone.

Fusion is **Reciprocal Rank Fusion (RRF)** — it uses only *ranks*, not scores,
so it sidesteps the incompatible scales (cosine 0–1 vs unbounded BM25). Each doc
scores ``sum over lists of 1/(rrf_k + rank)``; a doc ranked high in either list
rises, and docs high in *both* rise most.

``HybridRetriever`` exposes the same ``search(query, k)`` interface as
``Retriever``, so it drops into ``RagPipeline`` and the eval runners unchanged.
Heavy imports (torch via Retriever, rank_bm25, numpy) are deferred so the module
imports light and ``rrf_fuse`` is unit-testable in CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sarcolit.retrieval.search import Retriever, SearchHit

CORPUS_PATH = Path("data/sarcolit-corpus-v0.1/corpus.jsonl")
FETCH_K = 30
RRF_K = 60

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf_fuse(rankings: list[list[str]], rrf_k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of several ranked ID lists → fused (id, score)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class BM25Index:
    """In-memory BM25 over the corpus (title + abstract)."""

    def __init__(self, records: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.pmids = [r["pmid"] for r in records]
        corpus = [_tokenize(f"{r['title']} {r['abstract']}") for r in records]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[str]:
        import numpy as np

        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [self.pmids[i] for i in top]


class HybridRetriever:
    """Dense (bge) + sparse (BM25) retrieval fused with RRF.

    ``base`` is injectable for testing; ``bm25`` too (else built from the corpus).
    """

    def __init__(
        self,
        base: Retriever | None = None,
        bm25: BM25Index | None = None,
        corpus_path: Path = CORPUS_PATH,
        fetch_k: int = FETCH_K,
        rrf_k: int = RRF_K,
    ) -> None:
        if base is None:
            from sarcolit.retrieval.search import Retriever

            base = Retriever()
        self.base = base
        with corpus_path.open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        self.by_pmid = {r["pmid"]: r for r in records}
        self.bm25 = bm25 if bm25 is not None else BM25Index(records)
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        from sarcolit.retrieval.search import SearchHit

        dense_pmids = [h.pmid for h in self.base.search(query, k=self.fetch_k)]
        sparse_pmids = self.bm25.search(query, self.fetch_k)
        fused = rrf_fuse([dense_pmids, sparse_pmids], self.rrf_k)

        hits = []
        for pmid, score in fused[:k]:
            r = self.by_pmid[pmid]
            hits.append(
                SearchHit(
                    pmid=pmid,
                    score=score,
                    title=r["title"],
                    abstract=r["abstract"],
                    year=r.get("year"),
                    journal=r.get("journal", ""),
                    authors=r.get("authors", []),
                    doi=r.get("doi"),
                )
            )
        return hits

    def close(self) -> None:
        self.base.close()
