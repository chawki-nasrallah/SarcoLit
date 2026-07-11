"""Cross-encoder reranking for two-stage retrieval (v0.4-rag-improved).

Stage 1 (recall): the bge bi-encoder fetches top-``fetch_k`` fast (it embeds
query and doc separately, so docs are pre-computed).
Stage 2 (precision): a cross-encoder re-scores each ``[query, doc]`` pair jointly
— it sees query and document *together*, modelling fine distinctions a
bi-encoder misses — and reorders to the final top-k. Cross-encoders can't
pre-compute, so they only run on the ``fetch_k`` candidates.

``RerankingRetriever`` exposes the same ``search(query, k)`` interface as
``Retriever``, so it drops into ``RagPipeline`` and the eval runners unchanged.

Heavy imports (torch/transformers, and the base Retriever) are deferred so the
module imports light and the reordering logic is unit-testable with fakes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sarcolit.retrieval.search import Retriever, SearchHit

RERANK_MODEL = "BAAI/bge-reranker-base"
FETCH_K = 30


class Reranker:
    """Cross-encoder that scores ``[query, doc]`` pairs (higher = more relevant)."""

    def __init__(self, model_name: str = RERANK_MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        )

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Relevance score for each doc against the query (one forward pass)."""
        torch = self._torch
        with torch.no_grad():
            pairs = [[query, d] for d in docs]
            enc = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.view(-1)
            return logits.float().cpu().tolist()


class RerankingRetriever:
    """Two-stage retriever: bge fetch top-`fetch_k`, cross-encoder → top-k.

    ``base`` and ``reranker`` are injectable for testing (no GPU needed).
    """

    def __init__(
        self,
        base: Retriever | None = None,
        reranker: Reranker | None = None,
        fetch_k: int = FETCH_K,
    ) -> None:
        if base is None:
            from sarcolit.retrieval.search import Retriever

            base = Retriever()
        self.base = base
        self.reranker = reranker if reranker is not None else Reranker()
        self.fetch_k = fetch_k

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        candidates = self.base.search(query, k=self.fetch_k)
        if not candidates:
            return []
        docs = [f"{h.title}\n{h.abstract}" for h in candidates]
        scores = self.reranker.score(query, docs)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:k]]

    def close(self) -> None:
        self.base.close()
