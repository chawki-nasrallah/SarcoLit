"""Semantic search over the qdrant index.

Step 3 of ``v0.2-rag-baseline``. Wraps "embed the query → search qdrant →
return typed hits" behind a small ``Retriever`` that loads the embedding model
and opens the index once, so repeated searches are cheap. Step 4 (generation)
calls ``Retriever.search`` to get the abstracts it grounds Qwen's answer in.

Run a one-off search:
    ``uv run python -m sarcolit.retrieval.search "your question here"``
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

from sarcolit.retrieval.embedding import Embedder
from sarcolit.retrieval.index import COLLECTION, QDRANT_PATH

if TYPE_CHECKING:
    from qdrant_client.models import ScoredPoint


@dataclass
class SearchHit:
    """One retrieved abstract plus its similarity score."""

    pmid: str
    score: float
    title: str
    abstract: str
    year: int | None
    journal: str
    authors: list[str]
    doi: str | None


def _to_hit(point: ScoredPoint) -> SearchHit:
    p = point.payload or {}
    return SearchHit(
        pmid=str(p.get("pmid", point.id)),
        score=point.score,
        title=p.get("title", ""),
        abstract=p.get("abstract", ""),
        year=p.get("year"),
        journal=p.get("journal", ""),
        authors=p.get("authors", []),
        doi=p.get("doi"),
    )


class Retriever:
    """Holds the embedder + qdrant client so searches reuse both.

    ``embedder`` and ``client`` are injectable for testing (a stub embedder and
    an in-memory client need no GPU or on-disk index).
    """

    def __init__(
        self,
        qdrant_path: Path = QDRANT_PATH,
        collection: str = COLLECTION,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection = collection
        self.embedder = embedder or Embedder()
        self.client = client or QdrantClient(path=str(qdrant_path))

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Return the ``k`` abstracts most similar in meaning to ``query``."""
        qvec = self.embedder.embed_query(query).tolist()
        result = self.client.query_points(
            collection_name=self.collection,
            query=qvec,
            limit=k,
            with_payload=True,
        )
        return [_to_hit(p) for p in result.points]

    def close(self) -> None:
        self.client.close()


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('usage: python -m sarcolit.retrieval.search "your question"')
        return
    retriever = Retriever()
    try:
        for hit in retriever.search(query, k=5):
            print(f"[{hit.score:.3f}] ({hit.year}) {hit.title}")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
