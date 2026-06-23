"""Build the qdrant vector index from the v0.1 corpus.

Step 2 of ``v0.2-rag-baseline``. Loads ``corpus.jsonl``, embeds each record's
title+abstract with the bge model, and upserts ``(id=PMID, vector, payload)``
points into a local qdrant collection. Retrieval (Step 3) searches this index.

qdrant runs in **local mode** — everything persists to ``data/qdrant/`` on
disk, no server/Docker needed.

Run it:  ``uv run python -m sarcolit.retrieval.index``
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sarcolit.retrieval.embedding import EMBED_DIM, Embedder

CORPUS_PATH = Path("data/sarcolit-corpus-v0.1/corpus.jsonl")
QDRANT_PATH = Path("data/qdrant")
COLLECTION = "sarcolit_v0_1"


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    """Read the JSONL corpus into a list of record dicts."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def doc_text(record: dict) -> str:
    """The text we embed for a record: title then abstract.

    The title carries strong topical signal, so we prepend it to the abstract
    rather than embedding the abstract alone.
    """
    return f"{record['title']}\n{record['abstract']}"


def build_index(
    records: list[dict],
    vectors: np.ndarray,
    client: QdrantClient,
    collection: str = COLLECTION,
) -> int:
    """(Re)create the collection and upsert one point per record.

    Pure plumbing — takes vectors already computed, so it's testable with fake
    vectors and an in-memory client. Returns the number of points indexed.
    """
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=int(rec["pmid"]), vector=vec.tolist(), payload=rec)
        for rec, vec in zip(records, vectors, strict=True)
    ]
    # Upsert in chunks to keep request sizes modest.
    for start in range(0, len(points), 256):
        client.upsert(collection_name=collection, points=points[start : start + 256])
    return client.count(collection_name=collection).count


def run_index(
    corpus_path: Path = CORPUS_PATH,
    qdrant_path: Path = QDRANT_PATH,
    collection: str = COLLECTION,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> int:
    """Load corpus → embed → build index on disk. Returns points indexed."""
    records = load_corpus(corpus_path)
    texts = [doc_text(r) for r in records]

    vectors = Embedder().embed_documents(texts) if embed_fn is None else embed_fn(texts)

    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    try:
        return build_index(records, vectors, client, collection)
    finally:
        client.close()


def main() -> None:
    n = run_index()
    print(f"indexed {n} points into qdrant collection '{COLLECTION}' at {QDRANT_PATH}/")


if __name__ == "__main__":
    main()
