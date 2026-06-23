"""Tests for retrieval indexing plumbing (no GPU, no network).

The heavy embedding model is never loaded here: ``build_index`` takes vectors
as input, so we feed fake ones and use qdrant's in-memory mode.
"""

import json
from pathlib import Path

import numpy as np
import pytest

# These tests exercise GPU-free logic, but importing the retrieval modules pulls
# the `ml` stack (torch via the embedder, qdrant-client). CI installs only the
# light deps, so skip the whole module there rather than fail collection.
pytest.importorskip("torch")
pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402

from sarcolit.retrieval import index, search  # noqa: E402
from sarcolit.retrieval.embedding import EMBED_DIM  # noqa: E402


class StubEmbedder:
    """Stand-in for the bge embedder: returns a fixed query vector, no GPU."""

    def __init__(self, query_vector: np.ndarray) -> None:
        self._qvec = query_vector

    def embed_query(self, query: str) -> np.ndarray:
        return self._qvec


RECORDS = [
    {"pmid": "111", "title": "Surface EMG of aging muscle", "abstract": "Alpha."},
    {"pmid": "222", "title": "Handgrip strength thresholds", "abstract": "Beta."},
    {"pmid": "333", "title": "Sarcopenia diagnosis review", "abstract": "Gamma."},
]


def test_doc_text_prepends_title() -> None:
    text = index.doc_text(RECORDS[0])
    assert text == "Surface EMG of aging muscle\nAlpha."


def test_load_corpus_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    loaded = index.load_corpus(path)
    assert [r["pmid"] for r in loaded] == ["111", "222", "333"]


def test_build_index_upserts_points_with_payload() -> None:
    client = QdrantClient(":memory:")
    vectors = np.random.default_rng(0).random((3, EMBED_DIM), dtype=np.float32)

    n = index.build_index(RECORDS, vectors, client, collection="test")

    assert n == 3
    stored = client.retrieve(collection_name="test", ids=[222], with_payload=True)
    assert stored[0].payload["title"] == "Handgrip strength thresholds"


def test_run_index_end_to_end_with_fake_embedder(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")

    def fake_embed(texts: list[str]) -> np.ndarray:
        return np.random.default_rng(1).random((len(texts), EMBED_DIM), dtype=np.float32)

    n = index.run_index(
        corpus_path=corpus,
        qdrant_path=tmp_path / "qdrant",
        collection="test",
        embed_fn=fake_embed,
    )
    assert n == 3


def _onehot(i: int) -> np.ndarray:
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def test_retriever_ranks_nearest_first_and_maps_payload() -> None:
    # Give each record an orthogonal one-hot vector; query matches record #1.
    client = QdrantClient(":memory:")
    vectors = np.vstack([_onehot(i) for i in range(len(RECORDS))])
    index.build_index(RECORDS, vectors, client, collection="test")

    retriever = search.Retriever(
        collection="test", embedder=StubEmbedder(_onehot(1)), client=client
    )
    hits = retriever.search("anything", k=2)

    assert hits[0].pmid == "222"  # the record whose vector matches the query
    assert hits[0].title == "Handgrip strength thresholds"
    assert hits[0].score >= hits[1].score  # sorted by descending similarity
    assert isinstance(hits[0], search.SearchHit)
