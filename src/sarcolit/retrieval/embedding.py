"""Embedding model wrapper for SarcoLit retrieval.

One place to load the embedding model and turn text into vectors, shared by
indexing (`index.py`) and querying (`search.py`) so both sides land in the
*same* vector space and stay comparable.

Model: ``BAAI/bge-base-en-v1.5`` — a 768-dim general-purpose retriever, used
off-the-shelf (pre-trained) as the v0.2 baseline. A domain-specific embedder
(e.g. PubMedBERT) is a deferred v0.3 comparison, not the baseline.

Run directly through ``transformers`` (not ``sentence-transformers``, which
segfaulted on this Anaconda + pip-torch setup). bge's recipe: take the
``[CLS]`` token's last hidden state and L2-normalise it. Normalising means
cosine similarity reduces to a dot product, matching the qdrant collection.

bge quirk honoured here: the *query* side wants a short instruction prefix,
while *documents* (the abstracts) are embedded with no prefix. Mixing this up
quietly degrades retrieval, so the two cases are separate methods.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (universal PyTorch convention)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768
MAX_TOKENS = 512
# bge-v1.5's recommended retrieval instruction, prepended to queries only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """bge embedder over ``transformers``: text -> L2-normalised CLS vectors."""

    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def _encode(self, texts: list[str], batch_size: int, progress: bool) -> np.ndarray:
        chunks: list[np.ndarray] = []
        rng = range(0, len(texts), batch_size)
        for start in tqdm(rng, desc="embed", unit="batch", disable=not progress):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]  # bge uses [CLS]
            cls = F.normalize(cls, p=2, dim=1)
            chunks.append(cls.cpu().numpy())
        return np.vstack(chunks).astype(np.float32)

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed passages (no prefix). Returns an ``(n, EMBED_DIM)`` array."""
        return self._encode(texts, batch_size=batch_size, progress=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query (with the bge instruction prefix)."""
        return self._encode([QUERY_PREFIX + query], batch_size=1, progress=False)[0]
