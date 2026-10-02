"""Local Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2, 384 dims).

The model runs on the CPU inside the container; no embedding API is called.
`HashingEmbedder` is a tiny deterministic stand-in used only by the unit tests.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol, Sequence

from . import config


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                     show_progress_bar=False)
        return [v.tolist() for v in vectors]


class HashingEmbedder:
    """Deterministic bag-of-words vectors for tests (no model download)."""

    model_name = "hashing-test-embedder"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * config.EMBED_DIMS
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % config.EMBED_DIMS] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return SentenceTransformerEmbedder()
