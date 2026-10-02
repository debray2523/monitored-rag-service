"""Test doubles: an in-memory retriever with the same interface as WeaviateRetriever,
and helpers for mocked OpenAI-compatible LLM endpoints."""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import httpx

from rag.embeddings import HashingEmbedder
from rag.store import Hit


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class InMemoryRetriever:
    """Dense (cosine), BM25 (k1=1.2, b=0.75) and relative-score hybrid over chunks."""

    def __init__(self, chunks, flags=None):
        flags = flags or [[] for _ in chunks]
        self.chunks = [c for c, f in zip(chunks, flags) if not f]      # quarantined never searchable
        self.emb = HashingEmbedder()
        self.vecs = self.emb.embed([c.text for c in self.chunks])
        self.docs = [_tokens(c.text) for c in self.chunks]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        self.df = Counter(t for d in self.docs for t in set(d))

    def _hit(self, i, score, mode):
        c = self.chunks[i]
        return Hit(c.chunk_id, c.source, c.page, c.text, round(score, 4), mode)

    def _dense_scores(self, q):
        qv = self.emb.embed([q])[0]
        return [sum(a * b for a, b in zip(qv, v)) for v in self.vecs]

    def _bm25_scores(self, q, k1=1.2, b=0.75):
        n, out = len(self.docs), []
        for d in self.docs:
            tf, s = Counter(d), 0.0
            for t in set(_tokens(q)):
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / self.avgdl))
            out.append(s)
        return out

    def _top(self, scores, k, mode):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self._hit(i, scores[i], mode) for i in order if scores[i] > 0]

    def dense(self, q, k=4):
        return self._top(self._dense_scores(q), k, "dense")

    def bm25(self, q, k=4):
        return self._top(self._bm25_scores(q), k, "bm25")

    def hybrid(self, q, k=4, alpha=0.5):
        def norm(xs):
            lo, hi = min(xs), max(xs)
            return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]
        d, s = norm(self._dense_scores(q)), norm(self._bm25_scores(q))
        return self._top([alpha * a + (1 - alpha) * b for a, b in zip(d, s)], k, "hybrid")

    def search(self, mode, q, k=4):
        return getattr(self, mode)(q, k)


def chat_completion(text="Answer [cloud_outage_runbook.pdf p.1]", prompt_tokens=120, completion_tokens=30):
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


def mock_http(status=200, body=None, seen=None) -> httpx.Client:
    def handler(request: httpx.Request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json=body if body is not None else chat_completion())
    return httpx.Client(transport=httpx.MockTransport(handler))
