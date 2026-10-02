"""Retrieval-quality metrics computed against a labelled golden set.

A retrieved chunk is *relevant* if its (source file, page) is one of the pages the
golden set says contains the answer. From those 0/1 labels per rank we compute:

* hit@k        1 if any of the top-k results is relevant, else 0
* MRR          1 / rank of the first relevant result (0 if none)
* precision@k  relevant results / k
* nDCG@k       rank-discounted gain, normalised by the best possible ordering
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def relevance_labels(hits: Sequence, expected: Iterable[tuple[str, int]]) -> list[int]:
    wanted = {(s, int(p)) for s, p in expected}
    return [1 if (h.source, int(h.page)) in wanted else 0 for h in hits]


def hit_at_k(labels: Sequence[int]) -> float:
    return 1.0 if any(labels) else 0.0


def reciprocal_rank(labels: Sequence[int]) -> float:
    for rank, rel in enumerate(labels, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def precision_at_k(labels: Sequence[int], k: int) -> float:
    return sum(labels[:k]) / k if k else 0.0


def ndcg_at_k(labels: Sequence[int], k: int, n_relevant: int) -> float:
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(labels[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, n_relevant)))
    return dcg / ideal if ideal else 0.0


def summarise(rows: list[dict]) -> dict:
    """Average per-query metrics by retrieval mode."""
    out: dict[str, dict] = {}
    for mode in sorted({r["mode"] for r in rows}):
        sel = [r for r in rows if r["mode"] == mode]
        out[mode] = {m: round(sum(r[m] for r in sel) / len(sel), 3)
                     for m in ("hit_rate", "mrr", "precision", "ndcg")}
        out[mode]["queries"] = len(sel)
    return out
