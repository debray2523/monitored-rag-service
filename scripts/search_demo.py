"""Show dense, BM25 and hybrid results side by side for one query (hybrid-search evidence).

    python scripts/search_demo.py "Which alpha is used for SKU-7781 lookups?"
    python scripts/search_demo.py "How long can the API be down?" --alpha 0.75 --k 3

Output is printed and also appended to docs/evidence/search_demo.txt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config  # noqa: E402


def format_results(query: str, results: dict, alpha: float) -> str:
    lines = [f"QUERY: {query}   (hybrid alpha={alpha}: 0 = pure BM25, 1 = pure vector)"]
    for mode, hits in results.items():
        lines.append(f"  [{mode}]")
        for rank, h in enumerate(hits, 1):
            lines.append(f"    {rank}. {h.score:7.4f}  {h.source} p.{h.page}  {' '.join(h.text.split())[:70]}")
            if mode == "hybrid" and h.explain:
                lines.append(f"         explain: {' '.join(h.explain.split())[:160]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--alpha", type=float, default=config.HYBRID_ALPHA)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    from rag.embeddings import get_embedder
    from rag.store import WeaviateRetriever, connect, ensure_collection

    client = connect()
    try:
        r = WeaviateRetriever(ensure_collection(client), get_embedder())
        results = {"dense": r.dense(args.query, args.k), "bm25": r.bm25(args.query, args.k),
                   "hybrid": r.hybrid(args.query, args.k, alpha=args.alpha)}
    finally:
        client.close()
    text = format_results(args.query, results, args.alpha)
    print(text)
    out = Path("docs/evidence/search_demo.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
