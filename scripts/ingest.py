"""Ingest PDFs into Weaviate with local all-MiniLM-L6-v2 embeddings.

Each chunk is scanned for indirect prompt injection first; flagged chunks are
stored with injection_flag=true (for audit) and are excluded from every search.

    python scripts/ingest.py              # add / update
    python scripts/ingest.py --recreate   # drop and rebuild the collection
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config  # noqa: E402
from rag.pdf_loader import chunk_directory  # noqa: E402
from rag.sanitizer import scan_document  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="delete and recreate the collection")
    ap.add_argument("--pdf-dir", default=str(config.PDF_DIR))
    args = ap.parse_args()

    from rag.embeddings import get_embedder
    from rag.store import connect, count_objects, ensure_collection, ingest, iter_flagged

    chunks = chunk_directory(Path(args.pdf_dir))
    if not chunks:
        print(f"No PDFs with text found in {args.pdf_dir}")
        return 1
    print(f"{len(chunks)} chunks from {len({c.source for c in chunks})} PDFs")

    flags = [[f.rule for f in scan_document(c.text)] for c in chunks]
    for c, rules in zip(chunks, flags):
        if rules:
            print(f"  QUARANTINED {c.source} p.{c.page}: {', '.join(rules)}")

    t0 = time.time()
    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])
    print(f"Embedded with {config.EMBED_MODEL} ({len(vectors[0])} dims) in {time.time() - t0:.1f}s")

    client = connect()
    try:
        collection = ensure_collection(client, recreate=args.recreate)
        ingest(collection, chunks, vectors, flags)
        counts = count_objects(collection)
        print(f"Weaviate collection '{config.COLLECTION}': {counts['total']} objects, "
              f"{counts['flagged']} quarantined")
        for p in iter_flagged(collection):
            print(f"  flagged in store: {p['source']} p.{p['page']} {p['injection_rules']}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
