"""Retrieval evaluation, logged to Phoenix.

For every golden question, run dense, BM25 and hybrid retrieval inside a traced
RETRIEVER span (project "monitored-rag-evals"), label each retrieved chunk as
relevant/irrelevant, and log to Phoenix:

* document annotation "relevance" (1/0) on each retrieved document, from which
  Phoenix also computes its own nDCG / precision / hit metrics per span
* span annotations "hit_rate", "mrr", "precision@k", "ndcg@k"

Results are also written to docs/evidence/eval_results.json and eval_summary.md.

    python scripts/evaluate.py            # uses Weaviate + all-MiniLM-L6-v2
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config  # noqa: E402
from rag.metrics import (hit_at_k, ndcg_at_k, precision_at_k, reciprocal_rank,  # noqa: E402
                         relevance_labels, summarise)
from rag.tracing import flush_all, get_tracer, set_documents, span_id_hex  # noqa: E402

MODES = ("dense", "bm25", "hybrid")
GOLDEN = Path("eval/golden.jsonl")
OUT = Path("docs/evidence")


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_eval(retriever, tracer, golden: list[dict], k: int = config.TOP_K):
    rows, doc_rows = [], []
    for item in golden:
        expected = [(r["source"], r["page"]) for r in item["relevant"]]
        for mode in MODES:
            with tracer.start_as_current_span(f"eval_retrieval.{mode}",
                                              openinference_span_kind="retriever") as span:
                span.set_input(item["question"])
                span.set_attribute("eval.question_id", item["id"])
                span.set_attribute("retrieval.mode", mode)
                hits = retriever.search(mode, item["question"], k)
                set_documents(span, hits)
                sid = span_id_hex(span)
            labels = relevance_labels(hits, expected)
            rows.append({
                "id": item["id"], "mode": mode, "span_id": sid, "question": item["question"],
                "hit_rate": hit_at_k(labels), "mrr": reciprocal_rank(labels),
                "precision": precision_at_k(labels, k), "ndcg": ndcg_at_k(labels, k, len(expected)),
                "retrieved": [f"{h.source} p.{h.page}" for h in hits], "labels": labels,
            })
            for pos, (h, rel) in enumerate(zip(hits, labels)):
                doc_rows.append({"span_id": sid, "document_position": pos, "score": float(rel),
                                 "label": "relevant" if rel else "irrelevant",
                                 "explanation": f"{h.source} p.{h.page} vs expected {expected}"})
    return rows, doc_rows


def log_to_phoenix(rows: list[dict], doc_rows: list[dict], k: int = config.TOP_K) -> None:
    from phoenix.client import Client

    client = Client(base_url=config.PHOENIX_BASE_URL)
    if doc_rows:
        client.spans.log_document_annotations_dataframe(
            dataframe=pd.DataFrame(doc_rows), annotation_name="relevance",
            annotator_kind="CODE", sync=True)
    metric_names = {"hit_rate": "hit_rate", "mrr": "mrr",
                    "precision": f"precision@{k}", "ndcg": f"ndcg@{k}"}
    for key, name in metric_names.items():
        df = pd.DataFrame([{"span_id": r["span_id"], "score": r[key],
                            "label": f"{r[key]:.2f}", "explanation": f"{r['mode']} retrieval"}
                           for r in rows])
        client.spans.log_span_annotations_dataframe(
            dataframe=df, annotation_name=name, annotator_kind="CODE", sync=True)


def write_evidence(rows: list[dict], summary: dict, k: int, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.json").write_text(json.dumps({"k": k, "summary": summary, "per_query": rows},
                                                      indent=2), encoding="utf-8")
    lines = [f"# Retrieval evaluation (k={k}, {len({r['id'] for r in rows})} golden questions)", "",
             "| Mode | hit@k | MRR | precision@k | nDCG@k |", "|---|---|---|---|---|"]
    for mode, m in summary.items():
        lines.append(f"| {mode} | {m['hit_rate']} | {m['mrr']} | {m['precision']} | {m['ndcg']} |")
    lines += ["", "Per-question results are in `eval_results.json`; the same scores are visible in "
              "Phoenix (project `monitored-rag-evals`, columns *relevance*, *hit_rate*, *mrr*, "
              "*precision@k*, *ndcg@k*)."]
    (out / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    from rag.embeddings import get_embedder
    from rag.store import WeaviateRetriever, connect, ensure_collection

    golden = load_golden()
    client = connect()
    try:
        retriever = WeaviateRetriever(ensure_collection(client), get_embedder())
        tracer = get_tracer(config.PHOENIX_EVAL_PROJECT)
        rows, doc_rows = run_eval(retriever, tracer, golden)
    finally:
        client.close()
    flush_all()
    time.sleep(3)          # let Phoenix ingest the spans before annotating them
    log_to_phoenix(rows, doc_rows)
    summary = summarise(rows)
    write_evidence(rows, summary, config.TOP_K)
    print(json.dumps(summary, indent=2))
    print(f"Logged {len(doc_rows)} document annotations and {len(rows) * 4} span annotations "
          f"to Phoenix project '{config.PHOENIX_EVAL_PROJECT}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
