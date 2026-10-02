"""Optional: LLM-as-judge relevance for the retrievals traced by evaluate.py.

Reads the `eval_retrieval.hybrid` spans from Phoenix, asks an LLM whether each
retrieved document is relevant to the question, and logs the verdicts as
document annotations named "llm_relevance" with annotator kind LLM. Phoenix then
also computes and displays its own nDCG / precision / hit metrics for those spans
(Phoenix computes these only from LLM-kind document annotations).

Uses the same provider chain as the API (Groq first). Paced for the free tier.

    python scripts/llm_judge.py [--limit 13] [--pause 2.5]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config  # noqa: E402
from rag.llm import default_providers  # noqa: E402

PROMPT = (
    "You are grading a search system. Question:\n{question}\n\nRetrieved document:\n{document}\n\n"
    "Does the document contain information that helps answer the question? "
    "Reply with exactly one word: relevant or irrelevant."
)


def judge(provider, question: str, document: str) -> str:
    resp = provider.client().chat.completions.create(
        model=provider.model, temperature=0, max_tokens=300, **provider.extra,
        messages=[{"role": "user", "content": PROMPT.format(question=question, document=document[:1500])}])
    word = (resp.choices[0].message.content or "").strip().lower()
    return "irrelevant" if "irrelevant" in word else ("relevant" if "relevant" in word else "unknown")


def judge_spans(spans_df: pd.DataFrame, provider, pause: float = 2.5, retries: int = 2) -> pd.DataFrame:
    rows = []
    for span_id, row in spans_df.iterrows():
        question = row["attributes.input.value"]
        for pos, doc in enumerate(row["attributes.retrieval.documents"] or []):
            for attempt in range(retries + 1):          # bounded retry for 429s
                try:
                    label = judge(provider, question, doc.get("document.content", ""))
                    break
                except Exception as exc:
                    if attempt == retries:
                        raise
                    print(f"  retry after {type(exc).__name__}; waiting 20s")
                    time.sleep(20)
            rows.append({"span_id": span_id, "document_position": pos, "label": label,
                         "score": 1.0 if label == "relevant" else 0.0,
                         "explanation": f"judged by {provider.name}:{provider.model}"})
            time.sleep(pause)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=13, help="number of hybrid spans to judge")
    ap.add_argument("--pause", type=float, default=2.5, help="seconds between judge calls")
    args = ap.parse_args()
    providers = default_providers()
    if not providers:
        print("No LLM provider configured (set GROQ_API_KEY in .env).")
        return 1
    from phoenix.client import Client

    client = Client(base_url=config.PHOENIX_BASE_URL)
    df = client.spans.get_spans_dataframe(project_identifier=config.PHOENIX_EVAL_PROJECT)
    df = df[df["name"] == "eval_retrieval.hybrid"].head(args.limit)
    if df.empty:
        print("No eval_retrieval.hybrid spans found - run scripts/evaluate.py first.")
        return 1
    labels = judge_spans(df, providers[0], pause=args.pause)
    client.spans.log_document_annotations_dataframe(
        dataframe=labels, annotation_name="llm_relevance", annotator_kind="LLM", sync=True)
    agree = labels["label"].value_counts().to_dict()
    print(f"Judged {len(labels)} documents on {len(df)} spans: {agree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
