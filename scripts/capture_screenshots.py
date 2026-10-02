"""Capture the Phoenix evidence screenshots into docs/screenshots/ (headless Chromium).

Run inside the api container after demo_queries.py, evaluate.py and llm_judge.py:

    docker compose exec api pip install playwright==1.63.0
    docker compose exec api playwright install --with-deps chromium
    docker compose exec api python scripts/capture_screenshots.py

Finds the right traces through the Phoenix client, opens them in the Phoenix UI and
saves seven PNGs whose names match the links in README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config  # noqa: E402

OUT = Path("docs/screenshots")
BASE = config.PHOENIX_BASE_URL.rstrip("/")


def pick_traces(client):
    """Return (app_project_id, eval_project_id, answered_trace, blocked_trace, eval_trace)."""
    ids = {p["name"]: p["id"] for p in client.projects.list()}
    app_pid, eval_pid = ids[config.PHOENIX_PROJECT], ids[config.PHOENIX_EVAL_PROJECT]

    df = client.spans.get_spans_dataframe(project_identifier=config.PHOENIX_PROJECT)
    by_trace = df.groupby("context.trace_id")["name"].apply(set)
    roots = df[df["name"] == "rag_query"].set_index("context.trace_id")["attributes.input.value"]

    def first(trace_ids, prefer):
        trace_ids = list(trace_ids)
        for t in trace_ids:
            if prefer.lower() in str(roots.get(t, "")).lower():
                return t
        return trace_ids[0] if trace_ids else None

    groq_ok = df[(df["name"] == "llm.groq") & (df["status_code"] != "ERROR")]["context.trace_id"]
    any_llm = df[df["name"].str.startswith("llm.")]["context.trace_id"]
    answered = first(groq_ok.unique(), "recovery time") or first(any_llm.unique(), "recovery time")
    blocked = first([t for t, names in by_trace.items() if names == {"rag_query", "input_guardrail"}],
                    "ignore all previous")

    edf = client.spans.get_spans_dataframe(project_identifier=config.PHOENIX_EVAL_PROJECT)
    hyb = edf[edf["name"] == "eval_retrieval.hybrid"]
    pref = hyb[hyb["attributes.input.value"].str.contains("recovery time", case=False, na=False)]
    eval_trace = (pref if len(pref) else hyb)["context.trace_id"].iloc[0]
    return app_pid, eval_pid, answered, blocked, eval_trace


def main() -> int:
    from phoenix.client import Client
    from playwright.sync_api import sync_playwright

    app_pid, eval_pid, answered, blocked, eval_trace = pick_traces(Client(base_url=BASE))
    if not (answered and blocked):
        print("Missing traces: run scripts/demo_queries.py first.")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    trace_url = lambda pid, t: f"{BASE}/projects/{pid}/traces/{t}"  # noqa: E731

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})

        def shot(name):
            page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / name))
            print("saved", OUT / name)

        def open_(url):
            page.goto(url)
            page.wait_for_load_state("networkidle")

        def select(span_name):
            page.get_by_text(span_name, exact=True).first.click()

        open_(f"{BASE}/projects")
        shot("phoenix_projects.png")

        open_(trace_url(app_pid, answered))
        shot("trace_detail.png")
        select("hybrid_retrieval")
        shot("retrieval_span.png")
        select("llm.groq")
        shot("llm_span.png")

        open_(trace_url(app_pid, blocked))
        select("input_guardrail")
        shot("guardrail_span.png")

        page.set_viewport_size({"width": 1600, "height": 1300})
        open_(f"{BASE}/projects/{eval_pid}/spans")
        shot("eval_spans.png")

        open_(trace_url(eval_pid, eval_trace))
        shot("eval_span_detail.png")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
