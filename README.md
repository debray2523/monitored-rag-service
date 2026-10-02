# Monitored RAG Service — Cloud Outage & Injection Case Study

A retrieval-augmented question-answering service over operations PDFs, built to stay
observable and safe when things go wrong:

* **Weaviate** vector store, PDFs embedded locally with Hugging Face **all-MiniLM-L6-v2** (384 dims)
* **Hybrid search**: dense vectors + BM25 keyword scores fused in one Weaviate query (`alpha = 0.5`, relative-score fusion)
* **Input sanitization guardrail** that blocks prompt-injection patterns before retrieval or any LLM call, plus ingestion-time scanning that quarantines poisoned PDF chunks
* **Arize Phoenix** (`px.launch_app()`) at <http://localhost:6006> with OpenInference spans for the guardrail, retrieval, each LLM attempt (tokens and cost) and retrieval-quality evaluations
* **Outage tolerance**: Groq → Gemini (GCP) → extractive fallback, every attempt traced

## Architecture

```
            ┌────────────────────────── docker compose ──────────────────────────┐
 client ──► │ api :8000 (FastAPI)                                                 │
            │   1 input_guardrail ──blocked──► 400                                │
            │   2 hybrid_retrieval ──► weaviate :8080/:50051 (BM25 + vectors)     │
            │   3 llm.groq ─fail─► llm.gemini ─fail─► extractive_fallback         │
            │   every step = an OpenInference span ──OTLP──► phoenix :6006        │
            └─────────────────────────────────────────────────────────────────────┘
 scripts/ingest.py   PDFs ► chunks ► injection scan ► MiniLM vectors ► Weaviate
 scripts/evaluate.py golden set ► dense / bm25 / hybrid ► Phoenix annotations
```

## Quick start (Docker Desktop)

```bat
copy .env.example .env              & rem add GROQ_API_KEY (optional GEMINI_API_KEY)
docker compose build
docker compose up -d
docker compose exec api python scripts/make_sample_pdfs.py
docker compose exec api python scripts/ingest.py --recreate
docker compose exec api python scripts/search_demo.py "Which alpha is used for SKU-7781 lookups?"
docker compose exec api python scripts/demo_queries.py
docker compose exec api python scripts/evaluate.py
docker compose exec api python scripts/llm_judge.py          & rem optional, needs an LLM key
```

Open <http://localhost:6006> (Phoenix) and <http://localhost:8000/docs> (API).

```bat
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"What is the RTO for the RAG API?\"}"
```

Tests (no Docker, no network needed): `docker compose exec api python -m pytest -q` (45 tests).

## Rubric → where to look

| Criterion | Implementation | Evidence |
|---|---|---|
| Observability setup (15) | `scripts/launch_phoenix.py` (`px.launch_app()`), `phoenix` service in `docker-compose.yml`, `rag/tracing.py` | `docs/screenshots/phoenix_projects.png` |
| Vector indexing — Weaviate (15) | `rag/store.py` (`ensure_collection`, `ingest`), `rag/embeddings.py`, `rag/pdf_loader.py`, `scripts/ingest.py` | `docs/evidence/ingest_log.txt` |
| Hybrid search — dense + BM25 combined (20) | `WeaviateRetriever.hybrid()` in `rag/store.py`: `collection.query.hybrid(query, vector, alpha, fusion_type=RELATIVE_SCORE)` | `docs/screenshots/retrieval_span.png`, `docs/evidence/search_demo.txt`, dense vs BM25 vs hybrid in `docs/evidence/eval_summary.md` |
| Input sanitization guardrail (20) | `rag/sanitizer.py`, `input_guardrail` span in `rag/pipeline.py`, ingestion scan in `scripts/ingest.py` | `docs/evidence/demo_queries.json`, `docs/screenshots/guardrail_span.png`, `tests/test_sanitizer.py` |
| Tracing integration (15) | `rag/pipeline.py`, `rag/llm.py`: CHAIN / GUARDRAIL / RETRIEVER / LLM spans, token counts and cost | `docs/screenshots/trace_detail.png`, `docs/screenshots/llm_span.png` |
| Evaluation evidence (15) | `scripts/evaluate.py` (hit rate, MRR, precision@k, nDCG@k), `scripts/llm_judge.py` | `docs/screenshots/eval_spans.png`, `docs/screenshots/eval_span_detail.png`, `docs/evidence/eval_results.json` |

## Repository layout

```
rag/          config, sanitizer, pdf_loader, embeddings, store (Weaviate), tracing,
              llm (provider fallback), pipeline, api (FastAPI), metrics
scripts/      launch_phoenix, make_sample_pdfs, ingest, search_demo, demo_queries, evaluate, llm_judge,
              capture_screenshots
eval/         golden.jsonl — 13 questions with the page that answers each
data/pdfs/    4 sample PDFs (one carries a hidden prompt injection)
tests/        45 offline tests (in-memory retriever, mocked LLM HTTP, Weaviate signature checks)
docs/         evidence/ and screenshots/
```

Design decisions and known limits are in [DESIGN.md](DESIGN.md).
