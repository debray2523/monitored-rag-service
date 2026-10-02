# Evidence

Generated on the laptop by the scripts, then committed:

| File | Produced by |
|---|---|
| `ingest_log.txt` | `docker compose exec -T api python scripts/ingest.py --recreate > docs/evidence/ingest_log.txt` |
| `search_demo.txt` | `scripts/search_demo.py "<query>"` (dense vs BM25 vs hybrid, with Weaviate's score explanation) |
| `demo_queries.json` | `scripts/demo_queries.py` (normal questions answered, injections blocked with HTTP 400) |
| `eval_results.json`, `eval_summary.md` | `scripts/evaluate.py` (dense vs BM25 vs hybrid: hit rate, MRR, precision@4, nDCG@4) |
| `pytest_log.txt` | `docker compose exec -T api python -m pytest -v > docs/evidence/pytest_log.txt` |

Phoenix screenshots are in `../screenshots/`.
