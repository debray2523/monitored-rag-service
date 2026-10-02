# Retrieval evaluation (k=4, 13 golden questions)

| Mode | hit@k | MRR | precision@k | nDCG@k |
|---|---|---|---|---|
| bm25 | 1.0 | 0.962 | 0.25 | 0.972 |
| dense | 1.0 | 1.0 | 0.25 | 1.0 |
| hybrid | 1.0 | 1.0 | 0.25 | 1.0 |

Per-question results are in `eval_results.json`; the same scores are visible in Phoenix (project `monitored-rag-evals`, columns *relevance*, *hit_rate*, *mrr*, *precision@k*, *ndcg@k*).
