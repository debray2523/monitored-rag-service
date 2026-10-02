# Design notes

## 1. Why hybrid search

Operations questions mix two kinds of wording. "How long can the API be down?" is a
paraphrase that dense vectors match well. "What does ERR-LLM-504 mean?" or "alpha for
SKU-7781" hinge on exact tokens that an embedding model blurs but BM25 matches exactly.
Weaviate runs both in a single `query.hybrid()` call:

* dense: cosine similarity between the query vector (all-MiniLM-L6-v2, computed by us and
  passed as `vector=`) and the stored chunk vectors (HNSW index);
* sparse: BM25F over the `text` property (`bm25_k1=1.2`, `bm25_b=0.75`, word tokenization);
* fusion: `HybridFusion.RELATIVE_SCORE` min-max normalises each result list to 0–1, then
  `score = alpha * dense + (1 - alpha) * bm25`. `alpha = 0.5` by default (`HYBRID_ALPHA`).

Relative-score fusion keeps the size of each score gap, unlike rank fusion, which only
keeps the order. `explain_score` is requested so each retrieved document carries the
per-component breakdown, and `scripts/evaluate.py` runs dense-only, BM25-only and hybrid
on the same golden set so the benefit is measured, not assumed.

## 2. Ingestion and indexing

* `pypdf` extracts text per page; chunks of ~700 characters with 120 overlap, split on
  paragraph and sentence boundaries; chunk id = sha1(source, page, position)[:16], and
  Weaviate object uuid = `generate_uuid5(chunk_id)`, so re-ingesting updates instead of
  duplicating.
* Schema `DocChunk`: `text` (word tokenization, BM25-searchable), `chunk_id`/`source`
  (field tokenization), `page`, `injection_flag`, `injection_rules`. Vectors are
  self-provided (`Configure.Vectors.self_provided()`); Weaviate runs no vectorizer module.
* Embeddings run locally (sentence-transformers, CPU, normalised); no text leaves the machine
  for embedding.

## 3. Prompt-injection defences (layered)

1. **Input guardrail** (`rag/sanitizer.py`), before retrieval and before any LLM call:
   Unicode NFKC normalisation, zero-width and control characters removed, then regexes on
   three views of the text (normalised, de-leetspeaked, and letters-only "compact") so
   `1gn0re prev1ous instructions` and `I g n o r e ...` are caught. Rules:
   `instruction_override`, `system_prompt_override`, `new_instructions`, `role_hijack`,
   `jailbreak_keyword`, `system_prompt_exfiltration`, `chat_template_injection`,
   `data_exfiltration` (high: block) and `tool_or_code_execution` (medium: logged only).
   Empty or over-long (>1000 chars) input is also rejected. A blocked query returns HTTP 400
   and the trace shows only `rag_query` + `input_guardrail`: no retrieval, no LLM spend.
2. **Ingestion scan** (indirect injection): every chunk is scanned with the same rules;
   flagged chunks are stored with `injection_flag=true` for audit and excluded from every
   search by a Weaviate filter. The sample `vendor_brief_poisoned.pdf` hides a 4-pt grey
   instruction on page 1; that chunk is quarantined while page 2 stays searchable.
3. **Prompt structure**: retrieved passages are wrapped in `<passage source= page=>` tags
   and the system prompt tells the model that passage text is untrusted data, never
   instructions.

Limits, stated honestly: pattern rules catch known phrasings, not every paraphrase or
other language; the ingestion rules were tightened after the security policy's own sentence
("instructions intended to override the system's rules") was a false positive — the
override rule now needs an attacker-style qualifier ("previous", "all", "your"...). A
benign look-alike ("ignore previous quarter anomalies") is allowed. A classifier model
would be the next layer.

## 4. Tracing design (OpenInference on OpenTelemetry)

```
rag_query (CHAIN)
├── input_guardrail (GUARDRAIL)   guardrail.allowed / rules / reason
├── hybrid_retrieval (RETRIEVER)  retrieval.documents.N.document.{id,content,score,metadata}
├── llm.groq (LLM)                llm.token_count.*, llm.cost.*, messages; ERROR status on failure
├── llm.gemini (LLM)              only if Groq failed
└── extractive_fallback (CHAIN)   only if every provider failed
```

* Spans go over OTLP/HTTP to `http://<phoenix>:6006/v1/traces`; app traces are in project
  `monitored-rag`, evaluation traces in `monitored-rag-evals`.
* `phoenix.otel.register()` (arize-phoenix-otel 0.17.1) crashes with OpenTelemetry 1.45
  (`'HTTPSpanExporter' object has no attribute '_headers'`), so `rag/tracing.py` builds the
  standard SDK `TracerProvider` itself; the resource attribute `openinference.project.name`
  is what files spans under a Phoenix project.
* Cost = tokens × the per-million prices in `.env`, set as `llm.cost.prompt/completion/total`,
  which Phoenix sums into trace and project cost.

## 5. Evaluation

`eval/golden.jsonl` holds 13 questions, each with the (PDF, page) that answers it. For
each question and each mode (dense, bm25, hybrid) `scripts/evaluate.py` opens an
`eval_retrieval.<mode>` RETRIEVER span, labels each retrieved document relevant if its
(source, page) is expected, and logs to Phoenix:

* document annotation `relevance` (CODE) on every retrieved document;
* span annotations `hit_rate`, `mrr`, `precision@4`, `ndcg@4`.

Phoenix computes its own built-in nDCG / precision / hit panel (the "Retrieval Metrics" box
on a retriever span) only from document annotations whose annotator kind is **LLM**.
`scripts/llm_judge.py` therefore asks the LLM to grade each hybrid result and logs
`llm_relevance` with kind LLM, which makes that panel appear. The code-computed metrics do
not depend on an LLM and are reproducible.

Precision@4 is capped at 0.25 here because each question has exactly one relevant page;
hit rate, MRR and nDCG are the informative numbers.

## 6. Outage handling

Providers are tried in `LLM_ORDER` (default Groq, then Gemini on Google Cloud through its
OpenAI-compatible endpoint; set `LLM_ORDER=gemini,groq` to make GCP primary), each with a timeout and one SDK retry. Any exception or an empty completion
moves to the next provider; the failed attempt's span is marked ERROR with the exception
recorded. If all fail — or no key is configured — the service still answers with an
extractive summary of the top passages and cites them, so a model outage degrades the
answer instead of failing the request.

If Weaviate itself is unreachable, the `hybrid_retrieval` span records the exception with
ERROR status and the API returns HTTP 503 `{"status": "unavailable"}`; `/health` reports
`degraded`. The failure is visible in Phoenix instead of surfacing as an unhandled 500.

## 7. Why Docker-first on this laptop

Windows Application Control blocked a native SciPy DLL in an earlier project, and Phoenix
imports SciPy/scikit-learn. Running Phoenix, the API and Weaviate as Linux containers
under Docker Desktop avoids the policy entirely without weakening it. The image bakes the
MiniLM model at build time and runs with `HF_HUB_OFFLINE=1`.

## 8. Version pins that matter

* `graphql-core==3.2.13`: with 3.3.0, Phoenix 20.16 annotation summaries fail
  ("Int cannot represent non-integer value: <int64>") and span pages show "Something went
  wrong".
* torch is installed from the CPU-only index to avoid multi-GB CUDA wheels.
