"""All settings in one place, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


# --- Weaviate ---------------------------------------------------------------
# Local Docker (default) or Weaviate Cloud (set WEAVIATE_URL + WEAVIATE_API_KEY).
WEAVIATE_HOST = _env("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(_env("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(_env("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_URL = _env("WEAVIATE_URL", "")          # e.g. https://xyz.weaviate.cloud
WEAVIATE_API_KEY = _env("WEAVIATE_API_KEY", "")
COLLECTION = _env("WEAVIATE_COLLECTION", "DocChunk")

# --- Embeddings (local Hugging Face model) ------------------------------------
EMBED_MODEL = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIMS = 384

# --- Chunking and retrieval ---------------------------------------------------
PDF_DIR = Path(_env("PDF_DIR", "data/pdfs"))
CHUNK_CHARS = int(_env("CHUNK_CHARS", "700"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "120"))
TOP_K = int(_env("TOP_K", "4"))
# 0 = pure BM25 keyword search, 1 = pure vector search, 0.5 = equal blend.
HYBRID_ALPHA = float(_env("HYBRID_ALPHA", "0.5"))

# --- Input guardrail ------------------------------------------------------------
MAX_QUERY_CHARS = int(_env("MAX_QUERY_CHARS", "1000"))

# --- LLM providers, tried in order (multi-provider fallback) -------------------
# Order in which providers are tried; names without a key are skipped.
LLM_ORDER = [n.strip().lower() for n in _env("LLM_ORDER", "groq,gemini").split(",") if n.strip()]
LLM_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "30"))
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "700"))
GROQ_API_KEY = _env("GROQ_API_KEY", "")
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_API_KEY = _env("GEMINI_API_KEY", "")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# USD per 1M tokens, used to put a cost on every LLM span. Check current prices
# on each provider's pricing page and set them in .env.
PRICE = {
    "groq": (float(_env("GROQ_PRICE_IN", "0.075")), float(_env("GROQ_PRICE_OUT", "0.30"))),
    "gemini": (float(_env("GEMINI_PRICE_IN", "0.30")), float(_env("GEMINI_PRICE_OUT", "2.50"))),
}

# --- Phoenix ------------------------------------------------------------------
PHOENIX_PROJECT = _env("PHOENIX_PROJECT_NAME", "monitored-rag")
PHOENIX_EVAL_PROJECT = _env("PHOENIX_EVAL_PROJECT_NAME", "monitored-rag-evals")
# OTLP/HTTP endpoint of the Phoenix server that px.launch_app() starts.
PHOENIX_COLLECTOR_ENDPOINT = _env("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_BASE_URL = _env("PHOENIX_BASE_URL", "http://localhost:6006")
