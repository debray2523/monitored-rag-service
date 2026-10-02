"""FastAPI service: POST /query, GET /health."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config
from .llm import LLMChain
from .pipeline import answer
from .tracing import flush_all, get_tracer

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .embeddings import get_embedder
    from .store import WeaviateRetriever, connect, ensure_collection

    client = connect()
    state["client"] = client
    state["retriever"] = WeaviateRetriever(ensure_collection(client), get_embedder())
    state["llm"] = LLMChain()
    state["tracer"] = get_tracer(config.PHOENIX_PROJECT)
    yield
    flush_all()
    client.close()


app = FastAPI(title="Monitored RAG Service", version="1.0", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(config.TOP_K, ge=1, le=10)


@app.get("/health")
def health():
    try:
        ready = state.get("client") is not None and state["client"].is_ready()
    except Exception:
        ready = False
    providers = [p.name for p in state["llm"].providers] if "llm" in state else []
    return {"status": "ok" if ready else "degraded", "weaviate_ready": ready,
            "llm_providers": providers + ["extractive"]}


@app.post("/query")
def query(q: Query):
    result = answer(q.question, retriever=state["retriever"], llm=state["llm"],
                    tracer=state["tracer"], k=q.top_k)
    code = {"blocked": 400, "unavailable": 503}.get(result.status, 200)
    return JSONResponse(asdict(result), status_code=code)
