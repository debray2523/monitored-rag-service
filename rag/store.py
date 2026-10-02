"""Weaviate vector store: collection schema, ingestion, and three retrieval modes.

* dense  - near_vector on the all-MiniLM-L6-v2 embedding (semantic similarity)
* bm25   - sparse keyword search on the `text` property (exact terms, codes, names)
* hybrid - Weaviate runs BOTH searches and fuses the scores:
           score = alpha * dense_score + (1 - alpha) * bm25_score (relative-score fusion)

Every query excludes chunks that the ingestion scan flagged as containing an
indirect prompt injection (`injection_flag == True`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.init import AdditionalConfig, Auth, Timeout
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery
from weaviate.util import generate_uuid5

from . import config
from .pdf_loader import Chunk

RETURN_PROPS = ["chunk_id", "source", "page", "text"]


@dataclass
class Hit:
    chunk_id: str
    source: str
    page: int
    text: str
    score: float
    mode: str
    explain: str = ""
    extra: dict = field(default_factory=dict)


def connect() -> weaviate.WeaviateClient:
    """Weaviate Cloud if WEAVIATE_URL is set, otherwise the local Docker container."""
    extra = AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120))
    if config.WEAVIATE_URL:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=config.WEAVIATE_URL,
            auth_credentials=Auth.api_key(config.WEAVIATE_API_KEY),
            additional_config=extra,
        )
    return weaviate.connect_to_local(
        host=config.WEAVIATE_HOST, port=config.WEAVIATE_HTTP_PORT,
        grpc_port=config.WEAVIATE_GRPC_PORT, additional_config=extra,
    )


def ensure_collection(client: weaviate.WeaviateClient, recreate: bool = False):
    name = config.COLLECTION
    if recreate and client.collections.exists(name):
        client.collections.delete(name)
    if not client.collections.exists(name):
        client.collections.create(
            name=name,
            description="PDF chunks for the monitored RAG service",
            # We supply our own vectors (local all-MiniLM-L6-v2), so no server-side vectorizer.
            vector_config=Configure.Vectors.self_provided(),
            # BM25 parameters (these are Weaviate's defaults, made explicit).
            inverted_index_config=Configure.inverted_index(bm25_k1=1.2, bm25_b=0.75),
            properties=[
                Property(name="text", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
                Property(name="chunk_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="source", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="page", data_type=DataType.INT),
                Property(name="injection_flag", data_type=DataType.BOOL),
                Property(name="injection_rules", data_type=DataType.TEXT_ARRAY,
                         tokenization=Tokenization.FIELD),
            ],
        )
    return client.collections.get(name)


def ingest(collection, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]],
           flags: Sequence[list[str]]) -> int:
    """Upsert chunks with their vectors. `flags[i]` lists injection rules found in chunk i."""
    with collection.batch.fixed_size(batch_size=64) as batch:
        for chunk, vector, rules in zip(chunks, vectors, flags):
            batch.add_object(
                properties={
                    "text": chunk.text, "chunk_id": chunk.chunk_id, "source": chunk.source,
                    "page": chunk.page, "injection_flag": bool(rules), "injection_rules": list(rules),
                },
                vector=list(vector),
                uuid=generate_uuid5(chunk.chunk_id),  # re-ingesting overwrites, never duplicates
            )
    failed = collection.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} objects failed to import, first: {failed[0].message}")
    return len(chunks)


def _safe_only():
    return Filter.by_property("injection_flag").equal(False)


def _to_hits(result, mode: str) -> list[Hit]:
    hits = []
    for obj in result.objects:
        p = obj.properties
        md = obj.metadata
        if mode == "dense":
            score = 1.0 - float(md.distance or 0.0)   # cosine distance -> similarity
        else:
            score = float(md.score or 0.0)
        hits.append(Hit(chunk_id=p["chunk_id"], source=p["source"], page=int(p["page"]),
                        text=p["text"], score=round(score, 4), mode=mode,
                        explain=(getattr(md, "explain_score", None) or "")))
    return hits


class WeaviateRetriever:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder

    def dense(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        vector = self.embedder.embed([query])[0]
        res = self.collection.query.near_vector(
            near_vector=vector, limit=k, filters=_safe_only(),
            return_metadata=MetadataQuery(distance=True), return_properties=RETURN_PROPS)
        return _to_hits(res, "dense")

    def bm25(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        res = self.collection.query.bm25(
            query=query, query_properties=["text"], limit=k, filters=_safe_only(),
            return_metadata=MetadataQuery(score=True), return_properties=RETURN_PROPS)
        return _to_hits(res, "bm25")

    def hybrid(self, query: str, k: int = config.TOP_K, alpha: float = config.HYBRID_ALPHA) -> list[Hit]:
        vector = self.embedder.embed([query])[0]
        res = self.collection.query.hybrid(
            query=query, vector=vector, alpha=alpha, query_properties=["text"],
            fusion_type=HybridFusion.RELATIVE_SCORE, limit=k, filters=_safe_only(),
            return_metadata=MetadataQuery(score=True, explain_score=True),
            return_properties=RETURN_PROPS)
        return _to_hits(res, "hybrid")

    def search(self, mode: str, query: str, k: int = config.TOP_K) -> list[Hit]:
        return {"dense": self.dense, "bm25": self.bm25, "hybrid": self.hybrid}[mode](query, k)


def count_objects(collection) -> dict:
    total = collection.aggregate.over_all(total_count=True).total_count
    flagged = collection.aggregate.over_all(
        total_count=True, filters=Filter.by_property("injection_flag").equal(True)).total_count
    return {"total": total, "flagged": flagged}


def iter_flagged(collection) -> Iterable[dict]:
    res = collection.query.fetch_objects(
        filters=Filter.by_property("injection_flag").equal(True), limit=100,
        return_properties=["source", "page", "injection_rules"])
    for obj in res.objects:
        yield obj.properties
