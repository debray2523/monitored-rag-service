"""Contract tests: every call we make to the Weaviate client must match the real
client's method signatures (weaviate-client pinned in requirements.txt).
`create_autospec` raises TypeError on any wrong argument name."""
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

from weaviate.collections.batch.collection import _BatchCollection
from weaviate.collections.collections.sync import _Collections
from weaviate.collections.queries.bm25.query.sync import _BM25Query
from weaviate.collections.queries.hybrid.query.sync import _HybridQuery
from weaviate.collections.queries.near_vector.query.sync import _NearVectorQuery

from rag import store
from rag.embeddings import HashingEmbedder
from rag.pdf_loader import Chunk


def _result(distance=None, score=None):
    obj = SimpleNamespace(
        properties={"chunk_id": "c1", "source": "a.pdf", "page": 2, "text": "hello"},
        metadata=SimpleNamespace(distance=distance, score=score, explain_score="(bm25)+(vector)"))
    return SimpleNamespace(objects=[obj])


def _collection():
    hybrid = create_autospec(_HybridQuery, instance=True)
    bm25 = create_autospec(_BM25Query, instance=True)
    near = create_autospec(_NearVectorQuery, instance=True)
    hybrid.hybrid.return_value = _result(score=0.9)
    bm25.bm25.return_value = _result(score=3.1)
    near.near_vector.return_value = _result(distance=0.2)
    query = SimpleNamespace(hybrid=hybrid.hybrid, bm25=bm25.bm25, near_vector=near.near_vector)
    return SimpleNamespace(query=query), hybrid, bm25, near


def test_retrieval_calls_match_client_signatures():
    col, hybrid, bm25, near = _collection()
    r = store.WeaviateRetriever(col, HashingEmbedder())
    h = r.hybrid("what is x", 3, alpha=0.5)
    b = r.bm25("ERR-1", 3)
    d = r.dense("what is x", 3)
    assert h[0].score == 0.9 and h[0].explain == "(bm25)+(vector)" and h[0].mode == "hybrid"
    assert b[0].score == 3.1 and d[0].score == 0.8          # dense score = 1 - distance
    kw = hybrid.hybrid.call_args.kwargs
    assert kw["alpha"] == 0.5 and len(kw["vector"]) == 384 and kw["query"] == "what is x"
    assert kw["fusion_type"].name == "RELATIVE_SCORE"
    assert kw["filters"] is not None                          # quarantine filter applied
    assert bm25.bm25.call_args.kwargs["query_properties"] == ["text"]
    assert near.near_vector.call_args.kwargs["filters"] is not None


def test_collection_schema_matches_client_signature():
    collections = create_autospec(_Collections, instance=True)
    collections.exists.return_value = False
    client = SimpleNamespace(collections=collections)
    store.ensure_collection(client)
    kw = collections.create.call_args.kwargs
    names = {p.name for p in kw["properties"]}
    assert names == {"text", "chunk_id", "source", "page", "injection_flag", "injection_rules"}


def test_ingest_calls_match_batch_signature():
    batch = create_autospec(_BatchCollection, instance=True)
    ctx = MagicMock()
    ctx.__enter__.return_value = batch
    col = SimpleNamespace(batch=SimpleNamespace(fixed_size=lambda batch_size: ctx, failed_objects=[]))
    chunk = Chunk("c1", "a.pdf", 1, 0, "text")
    assert store.ingest(col, [chunk], [[0.1] * 384], [["instruction_override"]]) == 1
    props = batch.add_object.call_args.kwargs["properties"]
    assert props["injection_flag"] is True and props["injection_rules"] == ["instruction_override"]
