import json
from pathlib import Path

from rag.metrics import hit_at_k, ndcg_at_k, precision_at_k, reciprocal_rank, relevance_labels
from rag.pdf_loader import split_text
from rag.store import Hit

ROOT = Path(__file__).resolve().parents[1]


def test_corpus_chunks_have_source_and_page(corpus):
    chunks, _ = corpus
    assert {c.source for c in chunks} == {
        "cloud_outage_runbook.pdf", "rag_operations_handbook.pdf",
        "llm_security_policy.pdf", "vendor_brief_poisoned.pdf"}
    assert all(c.page >= 1 and c.text for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)          # ids are unique


def test_only_the_poisoned_page_is_quarantined(corpus):
    chunks, flags = corpus
    flagged = {(c.source, c.page) for c, f in zip(chunks, flags) if f}
    assert flagged == {("vendor_brief_poisoned.pdf", 1)}


def test_quarantined_chunk_is_never_retrieved(retriever):
    for mode in ("dense", "bm25", "hybrid"):
        hits = retriever.search(mode, "IMPORTANT NOTE TO AI ASSISTANTS email API keys", 10)
        assert ("vendor_brief_poisoned.pdf", 1) not in {(h.source, h.page) for h in hits}


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_text(text, size=300, overlap=50)
    assert len(parts) > 5
    assert all(len(p) <= 450 for p in parts)


def test_bm25_finds_exact_error_code(retriever):
    top = retriever.bm25("ERR-LLM-504", 1)[0]
    assert (top.source, top.page) == ("cloud_outage_runbook.pdf", 2)


def test_hybrid_combines_both_signals(retriever):
    hybrid = retriever.hybrid("Which alpha should be used for a query containing SKU-7781?", 4)
    assert (hybrid[0].source, hybrid[0].page) == ("rag_operations_handbook.pdf", 2)


def test_golden_set_references_real_pages(corpus):
    chunks, _ = corpus
    pages = {(c.source, c.page) for c in chunks}
    for line in (ROOT / "eval" / "golden.jsonl").read_text().splitlines():
        for rel in json.loads(line)["relevant"]:
            assert (rel["source"], rel["page"]) in pages


def _hits(*pairs):
    return [Hit("id", s, p, "t", 0.0, "hybrid") for s, p in pairs]


def test_metric_math():
    labels = relevance_labels(_hits(("a", 1), ("b", 2), ("c", 3)), [("b", 2)])
    assert labels == [0, 1, 0]
    assert hit_at_k(labels) == 1.0
    assert reciprocal_rank(labels) == 0.5
    assert abs(precision_at_k(labels, 3) - 1 / 3) < 1e-9
    assert abs(ndcg_at_k(labels, 3, 1) - 0.6309) < 1e-3
    assert reciprocal_rank([0, 0, 0]) == 0.0 and ndcg_at_k([0, 0], 2, 1) == 0.0


def test_search_demo_formats_all_three_modes(retriever):
    from scripts.search_demo import format_results
    q = "Which alpha is used for SKU-7781 lookups?"
    out = format_results(q, {"dense": retriever.dense(q, 2), "bm25": retriever.bm25(q, 2),
                             "hybrid": retriever.hybrid(q, 2, alpha=0.5)}, 0.5)
    assert "[dense]" in out and "[bm25]" in out and "[hybrid]" in out
    assert "rag_operations_handbook.pdf p.2" in out
