import sys
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.pdf_loader import chunk_directory  # noqa: E402
from rag.sanitizer import scan_document  # noqa: E402
from rag.tracing import tracer_from_provider  # noqa: E402
from tests.fakes import InMemoryRetriever  # noqa: E402


@pytest.fixture
def spans():
    """An OpenInference tracer that records spans in memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_from_provider(provider), exporter


@pytest.fixture(scope="session")
def corpus():
    chunks = chunk_directory(ROOT / "data" / "pdfs")
    flags = [[f.rule for f in scan_document(c.text)] for c in chunks]
    return chunks, flags


@pytest.fixture(scope="session")
def retriever(corpus):
    return InMemoryRetriever(*corpus)
