"""Phoenix / OpenTelemetry setup and small span helpers (OpenInference conventions)."""
from __future__ import annotations

import json
from typing import Sequence

from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import DocumentAttributes, SpanAttributes

from . import config

_providers: dict[str, object] = {}


def get_tracer(project: str = config.PHOENIX_PROJECT) -> OITracer:
    """Create (once per project) an OTLP/HTTP exporter to the Phoenix collector.

    Built with the standard OpenTelemetry SDK rather than phoenix.otel.register():
    register() in arize-phoenix-otel 0.17.1 crashes with OpenTelemetry 1.45
    ("'HTTPSpanExporter' object has no attribute '_headers'"). The resource attribute
    openinference.project.name is what Phoenix uses to file spans under a project.
    """
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if project not in _providers:
        provider = TracerProvider(resource=Resource.create({ResourceAttributes.PROJECT_NAME: project}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=config.PHOENIX_COLLECTOR_ENDPOINT)))
        _providers[project] = provider
    return tracer_from_provider(_providers[project])


def flush_all(timeout_ms: int = 10_000) -> None:
    for provider in _providers.values():
        provider.force_flush(timeout_ms)


def tracer_from_provider(provider) -> OITracer:
    """Wrap any OpenTelemetry TracerProvider (tests use an in-memory one)."""
    return OITracer(provider.get_tracer("monitored-rag"), config=TraceConfig())


def span_id_hex(span) -> str:
    return format(span.get_span_context().span_id, "016x")


def set_documents(span, hits: Sequence) -> None:
    """Attach retrieved documents so Phoenix shows them on the RETRIEVER span."""
    prefix = SpanAttributes.RETRIEVAL_DOCUMENTS
    for i, h in enumerate(hits):
        span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_ID}", h.chunk_id)
        span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}", h.text)
        span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_SCORE}", float(h.score))
        span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_METADATA}",
                           json.dumps({"source": h.source, "page": h.page, "mode": h.mode}))


def set_llm_messages(span, messages: list[dict], output: str | None = None) -> None:
    for i, m in enumerate(messages):
        span.set_attribute(f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.message.role", m["role"])
        span.set_attribute(f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.message.content", m["content"])
    if output is not None:
        span.set_attribute(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.role", "assistant")
        span.set_attribute(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.content", output)
