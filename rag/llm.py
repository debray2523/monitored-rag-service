"""Multi-provider generation with fallback.

Order: Groq -> Google Gemini (GCP) -> extractive answer (no LLM).
A provider outage, timeout or rate limit moves to the next one; the extractive
fallback always succeeds, so the service degrades instead of failing. Every
attempt is its own LLM span with model, token counts, cost and error status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode

from . import config
from .tracing import set_llm_messages


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    price_in: float    # USD per 1M prompt tokens
    price_out: float   # USD per 1M completion tokens
    extra: dict
    http_client: object | None = None   # injectable for tests

    def client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key,
                      timeout=config.LLM_TIMEOUT_S, max_retries=1, http_client=self.http_client)


@dataclass
class Generation:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    attempts: tuple = ()


def default_providers() -> list[Provider]:
    """Providers with a key configured, in LLM_ORDER (default "groq,gemini")."""
    available = {}
    if config.GROQ_API_KEY:
        available["groq"] = Provider("groq", config.GROQ_BASE_URL, config.GROQ_API_KEY, config.GROQ_MODEL,
                                     *config.PRICE["groq"], extra={"reasoning_effort": "low"})
    if config.GEMINI_API_KEY:
        available["gemini"] = Provider("gemini", config.GEMINI_BASE_URL, config.GEMINI_API_KEY,
                                       config.GEMINI_MODEL, *config.PRICE["gemini"], extra={})
    return [available[name] for name in config.LLM_ORDER if name in available]


def extractive_answer(hits: Sequence) -> str:
    if not hits:
        return "No relevant information was found in the indexed documents."
    lines = ["All language-model providers are unavailable. The most relevant passages are:"]
    for h in hits[:3]:
        snippet = " ".join(h.text.split())[:300]
        lines.append(f"- [{h.source} p.{h.page}] {snippet}")
    return "\n".join(lines)


class LLMChain:
    def __init__(self, providers: list[Provider] | None = None):
        self.providers = default_providers() if providers is None else providers

    def generate(self, messages: list[dict], hits: Sequence, tracer) -> Generation:
        attempts = []
        for p in self.providers:
            with tracer.start_as_current_span(f"llm.{p.name}", openinference_span_kind="llm") as span:
                span.set_attribute(SpanAttributes.LLM_PROVIDER, p.name)
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, p.model)
                set_llm_messages(span, messages)
                try:
                    resp = p.client().chat.completions.create(
                        model=p.model, messages=messages, temperature=0.1,
                        max_tokens=config.LLM_MAX_TOKENS, **p.extra)
                    text = (resp.choices[0].message.content or "").strip()
                    if not text:
                        raise RuntimeError("empty completion")
                    usage = resp.usage
                    pt, ct = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
                    cost_in, cost_out = pt * p.price_in / 1e6, ct * p.price_out / 1e6
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, pt)
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, ct)
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, pt + ct)
                    span.set_attribute(SpanAttributes.LLM_COST_PROMPT, cost_in)
                    span.set_attribute(SpanAttributes.LLM_COST_COMPLETION, cost_out)
                    span.set_attribute(SpanAttributes.LLM_COST_TOTAL, cost_in + cost_out)
                    set_llm_messages(span, [], output=text)
                    span.set_output(text)
                    attempts.append((p.name, "ok"))
                    return Generation(text, p.name, p.model, pt, ct, cost_in + cost_out, tuple(attempts))
                except Exception as exc:  # outage, timeout, 429, auth, empty reply
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:200]))
                    attempts.append((p.name, type(exc).__name__))
        with tracer.start_as_current_span("extractive_fallback", openinference_span_kind="chain") as span:
            text = extractive_answer(hits)
            span.set_attribute("fallback.reason", "no LLM provider available" if not attempts else
                               "all providers failed: " + ", ".join(f"{n}={s}" for n, s in attempts))
            span.set_output(text)
            attempts.append(("extractive", "ok"))
            return Generation(text, "extractive", "none", attempts=tuple(attempts))
