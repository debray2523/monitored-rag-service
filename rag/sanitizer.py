"""Input-sanitization guardrail (OWASP LLM01: prompt injection).

`sanitize_query()` is the wrapper every user query passes through before it can
reach retrieval or an LLM. `scan_document()` applies the same detectors to text
extracted from PDFs, so indirect injections hidden in documents are flagged at
ingestion and never retrieved.

Detection works on three views of the text, so simple obfuscation does not help
an attacker:
  1. normalised   - Unicode NFKC, zero-width characters removed, lower-cased,
                    whitespace collapsed
  2. compact      - letters and digits only ("i g n o r e  a l l" -> "ignoreall")
  3. de-leeted    - common digit/symbol substitutions undone ("1gn0re" -> "ignore")
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from . import config

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# (name, severity, pattern on the normalised text)
PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # Needs an attacker-style qualifier (previous/all/your...), so descriptive text such as
    # "instructions intended to override the system's rules" in a security policy is not flagged.
    ("instruction_override", "high", re.compile(
        r"\b(ignore|disregard|forget|override|bypass|skip)\b[\w\s,'-]{0,20}?"
        r"\b(previous|prior|above|earlier|preceding|all|any|your|my|these|those)\b[\w\s,'-]{0,30}?"
        r"\b(instructions?|prompts?|rules?|directions?|guidelines?|context|messages?)\b")),
    ("system_prompt_override", "high", re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b\s+(the\s+)?(system|developer)\s+"
        r"(prompt|instructions?|message)\b")),
    ("new_instructions", "high", re.compile(
        r"\b(new|updated|real|actual|true)\s+(system\s+)?(instructions?|rules|task)\s*[:\-]")),
    ("role_hijack", "high", re.compile(
        r"\b(you are now|from now on,? you|act as|pretend (to be|you are)|roleplay as)\b"
        r".{0,40}\b(dan|jailbreak|unrestricted|unfiltered|no (rules|restrictions|limits)|developer mode|admin|root)\b")),
    ("jailbreak_keyword", "high", re.compile(
        r"\b(do anything now|developer mode (enabled|on)|jailbreak(ed)?|dan mode)\b")),
    ("system_prompt_exfiltration", "high", re.compile(
        r"\b(reveal|print|show|repeat|output|display|leak|tell me)\b[\w\s,'-]{0,30}?"
        r"\b(system|hidden|initial|original|secret)\s+(prompt|instructions?|message|rules)\b")),
    ("chat_template_injection", "high", re.compile(
        r"(<\|?(im_start|im_end|system|endoftext)\|?>|\[/?inst\]|<<\s*sys\s*>>|^\s*#{2,}\s*(system|instruction)s?\b)",
        re.MULTILINE)),
    ("data_exfiltration", "high", re.compile(
        r"\b(send|post|upload|forward|email|exfiltrate)\b.{0,60}\b(to|at)\b.{0,20}(https?://|www\.|@[\w-]+\.)")),
    ("tool_or_code_execution", "medium", re.compile(
        r"\b(execute|run|eval)\b.{0,20}\b(shell|command|code|script|os\.system|subprocess)\b")),
]

# Compact (letters+digits only) signatures catch spaced/punctuated obfuscation.
COMPACT_SIGNATURES: list[tuple[str, str]] = [
    ("instruction_override", "ignorepreviousinstructions"),
    ("instruction_override", "ignoreallpreviousinstructions"),
    ("instruction_override", "ignoreallinstructions"),
    ("instruction_override", "disregardpreviousinstructions"),
    ("instruction_override", "forgetyourinstructions"),
    ("system_prompt_exfiltration", "revealyoursystemprompt"),
    ("system_prompt_exfiltration", "showmeyoursystemprompt"),
    ("jailbreak_keyword", "doanythingnow"),
]


@dataclass
class Finding:
    rule: str
    severity: str
    excerpt: str


@dataclass
class SanitizationResult:
    allowed: bool
    cleaned: str
    findings: list[Finding] = field(default_factory=list)
    reason: str = ""

    @property
    def rules(self) -> list[str]:
        return sorted({f.rule for f in self.findings})


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", text).strip()


def _detect(text: str) -> list[Finding]:
    norm = normalise(text).lower()
    views = [norm, norm.translate(LEET)]
    findings: list[Finding] = []
    seen: set[str] = set()
    for view in views:
        for rule, severity, pattern in PATTERNS:
            m = pattern.search(view)
            if m and rule not in seen:
                seen.add(rule)
                findings.append(Finding(rule, severity, m.group(0)[:80]))
    for view in views:
        compact = re.sub(r"[^a-z0-9]", "", view)
        for rule, signature in COMPACT_SIGNATURES:
            if signature in compact and rule not in seen:
                seen.add(rule)
                findings.append(Finding(rule, "high", signature))
    return findings


def sanitize_query(query: str) -> SanitizationResult:
    """Guardrail wrapper for user queries. Block on any high-severity finding."""
    if not isinstance(query, str) or not query.strip():
        return SanitizationResult(False, "", reason="empty query")
    cleaned = normalise(query)
    if len(cleaned) > config.MAX_QUERY_CHARS:
        return SanitizationResult(False, cleaned[: config.MAX_QUERY_CHARS],
                                  [Finding("length_limit", "high", f"{len(cleaned)} chars")],
                                  f"query longer than {config.MAX_QUERY_CHARS} characters")
    findings = _detect(cleaned)
    blocked = [f for f in findings if f.severity == "high"]
    if blocked:
        return SanitizationResult(False, cleaned, findings,
                                  "possible prompt injection: " + ", ".join(sorted({f.rule for f in blocked})))
    return SanitizationResult(True, cleaned, findings)


def scan_document(text: str) -> list[Finding]:
    """Same detectors, applied to document text at ingestion (indirect injection)."""
    return _detect(text)
