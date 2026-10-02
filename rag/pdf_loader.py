"""PDF text extraction and chunking."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from . import config


@dataclass
class Chunk:
    chunk_id: str      # stable id: hash of source + page + position
    source: str        # file name, e.g. incident_runbook.pdf
    page: int          # 1-based page number
    position: int      # chunk index within the page
    text: str


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append((i, text))
    return pages


def split_text(text: str, size: int = config.CHUNK_CHARS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph/sentence boundaries into ~size-char chunks with overlap."""
    if len(text) <= size:
        return [text]
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + piece).strip()
        else:
            current = (current + " " + piece).strip()
        while len(current) > size * 1.5:  # a single very long sentence
            chunks.append(current[:size])
            current = current[size - overlap:]
    if current:
        chunks.append(current)
    return chunks


def chunk_pdf(pdf_path: Path) -> list[Chunk]:
    out = []
    for page, text in extract_pages(pdf_path):
        for pos, piece in enumerate(split_text(text)):
            digest = hashlib.sha1(f"{pdf_path.name}|{page}|{pos}".encode()).hexdigest()[:16]
            out.append(Chunk(digest, pdf_path.name, page, pos, piece))
    return out


def chunk_directory(pdf_dir: Path = config.PDF_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        chunks.extend(chunk_pdf(pdf))
    return chunks
