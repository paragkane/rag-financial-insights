"""Paragraph-aware chunker tuned for SEC filings.

Strategy:
  1. Run text_cleaner.extract_sections() to bias chunks toward MD&A / risk /
     liquidity — the sections that carry the alpha signal.
  2. Split on paragraph breaks; greedily pack paragraphs into ~1200-char
     chunks with a 150-char overlap so context isn't sliced mid-sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.extraction.text_cleaner import extract_sections

DEFAULT_CHUNK_CHARS = 1200
DEFAULT_OVERLAP = 150


@dataclass
class Chunk:
    text: str
    section: str
    idx: int


def _split_paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def _pack(paragraphs: list[str], section: str, max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= max_chars:
            buf += "\n\n" + para
        else:
            chunks.append(Chunk(text=buf, section=section, idx=len(chunks)))
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = (tail + "\n\n" + para).strip()
    if buf:
        chunks.append(Chunk(text=buf, section=section, idx=len(chunks)))
    return chunks


def chunk_filing(
    filing_text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Return chunks tagged by source section."""
    sections = extract_sections(filing_text) or {}
    if not sections:
        sections = {"body": filing_text}

    all_chunks: list[Chunk] = []
    for section, body in sections.items():
        if not body or not body.strip():
            continue
        paras = _split_paragraphs(body)
        all_chunks.extend(
            _pack(paras, section=section, max_chars=max_chars, overlap=overlap)
        )
    return all_chunks
