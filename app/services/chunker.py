"""
Split extracted text into 512-token chunks with 50-token overlap.

Uses tiktoken with the cl100k_base encoding (compatible with both
text-embedding-3-small and GPT-4o).

Functions
---------
chunk_text(text)       — flat text → list[str]  (no page tracking)
chunk_pages(pages)     — [(page_num, text), ...] → list[{"chunk_text", "page_number"}]
                         Each chunk is tagged with the page it starts on.
"""
import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512     # tokens
CHUNK_OVERLAP = 50  # tokens
ENCODING = "cl100k_base"


def _sliding_window(tokens: list[int], enc: tiktoken.Encoding) -> list[str]:
    """Core sliding-window splitter shared by both public functions."""
    if not tokens:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def chunk_text(text: str) -> list[str]:
    """
    Tokenise *text*, split into overlapping windows, and return the
    decoded string for each window.  Page numbers are not tracked.
    """
    enc = tiktoken.get_encoding(ENCODING)
    tokens = enc.encode(text)
    chunks = _sliding_window(tokens, enc)
    logger.debug("chunk_text produced %d chunks from %d tokens", len(chunks), len(tokens))
    return chunks


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """
    Chunk a PDF document page by page so every chunk knows its source page.

    Parameters
    ----------
    pages : list of (1-based page_number, page_text) tuples

    Returns
    -------
    list of {"chunk_text": str, "page_number": int}
    """
    enc = tiktoken.get_encoding(ENCODING)
    result: list[dict[str, Any]] = []

    for page_num, page_text in pages:
        if not page_text.strip():
            continue
        tokens = enc.encode(page_text)
        page_chunks = _sliding_window(tokens, enc)
        for chunk in page_chunks:
            result.append({"chunk_text": chunk, "page_number": page_num})

    logger.debug("chunk_pages produced %d chunks from %d pages", len(result), len(pages))
    return result
