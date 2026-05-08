"""
Split extracted text into 512-token chunks with 50-token overlap.

Uses tiktoken with the cl100k_base encoding (compatible with both
text-embedding-3-small and GPT-4o).
"""
import logging

import tiktoken

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512     # tokens
CHUNK_OVERLAP = 50  # tokens
ENCODING = "cl100k_base"


def chunk_text(text: str) -> list[str]:
    """
    Tokenise *text*, split into overlapping windows, and return the
    decoded string for each window.
    """
    enc = tiktoken.get_encoding(ENCODING)
    tokens = enc.encode(text)

    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    logger.debug("chunker produced %d chunks from %d tokens", len(chunks), len(tokens))
    return chunks
