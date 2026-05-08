"""
OpenAI embeddings — text-embedding-3-small.

Provides both single-text and batch embedding helpers with retry logic.
"""
import logging
import os

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

MODEL = "text-embedding-3-small"
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input text."""
    if not texts:
        return []
    client = _get_client()
    response = await client.embeddings.create(model=MODEL, input=texts)
    # Response items are ordered by index
    vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    logger.debug("Embedded %d texts with model %s", len(texts), MODEL)
    return vectors


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
async def embed_query(text: str) -> list[float]:
    """Return a single embedding for a query string."""
    client = _get_client()
    response = await client.embeddings.create(model=MODEL, input=[text])
    return response.data[0].embedding
