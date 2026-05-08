"""
Answer generation — GPT-4o with retrieved context.

Builds a grounded prompt from the top-k retrieved chunks, calls GPT-4o,
and returns the answer text.
"""
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"
MAX_TOKENS = 1024

_SYSTEM_PROMPT = """You are a precise document question-answering assistant.
Answer the user's question using ONLY the context excerpts provided below.
If the answer cannot be found in the context, say "I don't have enough information in the provided document to answer that question."
Be concise and cite facts directly from the context."""

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _build_user_message(question: str, chunks: list[dict[str, Any]]) -> str:
    context_parts = [
        f"[Excerpt {i + 1} — chunk index {c['chunk_index']}]\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    ]
    context_block = "\n\n".join(context_parts)
    return f"Context excerpts from the document:\n\n{context_block}\n\n---\n\nQuestion: {question}"


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
async def generate_answer(question: str, chunks: list[dict[str, Any]]) -> str:
    """Call GPT-4o with a grounded prompt and return the answer string."""
    client = _get_client()
    user_message = _build_user_message(question, chunks)

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )

    answer = response.choices[0].message.content or ""
    logger.info(
        "Generated answer using %d context chunks (tokens: prompt=%s completion=%s)",
        len(chunks),
        response.usage.prompt_tokens if response.usage else "?",
        response.usage.completion_tokens if response.usage else "?",
    )
    return answer.strip()
