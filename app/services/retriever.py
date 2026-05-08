"""
Retrieve the most relevant chunks for a question using cosine similarity.
"""
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.vector_store import similarity_search
from app.services.embedder import embed_query

logger = logging.getLogger(__name__)

TOP_K = 5


async def retrieve_chunks(
    session: AsyncSession,
    document_id: str,
    question: str,
) -> list[dict[str, Any]]:
    """
    Embed *question*, search pgvector for the top-k most similar chunks,
    and return them sorted by descending similarity.
    """
    query_embedding = await embed_query(question)
    chunks = await similarity_search(
        session=session,
        document_id=document_id,
        query_embedding=query_embedding,
        top_k=TOP_K,
    )
    logger.info(
        "Retrieved %d chunks for document %s (question length: %d chars)",
        len(chunks),
        document_id,
        len(question),
    )
    return chunks
