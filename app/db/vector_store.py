"""
pgvector insert and cosine-similarity search operations.

Table: document_chunks
    id            UUID primary key
    document_id   FK → documents.id
    chunk_index   position of chunk within the document
    chunk_text    raw text of the chunk
    embedding     vector(1536)  — text-embedding-3-small dimensions
    page_number   source page in the original PDF (NULL for TXT files)
"""
import logging
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from .metadata import Base

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    page_number = Column(Integer, nullable=True)  # NULL for TXT files


async def insert_chunks(
    session: AsyncSession,
    document_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    page_numbers: list[int | None] | None = None,
) -> None:
    """Bulk-insert text chunks with their embeddings and optional page numbers."""
    pages = page_numbers if page_numbers is not None else [None] * len(chunks)
    rows = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=i,
            chunk_text=chunk,
            embedding=embedding,
            page_number=page,
        )
        for i, (chunk, embedding, page) in enumerate(zip(chunks, embeddings, pages))
    ]
    session.add_all(rows)
    await session.commit()
    logger.info("Inserted %d chunks for document %s", len(rows), document_id)


async def similarity_search(
    session: AsyncSession,
    document_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-k chunks most similar to the query embedding."""
    # Use pgvector cosine distance operator (<=>)
    stmt = text(
        """
        SELECT chunk_index, chunk_text, page_number,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM document_chunks
        WHERE document_id = :document_id
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
        """
    )
    result = await session.execute(
        stmt,
        {
            "embedding": str(query_embedding),
            "document_id": document_id,
            "top_k": top_k,
        },
    )
    rows = result.fetchall()
    logger.debug("similarity_search returned %d rows for document %s", len(rows), document_id)
    return [
        {
            "chunk_index": row.chunk_index,
            "chunk_text": row.chunk_text,
            "page_number": row.page_number,
            "similarity": float(row.similarity),
        }
        for row in rows
    ]
