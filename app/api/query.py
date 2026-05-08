"""
POST /query  — ask a natural-language question about an ingested document.
GET  /documents — list all ingested documents.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.metadata import get_document, get_session, list_documents
from app.models.schemas import (
    DocumentListResponse,
    DocumentMeta,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.services.generator import generate_answer
from app.services.retriever import retrieve_chunks

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_document(
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> QueryResponse:
    # ── Validate document exists ──────────────────────────────────────────────
    doc = await get_document(session, body.document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{body.document_id}' not found.",
        )

    # ── Retrieve relevant chunks ──────────────────────────────────────────────
    try:
        chunks = await retrieve_chunks(session, body.document_id, body.question)
    except Exception as exc:
        logger.error("Retrieval failed for document %s: %s", body.document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to retrieve document context. Please try again.",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant content found for the given question.",
        )

    # ── Generate answer ───────────────────────────────────────────────────────
    try:
        answer = await generate_answer(body.question, chunks)
    except Exception as exc:
        logger.error("Generation failed for document %s: %s", body.document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate an answer. Please try again.",
        ) from exc

    sources = [
        SourceChunk(chunk_text=c["chunk_text"], chunk_index=c["chunk_index"])
        for c in chunks
    ]

    logger.info(
        "Query answered: document_id=%s question_len=%d sources=%d",
        body.document_id,
        len(body.question),
        len(sources),
    )
    return QueryResponse(answer=answer, sources=sources)


@router.get("/documents", response_model=DocumentListResponse)
async def list_all_documents(
    session: AsyncSession = Depends(get_session),
) -> DocumentListResponse:
    docs = await list_documents(session)
    return DocumentListResponse(
        documents=[
            DocumentMeta(
                document_id=d.id,
                filename=d.filename,
                chunk_count=d.chunk_count,
                created_at=d.created_at,
            )
            for d in docs
        ]
    )
