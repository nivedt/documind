"""
POST /ingest — upload a PDF or TXT document, chunk, embed, and store it.
"""
import io
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.metadata import create_document, get_session
from app.db.vector_store import insert_chunks
from app.models.schemas import IngestResponse
from app.services.chunker import chunk_pages, chunk_text
from app.services.embedder import embed_texts

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/plain; charset=utf-8",
}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def _extract_pages_from_pdf(data: bytes) -> list[tuple[int, str]]:
    """Return a list of (1-based page_number, page_text) tuples."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [
            (i + 1, reader.pages[i].extract_text() or "")
            for i in range(len(reader.pages))
        ]
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract text from PDF: {exc}",
        ) from exc


def _extract_text_from_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    # ── Validate ──────────────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in {"application/pdf", "text/plain"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Only PDF and TXT are accepted.",
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 20 MB limit.",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── Extract text (page-aware for PDFs) ───────────────────────────────────
    file_size = len(data)
    logger.info("Ingesting file: %s (%s, %d bytes)", file.filename, content_type, file_size)

    page_count: int | None = None
    chunk_dicts: list[dict]  # [{"chunk_text": str, "page_number": int | None}]

    if content_type == "application/pdf":
        pdf_pages = _extract_pages_from_pdf(data)
        page_count = len(pdf_pages)
        all_text = "".join(t for _, t in pdf_pages)
        if not all_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No extractable text found in the uploaded PDF.",
            )
        chunk_dicts = chunk_pages(pdf_pages)
        file_type = "pdf"
    else:
        raw_text = _extract_text_from_txt(data)
        if not raw_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No extractable text found in the uploaded file.",
            )
        chunk_dicts = [{"chunk_text": c, "page_number": None} for c in chunk_text(raw_text)]
        file_type = "txt"

    if not chunk_dicts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document produced zero chunks after processing.",
        )

    chunks = [d["chunk_text"] for d in chunk_dicts]
    page_numbers: list[int | None] = [d["page_number"] for d in chunk_dicts]

    # ── Embed ─────────────────────────────────────────────────────────────────
    try:
        embeddings = await embed_texts(chunks)
    except Exception as exc:
        logger.error("Embedding failed for %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate embeddings. Please try again.",
        ) from exc

    # ── Store ─────────────────────────────────────────────────────────────────
    document_id = str(uuid.uuid4())
    await insert_chunks(session, document_id, chunks, embeddings, page_numbers)
    doc = await create_document(
        session,
        document_id=document_id,
        filename=file.filename or "unknown",
        chunk_count=len(chunks),
        file_size=file_size,
        file_type=file_type,
        page_count=page_count,
    )

    logger.info(
        "Ingest complete: document_id=%s type=%s chunks=%d pages=%s",
        document_id, file_type, len(chunks), page_count,
    )
    return IngestResponse(
        document_id=doc.id,
        filename=doc.filename,
        chunk_count=doc.chunk_count,
        file_type=doc.file_type,
        page_count=doc.page_count,
        status="processed",
    )
