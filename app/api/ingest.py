"""
POST /ingest         — validate and queue a document for background processing.
GET  /status/{job_id} — poll the status of a queued ingestion job.

Ingest is intentionally non-blocking: the endpoint returns 202 immediately
after validation. All CPU/IO-heavy work (PDF parsing, embedding, DB writes)
runs in a FastAPI BackgroundTask so the HTTP worker is free for other requests.
"""
import io
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, status

from app.db.metadata import create_document, session_factory
from app.db.vector_store import insert_chunks
from app.jobs import JobRecord, create_job, get_job, update_job
from app.limiter import limiter
from app.models.schemas import IngestJobAccepted, JobStatusResponse
from app.services.chunker import chunk_pages, chunk_text
from app.services.embedder import embed_texts

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ── Text extraction helpers ───────────────────────────────────────────────────
# These raise ValueError (not HTTPException) so they're safe to call from
# background tasks where there is no HTTP response context.

def _parse_pdf_pages(data: bytes) -> list[tuple[int, str]]:
    """Return [(1-based page_number, page_text), ...] or raise ValueError."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [
            (i + 1, reader.pages[i].extract_text() or "")
            for i in range(len(reader.pages))
        ]
    except Exception as exc:
        raise ValueError(f"Could not extract text from PDF: {exc}") from exc


def _decode_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


# ── Background processing ─────────────────────────────────────────────────────

async def _process_document(
    job_id: str,
    filename: str,
    content_type: str,
    data: bytes,
) -> None:
    """Run the full ingest pipeline and update job state when done."""
    update_job(job_id, status="processing")
    logger.info("Background ingest started: job=%s file=%s", job_id, filename)

    try:
        file_size = len(data)

        # ── Extract ───────────────────────────────────────────────────────────
        if content_type == "application/pdf":
            pdf_pages = _parse_pdf_pages(data)
            all_text = "".join(t for _, t in pdf_pages)
            if not all_text.strip():
                raise ValueError("No extractable text found in the uploaded PDF.")
            chunk_dicts = chunk_pages(pdf_pages)
            file_type = "pdf"
            page_count: int | None = len(pdf_pages)
        else:
            raw_text = _decode_txt(data)
            if not raw_text.strip():
                raise ValueError("No extractable text found in the uploaded file.")
            chunk_dicts = [{"chunk_text": c, "page_number": None} for c in chunk_text(raw_text)]
            file_type = "txt"
            page_count = None

        if not chunk_dicts:
            raise ValueError("Document produced zero chunks after processing.")

        chunks = [d["chunk_text"] for d in chunk_dicts]
        page_numbers: list[int | None] = [d["page_number"] for d in chunk_dicts]

        # ── Embed ─────────────────────────────────────────────────────────────
        embeddings = await embed_texts(chunks)

        # ── Store ─────────────────────────────────────────────────────────────
        document_id = str(uuid.uuid4())
        async with session_factory() as session:
            await insert_chunks(session, document_id, chunks, embeddings, page_numbers)
            await create_document(
                session,
                document_id=document_id,
                filename=filename,
                chunk_count=len(chunks),
                file_size=file_size,
                file_type=file_type,
                page_count=page_count,
            )

        update_job(job_id, status="complete", document_id=document_id, chunk_count=len(chunks))
        logger.info(
            "Background ingest complete: job=%s document=%s chunks=%d",
            job_id, document_id, len(chunks),
        )

    except Exception as exc:
        logger.exception("Background ingest failed: job=%s", job_id)
        update_job(job_id, status="failed", error=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for background processing",
)
@limiter.limit("10/minute")
async def ingest_document(
    request: Request,
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> IngestJobAccepted:
    """
    Validate the uploaded file synchronously, then immediately return a
    `job_id`. Chunking, embedding, and storage happen in the background.
    Poll `GET /status/{job_id}` to track completion.
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in {"application/pdf", "text/plain"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Only PDF and TXT are accepted.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 20 MB limit.",
        )

    # ── Queue ─────────────────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    create_job(job_id, filename)

    background_tasks.add_task(
        _process_document,
        job_id=job_id,
        filename=filename,
        content_type=content_type,
        data=data,
    )

    logger.info("Ingest queued: job=%s file=%s size=%d", job_id, filename, len(data))
    return IngestJobAccepted(job_id=job_id)


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check the status of an ingestion job",
)
async def get_ingest_status(job_id: str) -> JobStatusResponse:
    """
    Returns the current state of a background ingestion job.

    | `status`     | Meaning                                          |
    |---|---|
    | `pending`    | Queued, not yet started                          |
    | `processing` | Actively chunking / embedding / writing to DB    |
    | `complete`   | Done — `document_id` is populated                |
    | `failed`     | Error — see the `error` field for details        |
    """
    record: JobRecord | None = get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        filename=record.filename,
        document_id=record.document_id,
        chunk_count=record.chunk_count,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
