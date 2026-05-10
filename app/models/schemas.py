from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    file_type: Optional[str] = None
    page_count: Optional[int] = None
    status: str = "processed"


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    document_id: str = Field(..., description="UUID of the ingested document")
    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")


class SourceChunk(BaseModel):
    chunk_text: str
    chunk_index: int
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentMeta(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    created_at: datetime
    uploaded_at: Optional[datetime] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    page_count: Optional[int] = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
