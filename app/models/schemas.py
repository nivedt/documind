from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str = "processed"


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    document_id: str = Field(..., description="UUID of the ingested document")
    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")


class SourceChunk(BaseModel):
    chunk_text: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentMeta(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
