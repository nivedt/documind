"""
In-memory job store for background ingestion tasks.

Suitable for single-process deployments. For multi-worker production use,
replace with a Redis-backed store or a dedicated jobs table in PostgreSQL.
Jobs are evicted (oldest-first) once the store exceeds _MAX_JOBS entries.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

logger = logging.getLogger(__name__)

JobStatus = Literal["pending", "processing", "complete", "failed"]
_MAX_JOBS = 1000
_store: dict[str, "JobRecord"] = {}


@dataclass
class JobRecord:
    job_id: str
    filename: str
    status: JobStatus = "pending"
    document_id: Optional[str] = None
    chunk_count: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None


def create_job(job_id: str, filename: str) -> JobRecord:
    record = JobRecord(job_id=job_id, filename=filename)
    _store[job_id] = record
    _evict_if_needed()
    return record


def get_job(job_id: str) -> Optional[JobRecord]:
    return _store.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    record = _store.get(job_id)
    if record is None:
        logger.warning("update_job called for unknown job_id=%s", job_id)
        return
    for key, value in kwargs.items():
        setattr(record, key, value)
    record.updated_at = datetime.now(timezone.utc)


def _evict_if_needed() -> None:
    if len(_store) > _MAX_JOBS:
        oldest = sorted(_store.values(), key=lambda j: j.created_at)[:100]
        for j in oldest:
            del _store[j.job_id]
        logger.debug("Evicted %d stale job records", len(oldest))
