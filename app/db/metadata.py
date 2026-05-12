"""
Document metadata persistence — SQLAlchemy async ORM.

Table: documents
    id           UUID primary key
    filename     original filename
    chunk_count  number of chunks stored for this document
    created_at   ingestion timestamp
    uploaded_at  ingestion timestamp (explicit upload time)
    file_size    file size in bytes
    file_type    'pdf' or 'txt'
    page_count   number of pages (NULL for txt files)
"""
import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

_SessionFactory = async_sessionmaker(engine, expire_on_commit=False)

# Public alias — use this in background tasks that cannot use Depends()
session_factory = _SessionFactory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _SessionFactory() as session:
        yield session


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True)
    filename = Column(String(512), nullable=False)
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    uploaded_at = Column(DateTime(timezone=True), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    file_type = Column(String(8), nullable=True)   # 'pdf' or 'txt'
    page_count = Column(Integer, nullable=True)     # NULL for txt files


async def init_db() -> None:
    """Create tables and enable pgvector extension."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialised")


async def create_document(
    session: AsyncSession,
    document_id: str,
    filename: str,
    chunk_count: int,
    file_size: int | None = None,
    file_type: str | None = None,
    page_count: int | None = None,
) -> Document:
    now = datetime.now(timezone.utc)
    doc = Document(
        id=document_id,
        filename=filename,
        chunk_count=chunk_count,
        created_at=now,
        uploaded_at=now,
        file_size=file_size,
        file_type=file_type,
        page_count=page_count,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    logger.info(
        "Saved document metadata: id=%s filename=%s chunks=%d type=%s size=%s pages=%s",
        document_id, filename, chunk_count, file_type, file_size, page_count,
    )
    return doc


async def list_documents(session: AsyncSession) -> list[Document]:
    from sqlalchemy import select
    result = await session.execute(select(Document).order_by(Document.created_at.desc()))
    return list(result.scalars().all())


async def get_document(session: AsyncSession, document_id: str) -> Document | None:
    from sqlalchemy import select
    result = await session.execute(
        select(Document).where(Document.id == document_id)
    )
    return result.scalar_one_or_none()
