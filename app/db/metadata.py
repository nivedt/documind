"""
Document metadata persistence — SQLAlchemy async ORM.

Table: documents
    id           UUID primary key
    filename     original filename
    chunk_count  number of chunks stored for this document
    created_at   ingestion timestamp
"""
import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import Column, DateTime, Integer, String, text
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
) -> Document:
    doc = Document(
        id=document_id,
        filename=filename,
        chunk_count=chunk_count,
        created_at=datetime.now(timezone.utc),
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    logger.info("Saved document metadata: id=%s filename=%s chunks=%d", document_id, filename, chunk_count)
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
