# DocuMind — Project Context

## Description
DocuMind is a production-ready REST API that lets users upload PDF or TXT documents,
then ask natural-language questions and receive accurate answers with source citations
pinpointing which part of the document the answer came from.

## Stack
| Layer | Technology |
|---|---|
| API framework | FastAPI (Python 3.12) |
| RAG orchestration | LangChain 0.2 |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| LLM | GPT-4o |
| Vector store | PostgreSQL 15 + pgvector extension |
| ORM | SQLAlchemy 2 (async) |
| Container | Docker Compose (app + pgvector/pgvector:pg15) |
| Validation | Pydantic v2 |
| Env config | python-dotenv |

## Stage 1 — Complete ✅

- [x] `requirements.txt` — all pinned dependencies
- [x] `Dockerfile` — python:3.12-slim, non-root-ready
- [x] `docker-compose.yml` — app + pgvector postgres, healthcheck, volume
- [x] `.env.example` — all required env vars documented
- [x] `app/models/schemas.py` — Pydantic v2 request/response schemas
- [x] `app/db/metadata.py` — SQLAlchemy async ORM, `documents` table, CRUD helpers
- [x] `app/db/vector_store.py` — `document_chunks` table, `insert_chunks`, `similarity_search`
- [x] `app/services/chunker.py` — tiktoken 512-token chunks, 50-token overlap
- [x] `app/services/embedder.py` — async OpenAI embeddings with tenacity retry
- [x] `app/services/retriever.py` — embeds question → pgvector cosine search
- [x] `app/services/generator.py` — grounded GPT-4o prompt → answer string
- [x] `app/api/ingest.py` — `POST /ingest` with file validation, PDF/TXT extraction
- [x] `app/api/query.py` — `POST /query`, `GET /documents`
- [x] `app/main.py` — FastAPI app, lifespan init_db, CORS, routers, `/health`

## Architecture Decisions

**Async throughout** — `asyncpg` driver + SQLAlchemy async session; all service
functions are `async def` so FastAPI can handle concurrent requests without blocking.

**pgvector over a dedicated vector DB** — keeps the stack simple (one database) while
still giving production-grade cosine similarity via the `<=>` operator and an IVFFlat
index (to be added in Stage 2).

**Chunker uses tiktoken** — token counts are model-accurate; a naive character split
would silently over- or under-fill the embedding context window.

**Retry with exponential back-off** — `tenacity` wraps all OpenAI calls so transient
rate-limit or network errors recover automatically.

**Temperature 0 for generation** — answers must be grounded and reproducible; creative
variation would introduce hallucinations on a factual QA task.

**Document existence check before query** — the query route validates the document ID
against the `documents` table, not just the chunks table, to return a clean 404.

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `DATABASE_URL` | Async SQLAlchemy URL — `postgresql+asyncpg://...` |
| `POSTGRES_USER` | Postgres username (Docker Compose) |
| `POSTGRES_PASSWORD` | Postgres password (Docker Compose) |
| `POSTGRES_DB` | Postgres database name (Docker Compose) |
| `LOG_LEVEL` | Python log level, default `info` |

## Running Locally

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY in .env
docker compose up --build
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

## What to Build Next — Stage 2

- [ ] **IVFFlat index** on `document_chunks.embedding` for sub-linear search at scale
- [ ] **Alembic migrations** to manage schema changes safely in production
- [ ] **Authentication** — API key or JWT middleware
- [ ] **Rate limiting** — per-IP or per-API-key request throttling
- [ ] **Streaming responses** — SSE or WebSocket for `/query` so answers stream token-by-token
- [ ] **Async background ingestion** — return a job ID immediately; process in a Celery/ARQ worker
- [ ] **Multi-document query** — accept a list of `document_id`s and merge retrieval results
- [ ] **Evaluation harness** — RAGAS or a custom script to measure answer faithfulness/relevance
- [ ] **CI/CD** — GitHub Actions: lint (ruff), type-check (mypy), integration tests (pytest + testcontainers)
- [ ] **Observability** — structured JSON logs, Prometheus metrics endpoint, OpenTelemetry tracing
