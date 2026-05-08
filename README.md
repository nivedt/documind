# DocuMind

Upload PDF or TXT documents and ask natural-language questions. Get back accurate answers with source citations showing exactly which part of the document each answer came from.

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
#    → set OPENAI_API_KEY in .env

# 2. Start the stack
docker compose up --build

# 3. API is live
#    REST: http://localhost:8000
#    Docs: http://localhost:8000/docs
```

## API Reference

### `POST /ingest`
Upload a document (PDF or TXT, max 20 MB).

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your-document.pdf"
```

Response:
```json
{
  "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename": "your-document.pdf",
  "chunk_count": 42,
  "status": "processed"
}
```

### `POST /query`
Ask a question about an ingested document.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"document_id": "<id>", "question": "What are the main findings?"}'
```

Response:
```json
{
  "answer": "The main findings are ...",
  "sources": [
    {"chunk_text": "...", "chunk_index": 7},
    {"chunk_text": "...", "chunk_index": 8}
  ]
}
```

### `GET /documents`
List all ingested documents.

### `GET /health`
Returns `{ "status": "ok" }`.

## Architecture

```
Upload → extract text → chunk (512 tok / 50 overlap)
       → embed (text-embedding-3-small)
       → store vectors (pgvector) + metadata (postgres)

Query  → embed question
       → cosine similarity search → top 5 chunks
       → grounded GPT-4o prompt
       → answer + source citations
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:pass@host/db` |
| `POSTGRES_USER` | Docker | Postgres username |
| `POSTGRES_PASSWORD` | Docker | Postgres password |
| `POSTGRES_DB` | Docker | Postgres database name |
| `LOG_LEVEL` | No | Default: `info` |
