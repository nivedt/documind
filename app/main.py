"""
DocuMind — FastAPI application entry point.
"""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.db.metadata import init_db
from app.models.schemas import HealthResponse

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DocuMind — initialising database…")
    await init_db()
    logger.info("DocuMind is ready")
    yield
    logger.info("DocuMind shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind",
    description="Upload documents and ask natural language questions with source citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, tags=["Ingest"])
app.include_router(query_router, tags=["Query & Documents"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
