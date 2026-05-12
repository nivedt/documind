"""
Request logging middleware.

Logs every HTTP request with method, path, status code, and elapsed time.

Example output:
    POST /ingest 202 143.7ms
    POST /query  200  87.2ms
    GET  /health 200   1.1ms
"""
import logging
import time
from typing import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("documind.access")


async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%-6s %-40s %d  %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response
