import logging
from contextlib import asynccontextmanager

import anthropic
import psycopg
import voyageai.error
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import deps
from app.api.routes import documents, query

logger = logging.getLogger("rag_api")

UPSTREAM_UNAVAILABLE = "Upstream service unavailable, try again later"


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # Only close a pool that was actually created; never open one just to close it.
    if deps.get_pool.cache_info().currsize:
        deps.get_pool().close()


async def upstream_unavailable(request: Request, exc: Exception) -> JSONResponse:
    # Log the real cause server-side; the client only ever sees a generic message.
    logger.error(
        "Upstream failure on %s %s: %r", request.method, request.url.path, exc, exc_info=exc
    )
    return JSONResponse({"detail": UPSTREAM_UNAVAILABLE}, status_code=503)


app = FastAPI(title="rag-api", lifespan=lifespan)
# psycopg_pool.PoolTimeout is a subclass of psycopg.OperationalError.
for upstream_error in (
    voyageai.error.VoyageError,
    anthropic.APIError,
    psycopg.OperationalError,
):
    app.add_exception_handler(upstream_error, upstream_unavailable)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
