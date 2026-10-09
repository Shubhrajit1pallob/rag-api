from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import deps
from app.api.routes import documents, query


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # Only close a pool that was actually created; never open one just to close it.
    if deps.get_vector_store.cache_info().currsize:
        deps.get_vector_store().close()


app = FastAPI(title="rag-api", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
