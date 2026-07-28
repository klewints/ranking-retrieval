import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from backend.api.routers.search_router import router as search_router
from backend.retrieval.retrieval_service import RetrievalService
from backend.services.search_service import SearchService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading processed dataset and building search index")
    app.state.search_service = SearchService()
    total_candidates = len(app.state.search_service.engine.index.get_all_candidates())
    logger.info("Search index loaded with %d candidates", total_candidates)

    app.state.retrieval_service = RetrievalService()
    try:
        app.state.retrieval_service.load()
        logger.info("FAISS retrieval index loaded successfully")
    except Exception as exc:
        logger.warning(
            "Retrieval index could not be loaded: %s. Retrieval endpoints may be unavailable.",
            exc,
        )

    yield


app = FastAPI(title="Music Recommendation Search", version="0.1", lifespan=lifespan)
app.include_router(search_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "running",
        "service": "music search API",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=False)
