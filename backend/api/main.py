import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from backend.api.routers.recommend_router import router as recommend_router
from backend.api.routers.search_router import router as search_router
from backend.config import Config
from backend.retrieval.faiss_index import FaissIndex
from backend.retrieval.retrieval_service import RetrievalService
from backend.retrieval.ranking_model import RankingModelWrapper
from backend.retrieval.two_tower import TwoTowerEmbeddings
from backend.retrieval.lightgcn import LightGCNEmbeddings
from backend.retrieval.embedding_store import EmbeddingStore
from backend.services.recommendation_service import RecommendationService
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

    # initialize embedding store (detect artifacts)
    app.state.embedding_store = EmbeddingStore()
    try:
        if Config.AUTO_LOAD_MODELS:
            app.state.embedding_store.load(load_all=True)
        else:
            app.state.embedding_store.load(load_all=False)
        logger.info("EmbeddingStore initialized. Available models: %s", app.state.embedding_store.available_models())
    except Exception as exc:  # non-fatal
        logger.warning("EmbeddingStore could not be initialized: %s", exc)

    # Load ranking model (optional)
    ranking_model = None
    try:
        ranking_model = RankingModelWrapper.load(Config.RANKING_MODEL_PATH)
        logger.info("Ranking model loaded from %s", Config.RANKING_MODEL_PATH)
    except Exception as exc:
        logger.warning("Ranking model could not be loaded: %s", exc)

    # initialize retrieval service and attempt FAISS load
    app.state.retrieval_service = RetrievalService(
        faiss_index=FaissIndex(),
        tracks_df=app.state.search_service.data,
        embedding_store=app.state.embedding_store,
    )
    try:
        app.state.retrieval_service.load()
        logger.info("Retrieval initialized (FAISS may be disabled)")
    except Exception as exc:
        logger.warning(
            "Retrieval initialization error: %s. Retrieval endpoints may be unavailable.",
            exc,
        )

    # RetrievalManager orchestration
    from backend.retrieval.retrieval_manager import RetrievalManager

    app.state.retrieval_manager = RetrievalManager(
        tracks_df=app.state.search_service.data,
        embedding_store=app.state.embedding_store,
        faiss_index=FaissIndex(),
    )
    try:
        app.state.retrieval_manager.load()
        logger.info("RetrievalManager initialized")
    except Exception as exc:
        logger.warning("RetrievalManager failed to initialize: %s", exc)

    app.state.recommendation_service = RecommendationService(
        search_service=app.state.search_service,
        retrieval_service=app.state.retrieval_service,
        ranking_model=ranking_model,
        embedding_store=app.state.embedding_store,
        retrieval_manager=app.state.retrieval_manager,
    )

    yield


app = FastAPI(title="Music Recommendation Search", version="0.1", lifespan=lifespan)
app.include_router(search_router)
app.include_router(recommend_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "running",
        "service": "music search API",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=False)
