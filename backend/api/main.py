import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from backend.api.routers.recommend_router import router as recommend_router
from backend.api.routers.search_router import router as search_router
from backend.config import Config
from backend.retrieval.faiss_index import FaissIndex
from backend.retrieval.lightgcn import LightGCNEmbeddings
from backend.retrieval.retrieval_service import RetrievalService
from backend.retrieval.ranking_model import RankingModelWrapper
from backend.retrieval.two_tower import TwoTowerEmbeddings
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

    two_tower_model = None
    lightgcn_model = None
    ranking_model = None

    try:
        two_tower_model = TwoTowerEmbeddings.load(Config.TWO_TOWER_MODEL_PATH)
        logger.info("Two-Tower model loaded from %s", Config.TWO_TOWER_MODEL_PATH)
    except Exception as exc:
        logger.warning("Two-Tower model could not be loaded: %s", exc)

    try:
        lightgcn_model = LightGCNEmbeddings.load(Config.LIGHTGCN_MODEL_PATH)
        logger.info("LightGCN model loaded from %s", Config.LIGHTGCN_MODEL_PATH)
    except Exception as exc:
        logger.warning("LightGCN model could not be loaded: %s", exc)

    try:
        ranking_model = RankingModelWrapper.load(Config.RANKING_MODEL_PATH)
        logger.info("Ranking model loaded from %s", Config.RANKING_MODEL_PATH)
    except Exception as exc:
        logger.warning("Ranking model could not be loaded: %s", exc)

    app.state.retrieval_service = RetrievalService(
        faiss_index=FaissIndex(),
        tracks_df=app.state.search_service.data,
        two_tower=two_tower_model,
        lightgcn=lightgcn_model,
    )
    try:
        app.state.retrieval_service.load()
        logger.info("FAISS retrieval index loaded successfully")
    except Exception as exc:
        logger.warning(
            "Retrieval index could not be loaded: %s. Retrieval endpoints may be unavailable.",
            exc,
        )

    app.state.recommendation_service = RecommendationService(
        search_service=app.state.search_service,
        retrieval_service=app.state.retrieval_service,
        ranking_model=ranking_model,
        two_tower=two_tower_model,
        lightgcn=lightgcn_model,
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
