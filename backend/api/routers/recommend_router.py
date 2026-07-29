from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import Optional

from backend.api.schemas import HealthResponse, RecommendationResponse, SimilarResponse
from backend.services.recommendation_service import RecommendationService

router = APIRouter()


def get_recommendation_service(request: Request) -> RecommendationService:
    service = getattr(request.app.state, 'recommendation_service', None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail='Recommendation service is not available. Application startup may still be in progress.',
        )
    return service


@router.get('/recommend', response_model=RecommendationResponse)
def recommend(
    user_id: Optional[str] = Query(None, description='Optional user identifier'),
    q: Optional[str] = Query(None, description='Optional search query'),
    limit: int = Query(20, ge=1, le=50, description='Number of recommendations to return'),
    service: RecommendationService = Depends(get_recommendation_service),
):
    if not user_id and not q:
        raise HTTPException(status_code=400, detail='Please provide a user_id or search query')

    try:
        return service.recommend(user_id=user_id, query=q, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get('/similar', response_model=SimilarResponse)
def similar(
    track_id: str = Query(..., description='Track identifier to find similar songs'),
    limit: int = Query(20, ge=1, le=50, description='Number of similar songs to return'),
    service: RecommendationService = Depends(get_recommendation_service),
):
    try:
        return service.similar(track_id=track_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get('/health', response_model=HealthResponse)
def health(service: RecommendationService = Depends(get_recommendation_service)):
    retrieval_ready = service.retrieval_service.is_ready()
    # prefer retrieval_manager status when available
    retrieval_status = None
    embedding_info = None
    try:
        if getattr(service, 'retrieval_manager', None) is not None:
            retrieval_status = service.retrieval_manager.get_status()
        else:
            # provide minimal diagnostics
            retrieval_status = {
                'faiss': service.retrieval_service.faiss_index.get_index_info() if hasattr(service.retrieval_service.faiss_index, 'get_index_info') else {'loaded': False},
                'candidate_generator_initialized': getattr(service.retrieval_service, 'candidate_generator', None) is not None,
            }
    except Exception:
        retrieval_status = {'faiss': {'loaded': False}}

    try:
        embedding_info = service.embedding_store.get_model_info() if getattr(service, 'embedding_store', None) is not None else {}
    except Exception:
        embedding_info = {}

    return {
        'status': 'ok',
        'search_ready': getattr(service.search_service, 'engine', None) is not None,
        'retrieval_ready': retrieval_ready,
        'ranking_ready': service.ranking_model is not None,
        'retrieval_status': retrieval_status,
        'embedding_info': embedding_info,
    }
