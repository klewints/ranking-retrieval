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
    return {
        'status': 'ok',
        'search_ready': getattr(service.search_service, 'engine', None) is not None,
        'retrieval_ready': service.retrieval_service.is_ready(),
        'ranking_ready': service.ranking_model is not None,
    }
