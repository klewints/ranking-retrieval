from fastapi import APIRouter, Depends, Query, HTTPException, Request
from backend.api.schemas import SearchResponse
from backend.services.search_service import SearchService

router = APIRouter()


def get_search_service(request: Request) -> SearchService:
    search_service = getattr(request.app.state, "search_service", None)
    if search_service is None:
        raise HTTPException(
            status_code=503,
            detail="Search service is not available. Application startup may still be in progress.",
        )
    return search_service


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    search_service: SearchService = Depends(get_search_service),
):
    try:
        return search_service.search(q)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
