from fastapi import APIRouter, Query, HTTPException
from typing import List
from backend.services.search_service import default_search_service
from backend.api.schemas import SearchResponse, SearchResult

router = APIRouter()

@router.get('/search', response_model=SearchResponse)
def search(q: str = Query(..., min_length=1, description='Search query')):
    try:
        results = default_search_service.search(q, limit=20)
        items = [SearchResult(**r) for r in results]
        return SearchResponse(results=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
