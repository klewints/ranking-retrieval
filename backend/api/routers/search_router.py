import re

from fastapi import APIRouter, Query

from backend.api.schemas import CatalogSearchResponse
from backend.services.search_service import SearchService

router = APIRouter()


def _to_card(result: dict, index: int) -> dict:
    title = str(result.get("name", "")).strip() or f"Result {index + 1}"
    category = str(result.get("category", "item")).strip() or "item"
    score = float(result.get("score", 0.0))
    safe_id = re.sub(r"[^a-z0-9]+", "-", f"{category}-{title}".lower()).strip("-")
    return {
        "id": safe_id or f"item-{index + 1}",
        "title": title,
        "score": round(score, 2),
    }


@router.get("/search", response_model=CatalogSearchResponse)
def search(q: str = Query(..., min_length=1, description="Search query")):
    query = q.strip()

    try:
        search_service = SearchService()
        search_response = search_service.search(query, limit=20)
    except Exception:
        return {
            "corrected_query": query,
            "results": [],
        }

    raw_results = search_response.get("results", [])
    return {
        "corrected_query": search_response.get("corrected_query", query),
        "results": [_to_card(item, index) for index, item in enumerate(raw_results[:20])],
    }
