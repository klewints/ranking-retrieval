from pydantic import BaseModel, Field
from typing import List, Optional


class SearchResult(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Blank Space"})
    category: str = Field(..., json_schema_extra={"example": "track"})
    score: float = Field(..., ge=0.0, le=100.0)


class SearchResponse(BaseModel):
    corrected_query: str = Field(..., json_schema_extra={"example": "Taylor Swift"})
    results: List[SearchResult] = Field(default_factory=list)


class CatalogSearchResult(BaseModel):
    id: str = Field(..., json_schema_extra={"example": "track-blank-space"})
    title: str = Field(..., json_schema_extra={"example": "Blank Space"})
    score: float = Field(..., ge=0.0, le=100.0)


class CatalogSearchResponse(BaseModel):
    corrected_query: str = Field(..., json_schema_extra={"example": "Taylor Swift"})
    results: List[CatalogSearchResult] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    track_id: str = Field(..., json_schema_extra={"example": "123"})
    track: str = Field(..., json_schema_extra={"example": "Blank Space"})
    artist: str = Field(..., json_schema_extra={"example": "Taylor Swift"})
    album: Optional[str] = Field(None, json_schema_extra={"example": "1989"})
    genres: List[str] = Field(default_factory=list)
    popularity: float = Field(..., json_schema_extra={"example": 85.0})
    score: float = Field(..., json_schema_extra={"example": 0.92})
    search_relevance: Optional[float] = Field(None, json_schema_extra={"example": 95.0})


class RecommendationResponse(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    query: Optional[str] = Field(None, description="Optional search query")
    corrected_query: Optional[str] = Field(None, description="Corrected search query from fuzzy matching")
    results: List[RecommendationItem] = Field(default_factory=list)


class SimilarResponse(BaseModel):
    track_id: str = Field(..., description="Track identifier to find similar songs")
    results: List[RecommendationItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    search_ready: bool
    retrieval_ready: bool
    ranking_ready: bool
    retrieval_status: Optional[dict] = None
    embedding_info: Optional[dict] = None
