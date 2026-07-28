from pydantic import BaseModel, Field
from typing import List


class SearchResult(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Blank Space"})
    category: str = Field(..., json_schema_extra={"example": "track"})
    score: float = Field(..., ge=0.0, le=100.0)


class SearchResponse(BaseModel):
    corrected_query: str = Field(..., json_schema_extra={"example": "Taylor Swift"})
    results: List[SearchResult] = Field(default_factory=list)
