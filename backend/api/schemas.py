from pydantic import BaseModel, Field
from typing import List


class SearchResult(BaseModel):
    name: str = Field(..., example="Blank Space")
    score: float = Field(..., ge=0.0, le=100.0)


class SearchResponse(BaseModel):
    tracks: List[SearchResult] = Field(default_factory=list)
    artists: List[SearchResult] = Field(default_factory=list)
    albums: List[SearchResult] = Field(default_factory=list)
