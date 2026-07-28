from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd

from backend.config import Config
from backend.search.fuzzy_search import SearchMatcher
from backend.search.search_index import SearchIndex


@dataclass(frozen=True)
class SearchResultItem:
    name: str
    category: str
    score: float


class SearchEngine:
    def __init__(self, dataframe: pd.DataFrame):
        self.index = SearchIndex(dataframe)
        self.matcher = SearchMatcher()

    def search(self, query: str, limit: int = Config.DEFAULT_SEARCH_LIMIT) -> Dict[str, object]:
        if not query or not query.strip():
            return {"corrected_query": query or "", "results": []}

        cleaned_query = query.strip()
        corrected_query = self._correct_query(cleaned_query)
        results: List[SearchResultItem] = []

        for category in self.index.categories():
            candidates = self.index.get_candidates(category)
            matches = self.matcher.find_matches(corrected_query, candidates, limit)
            for match in matches:
                results.append(
                    SearchResultItem(
                        name=match["name"], category=category, score=match["score"]
                    )
                )

        results.sort(key=lambda item: (item.score, item.category), reverse=True)
        return {
            "corrected_query": corrected_query,
            "results": [asdict(item) for item in results[:limit]],
        }

    def _correct_query(self, query: str) -> str:
        candidate = self.matcher.find_best_match(query, self.index.get_all_candidates())
        if candidate and candidate["name"].strip().lower() != query.strip().lower():
            return candidate["name"]
        return query
