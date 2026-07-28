from __future__ import annotations

import ast
import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SEARCH_FIELDS = {
    "track": ["track_name", "name", "title", "track_display", "title_display"],
    "artist": ["artist_name", "artist", "artists", "artist_display"],
    "album": ["album_name", "album", "album_display"],
    "genre": ["genres_list", "genre", "genres", "genres_display"],
}


class SearchIndex:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.candidates = self._build_candidates()

    def _find_column(self, candidate_names: List[str]) -> Optional[str]:
        for name in candidate_names:
            if name in self.dataframe.columns:
                return name
        return None

    def _build_candidates(self) -> Dict[str, List[str]]:
        candidates: Dict[str, List[str]] = {}

        for category, fields in SEARCH_FIELDS.items():
            column = self._find_column(fields)
            if not column:
                continue

            if category == "genre":
                candidates[category] = self._unique_genres(column)
            else:
                candidates[category] = self._unique_values(column)

            if not candidates[category]:
                logger.debug("No candidates found for category '%s'", category)

        return candidates

    def _unique_values(self, column: str) -> List[str]:
        values = (
            self.dataframe[column]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )
        return sorted(values.unique().tolist())

    def _unique_genres(self, column: str) -> List[str]:
        genre_names: set[str] = set()

        for raw in self.dataframe[column].dropna().astype(str):
            for genre in self._parse_genre_list(raw):
                genre_names.add(genre)

        return sorted(genre_names)

    def _parse_genre_list(self, raw: str) -> List[str]:
        raw = raw.strip()
        if not raw:
            return []

        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if item]
        except (ValueError, SyntaxError):
            pass

        separators = [",", ";", "|"]
        for separator in separators:
            if separator in raw:
                return [part.strip() for part in raw.split(separator) if part.strip()]

        return [raw]

    def get_candidates(self, category: str) -> List[str]:
        return self.candidates.get(category, [])

    def categories(self) -> List[str]:
        return list(self.candidates.keys())

    def get_all_candidates(self) -> List[str]:
        return [candidate for candidates in self.candidates.values() for candidate in candidates]
