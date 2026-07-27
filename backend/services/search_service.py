"""
Stage 2 Search Service

Provides typo-tolerant search over cleaned music metadata.

Used by:
backend/api/routers/search_router.py
"""

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from rapidfuzz import fuzz, process


class SearchService:
    """
    Search service for songs, artists and albums.

    Loads cleaned dataset once during startup and performs
    fuzzy matching using RapidFuzz.
    """

    def __init__(self):
        self.data = self._load_dataset()

        self.tracks = []
        self.artists = []
        self.albums = []

        self._prepare_indexes()

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load processed music dataset.
        """

        base_dir = Path(__file__).resolve().parents[2]

        dataset_path = (
            base_dir
            / "data"
            / "processed"
            / "tracks_cleaned.csv"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        return pd.read_csv(dataset_path)


    def _find_column(self, possible_names):
        """
        Find matching dataframe column.
        """

        for name in possible_names:
            if name in self.data.columns:
                return name

        return None


    def _prepare_indexes(self):
        """
        Create searchable indexes.
        """

        track_col = self._find_column(
            [
                "track_name",
                "name",
                "title"
            ]
        )

        artist_col = self._find_column(
            [
                "artist_name",
                "artist",
                "artists"
            ]
        )

        album_col = self._find_column(
            [
                "album_name",
                "album"
            ]
        )


        if track_col:
            self.tracks = (
                self.data[track_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        if artist_col:
            self.artists = (
                self.data[artist_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        if album_col:
            self.albums = (
                self.data[album_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


    def search(
        self,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search tracks, artists and albums.

        Example:
            search("taylr swft")

        returns Taylor Swift matches.
        """

        if not query:
            return {
                "tracks": [],
                "artists": [],
                "albums": []
            }


        query = query.strip()


        return {
            "tracks": self._fuzzy_search(
                query,
                self.tracks,
                limit
            ),

            "artists": self._fuzzy_search(
                query,
                self.artists,
                limit
            ),

            "albums": self._fuzzy_search(
                query,
                self.albums,
                limit
            )
        }


    def _fuzzy_search(
        self,
        query: str,
        choices: List[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        RapidFuzz typo tolerant matching.
        """

        if not choices:
            return []


        results = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,
            limit=limit
        )


        matches = []

        for item, score, _ in results:

            matches.append(
                {
                    "name": item,
                    "score": round(score, 2)
                }
            )


        return matches



# Singleton instance used by FastAPI
default_search_service = SearchService()