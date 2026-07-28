"""Search service wrapper for the application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from backend.config import Config
from backend.search.search_engine import SearchEngine


class SearchService:
    def __init__(
        self,
        processed_dir: Optional[Union[str, Path]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
    ):
        if dataset_path is not None:
            self.dataset_path = Path(dataset_path)
        elif processed_dir is not None:
            self.dataset_path = Path(processed_dir) / "tracks_cleaned.csv"
        else:
            self.dataset_path = Path(Config.TRACKS_CLEANED_PATH)

        self.data = self._load_dataset()
        self.engine = SearchEngine(self.data)
        self.tracks = self.engine.index.get_candidates("track")
        self.artists = self.engine.index.get_candidates("artist")
        self.albums = self.engine.index.get_candidates("album")
        self.genres = self.engine.index.get_candidates("genre")

    def _load_dataset(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        return pd.read_csv(self.dataset_path)

    def search(self, query: str, limit: int = Config.DEFAULT_SEARCH_LIMIT) -> Dict[str, Any]:
        return self.engine.search(query, limit)