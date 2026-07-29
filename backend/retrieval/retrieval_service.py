from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config import Config
from backend.retrieval.candidate_generator import CandidateGenerator, DefaultCandidateGenerator
from backend.retrieval.faiss_index import FaissIndex
from backend.retrieval.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        faiss_index: Optional[FaissIndex] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        tracks_df: Optional[pd.DataFrame] = None,
        embedding_store: Optional[EmbeddingStore] = None,
    ):
        self.faiss_index = faiss_index or FaissIndex()
        self.tracks_df = tracks_df
        self.embedding_store = embedding_store
        self.candidate_generator = candidate_generator

    def load(self) -> None:
        # Load or validate FAISS index (non-fatal)
        self.faiss_index.load()
        if self.tracks_df is None:
            self.tracks_df = pd.read_csv(Config.TRACKS_CLEANED_PATH)
        if self.candidate_generator is None:
            self.candidate_generator = DefaultCandidateGenerator(
                self.tracks_df,
                self.faiss_index,
                embedding_store=self.embedding_store,
            )
        logger.info("Retrieval initialized. FAISS index path: %s", self.faiss_index.index_path)

    def is_ready(self) -> bool:
        return self.faiss_index.is_loaded()

    def retrieve_by_user(self, user_id: str, limit: int = 20) -> List[str]:
        if not self.is_ready():
            raise RuntimeError(
                "Retrieval models are unavailable. FAISS index is not loaded."
            )

        return self.candidate_generator.retrieve_by_user(user_id, limit)

    def retrieve_by_search(self, search_results: List[Dict[str, Any]], limit: int = 20) -> List[str]:
        if not self.is_ready():
            raise RuntimeError(
                "Retrieval models are unavailable. FAISS index is not loaded."
            )

        return self.candidate_generator.retrieve_by_search(search_results, limit)
