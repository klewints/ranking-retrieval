from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.retrieval.candidate_generator import CandidateGenerator, DefaultCandidateGenerator
from backend.retrieval.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        faiss_index: Optional[FaissIndex] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
    ):
        self.faiss_index = faiss_index or FaissIndex()
        self.candidate_generator = candidate_generator or DefaultCandidateGenerator()

    def load(self) -> None:
        self.faiss_index.load()
        logger.info("Retrieval FAISS index loaded from %s", self.faiss_index.index_path)

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
