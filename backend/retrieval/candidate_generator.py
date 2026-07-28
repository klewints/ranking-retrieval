from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class CandidateGenerator(ABC):
    @abstractmethod
    def retrieve_by_user(self, user_id: str, limit: int = 20) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_by_search(self, search_results: List[Dict[str, Any]], limit: int = 20) -> List[str]:
        raise NotImplementedError


class DefaultCandidateGenerator(CandidateGenerator):
    def retrieve_by_user(self, user_id: str, limit: int = 20) -> List[str]:
        raise NotImplementedError("User-based retrieval is not implemented yet")

    def retrieve_by_search(self, search_results: List[Dict[str, Any]], limit: int = 20) -> List[str]:
        raise NotImplementedError("Search-based retrieval is not implemented yet")
