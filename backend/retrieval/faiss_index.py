from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from backend.config import Config

logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(
        self,
        index_path: Path = Config.FAISS_INDEX_PATH,
        track_ids_path: Path = Config.FAISS_TRACK_IDS_PATH,
    ):
        self.index_path = Path(index_path)
        self.track_ids_path = Path(track_ids_path)
        self.index = None
        self.track_ids: list[str] = []

    def load(self) -> None:
        if faiss is None:
            raise RuntimeError(
                "FAISS library is not installed. Install faiss to use retrieval index loading."
            )

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index file does not exist at {self.index_path}"
            )
        if not self.track_ids_path.exists():
            raise FileNotFoundError(
                f"FAISS track ID mapping does not exist at {self.track_ids_path}"
            )

        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        self.track_ids = self._load_track_ids()

    def _load_track_ids(self) -> list[str]:
        import pickle

        with open(self.track_ids_path, 'rb') as handle:
            return pickle.load(handle)

    def is_loaded(self) -> bool:
        return self.index is not None and bool(self.track_ids)

    def get_track_ids(self, indices: List[int]) -> List[str]:
        return [self.track_ids[i] for i in indices if 0 <= i < len(self.track_ids)]

    def query(self, embedding: np.ndarray, top_k: int = Config.RETRIEVAL_TOP_K) -> Tuple[List[str], List[float]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        embedding = np.asarray(embedding, dtype=np.float32)
        distances, indices = self.index.search(embedding, top_k)
        return self.get_track_ids(indices[0].tolist()), distances[0].tolist()
