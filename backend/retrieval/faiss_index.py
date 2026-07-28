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
    def __init__(self, index_path: Path = Config.FAISS_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index = None

    def load(self) -> None:
        if faiss is None:
            raise RuntimeError(
                "FAISS library is not installed. Install faiss to use retrieval index loading."
            )

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index file does not exist at {self.index_path}"
            )

        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))

    def is_loaded(self) -> bool:
        return self.index is not None

    def query(self, embedding: np.ndarray, top_k: int = Config.RETRIEVAL_TOP_K) -> Tuple[List[int], List[float]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        embedding = np.asarray(embedding, dtype=np.float32)
        distances, indices = self.index.search(embedding, top_k)
        return indices[0].tolist(), distances[0].tolist()
