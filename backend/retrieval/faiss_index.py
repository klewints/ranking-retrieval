from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
        self.track_ids: List[str] = []

    def load(self) -> None:
        """Attempt to load FAISS index and track id mapping.

        Loading is non-fatal: missing faiss library or missing files will result in a logged warning and a disabled index.
        """
        if faiss is None:
            logger.warning("FAISS library not installed; FAISS features will be disabled.")
            self.index = None
            self.track_ids = []
            return

        if not self.index_path.exists():
            logger.warning("FAISS index file does not exist at %s", self.index_path)
            self.index = None
            self.track_ids = []
            return

        if not self.track_ids_path.exists():
            logger.warning("FAISS track ID mapping does not exist at %s", self.track_ids_path)
            self.index = None
            self.track_ids = []
            return

        try:
            logger.info("Loading FAISS index from %s", self.index_path)
            self.index = faiss.read_index(str(self.index_path))
            self.track_ids = self._load_track_ids()
            # basic validation
            if self.index is not None:
                dim = getattr(self.index, 'd', None)
                ntotal = getattr(self.index, 'ntotal', None)
                if ntotal is not None and self.track_ids and ntotal != len(self.track_ids):
                    logger.warning(
                        "FAISS index ntotal (%s) does not match track_ids length (%s)", ntotal, len(self.track_ids)
                    )
        except Exception as exc:
            logger.exception("Failed to load FAISS index: %s", exc)
            self.index = None
            self.track_ids = []

    def _load_track_ids(self) -> List[str]:
        import pickle

        with open(self.track_ids_path, 'rb') as handle:
            return pickle.load(handle)

    def is_loaded(self) -> bool:
        return self.index is not None and bool(self.track_ids)

    def get_index_info(self) -> Dict[str, Any]:
        if not self.index:
            return {"loaded": False, "path": str(self.index_path)}
        return {
            "loaded": True,
            "path": str(self.index_path),
            "dim": getattr(self.index, 'd', None),
            "ntotal": getattr(self.index, 'ntotal', None),
        }

    def reload(self) -> None:
        self.load()

    def rebuild(self) -> None:
        raise NotImplementedError("FAISS rebuild is handled by offline scripts; call scripts/build_faiss.py to recreate indexes.")

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
