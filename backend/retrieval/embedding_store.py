from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config import Config
from backend.retrieval.two_tower import TwoTowerEmbeddings
from backend.retrieval.lightgcn import LightGCNEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    name: str
    loaded: bool = False
    obj: Optional[Any] = None
    path: Optional[Path] = None


class EmbeddingStore:
    """Central in-memory store for user and item embeddings.

    Responsibilities:
    - Discover available embedding artifacts using Config paths
    - Lazy-load Two-Tower and LightGCN embeddings
    - Cache loaded models
    - Provide accessor methods for user/item embeddings and matrices
    - Never raise fatal exceptions on missing artifacts; log and mark unavailable
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self._lock = RLock()
        self.model_dir = Path(model_dir) if model_dir is not None else Path(Config.MODEL_DIR)
        self._models: Dict[str, ModelInfo] = {}
        # register known model names and default paths
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._models["two_tower"] = ModelInfo(
            name="two_tower", path=Path(Config.TWO_TOWER_MODEL_PATH)
        )
        self._models["lightgcn"] = ModelInfo(
            name="lightgcn", path=Path(Config.LIGHTGCN_MODEL_PATH)
        )

    def available_models(self) -> List[str]:
        return [name for name, info in self._models.items() if info.path and info.path.exists()]

    def get_model(self, name: str) -> Optional[Any]:
        info = self._models.get(name)
        if not info:
            return None
        if info.loaded:
            return info.obj
        # lazy load
        try:
            self.load_model(name)
        except Exception as exc:  # do not let loading crash callers
            logger.warning("Failed to load model %s: %s", name, exc)
            return None
        return info.obj

    def load(self, load_all: bool = False) -> None:
        """Discover and optionally load models.

        If load_all is False (default), only discovers available models; if True, attempts to load them.
        """
        with self._lock:
            for name, info in list(self._models.items()):
                if info.path and info.path.exists():
                    logger.info("Embedding artifact detected: %s at %s", name, info.path)
                    if load_all and not info.loaded:
                        try:
                            self.load_model(name)
                        except Exception as exc:
                            logger.warning("Error loading embedding %s: %s", name, exc)
                else:
                    logger.debug("Embedding artifact not found for %s (expected at %s)", name, info.path)

    def load_model(self, name: str) -> None:
        """Load a specific model by name into memory (lazy load)."""
        with self._lock:
            info = self._models.get(name)
            if not info:
                raise KeyError(f"Unknown model: {name}")
            if info.loaded:
                logger.debug("Model %s is already loaded", name)
                return

            path = info.path
            if not path or not path.exists():
                raise FileNotFoundError(f"Model {name} not found at {path}")

            logger.info("Loading embedding model '%s' from %s", name, path)
            if name == "two_tower":
                obj = TwoTowerEmbeddings.load(path)
            elif name == "lightgcn":
                obj = LightGCNEmbeddings.load(path)
            else:
                raise KeyError(f"Unsupported model: {name}")

            info.obj = obj
            info.loaded = True
            logger.info("Model '%s' loaded: users=%s items=%s", name, getattr(obj, 'user_embeddings', None) is not None, getattr(obj, 'item_embeddings', None) is not None)

    def is_loaded(self, name: Optional[str] = None) -> bool:
        if name:
            info = self._models.get(name)
            return bool(info and info.loaded)
        return any(info.loaded for info in self._models.values())

    def reload(self) -> None:
        """Unload and reload all known models."""
        with self._lock:
            for name, info in self._models.items():
                info.obj = None
                info.loaded = False
            self.load(load_all=True)

    # Embedding accessors
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Return user embedding for the first model that contains it. Order: two_tower, lightgcn."""
        # prefer Two-Tower
        tt = self.get_model("two_tower")
        if tt is not None:
            try:
                emb = tt.get_user_embedding(user_id)
                if emb is not None:
                    logger.debug("EmbeddingStore: user %s found in two_tower", user_id)
                    return np.asarray(emb, dtype=np.float32)
            except Exception:
                logger.exception("Error fetching user embedding from two_tower")
        lg = self.get_model("lightgcn")
        if lg is not None:
            try:
                emb = lg.get_user_embedding(user_id)
                if emb is not None:
                    logger.debug("EmbeddingStore: user %s found in lightgcn", user_id)
                    return np.asarray(emb, dtype=np.float32)
            except Exception:
                logger.exception("Error fetching user embedding from lightgcn")
        logger.debug("EmbeddingStore: user %s not found in any model", user_id)
        return None

    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Return item embedding for the first model that contains it. Order: two_tower, lightgcn."""
        tt = self.get_model("two_tower")
        if tt is not None:
            try:
                emb = tt.get_item_embedding(item_id)
                if emb is not None:
                    logger.debug("EmbeddingStore: item %s found in two_tower", item_id)
                    return np.asarray(emb, dtype=np.float32)
            except Exception:
                logger.exception("Error fetching item embedding from two_tower")
        lg = self.get_model("lightgcn")
        if lg is not None:
            try:
                emb = lg.get_item_embedding(item_id)
                if emb is not None:
                    logger.debug("EmbeddingStore: item %s found in lightgcn", item_id)
                    return np.asarray(emb, dtype=np.float32)
            except Exception:
                logger.exception("Error fetching item embedding from lightgcn")
        logger.debug("EmbeddingStore: item %s not found in any model", item_id)
        return None

    def get_item_matrix(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        info = self._models.get(name) if name else None
        if info and info.loaded and hasattr(info.obj, 'item_embeddings'):
            return np.asarray(info.obj.item_embeddings, dtype=np.float32)
        # fallback to two_tower then lightgcn
        for candidate in ["two_tower", "lightgcn"]:
            m = self._models.get(candidate)
            if m and m.loaded and hasattr(m.obj, 'item_embeddings'):
                return np.asarray(m.obj.item_embeddings, dtype=np.float32)
        return None

    def get_user_matrix(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        info = self._models.get(name) if name else None
        if info and info.loaded and hasattr(info.obj, 'user_embeddings'):
            return np.asarray(info.obj.user_embeddings, dtype=np.float32)
        for candidate in ["two_tower", "lightgcn"]:
            m = self._models.get(candidate)
            if m and m.loaded and hasattr(m.obj, 'user_embeddings'):
                return np.asarray(m.obj.user_embeddings, dtype=np.float32)
        return None

    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "loaded": info.loaded,
                "path": str(info.path) if info.path else None,
                "has_user_embeddings": bool(getattr(info.obj, 'user_embeddings', None) is not None),
                "has_item_embeddings": bool(getattr(info.obj, 'item_embeddings', None) is not None),
            }
            for name, info in self._models.items()
        }
