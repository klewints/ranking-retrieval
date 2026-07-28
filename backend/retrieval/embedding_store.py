from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import Config


class EmbeddingStore:
    def __init__(self, embedding_path: Path = Config.EMBEDDING_MODEL_PATH):
        self.embedding_path = Path(embedding_path)
        self.embeddings = None

    def load(self) -> None:
        if not self.embedding_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {self.embedding_path}."
            )

        raise NotImplementedError(
            "Embedding store is a placeholder until a trained model is available."
        )

    def get_embedding(self, item_id: str) -> Any:
        if self.embeddings is None:
            raise RuntimeError("Embedding store is not loaded")

        raise NotImplementedError(
            "Embedding retrieval is not implemented yet."
        )
