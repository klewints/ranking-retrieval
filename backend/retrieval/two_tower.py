from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from backend.config import Config


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = Config.EMBEDDING_DIMENSION):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.embedding_dim = embedding_dim

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        return (user_emb * item_emb).sum(dim=-1)

    def get_user_embeddings(self) -> torch.Tensor:
        return self.user_embedding.weight.detach().cpu()

    def get_item_embeddings(self) -> torch.Tensor:
        return self.item_embedding.weight.detach().cpu()


@dataclass
class TwoTowerEmbeddings:
    user_to_index: Dict[str, int]
    item_to_index: Dict[str, int]
    index_to_item: List[str]
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    @classmethod
    def load(cls, path: Path | str) -> "TwoTowerEmbeddings":
        data = torch.load(str(path), map_location="cpu")
        model_state = data["model_state"]
        user_to_index = data["user_to_index"]
        item_to_index = data["item_to_index"]
        index_to_item = data["index_to_item"]
        num_users = data["num_users"]
        num_items = data["num_items"]
        embedding_dim = data["embedding_dim"]

        model = TwoTowerModel(num_users, num_items, embedding_dim)
        model.load_state_dict(model_state)
        return cls(
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            index_to_item=index_to_item,
            user_embeddings=model.get_user_embeddings().numpy(),
            item_embeddings=model.get_item_embeddings().numpy(),
        )

    def has_user(self, user_id: str) -> bool:
        return user_id in self.user_to_index

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        idx = self.user_to_index.get(user_id)
        if idx is None:
            return None
        return self.user_embeddings[idx]

    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        idx = self.item_to_index.get(item_id)
        if idx is None:
            return None
        return self.item_embeddings[idx]

    def get_populated_item_ids(self) -> List[str]:
        return list(self.index_to_item)


def save_two_tower(
    model: TwoTowerModel,
    user_to_index: Dict[str, int],
    item_to_index: Dict[str, int],
    path: Path | str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index_to_item = [None] * len(item_to_index)
    for item_id, idx in item_to_index.items():
        index_to_item[idx] = item_id

    state = {
        "model_state": model.state_dict(),
        "user_to_index": user_to_index,
        "item_to_index": item_to_index,
        "index_to_item": index_to_item,
        "num_users": model.user_embedding.num_embeddings,
        "num_items": model.item_embedding.num_embeddings,
        "embedding_dim": model.embedding_dim,
    }
    torch.save(state, str(path))
