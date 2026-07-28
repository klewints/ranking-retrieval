from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from backend.config import Config


class LightGCNModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = Config.EMBEDDING_DIMENSION,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.embedding_dim = embedding_dim

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings = [all_embeddings]

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(adjacency, all_embeddings)
            embeddings.append(all_embeddings)

        all_embeddings = torch.stack(embeddings, dim=0).mean(dim=0)
        return all_embeddings[: self.num_users], all_embeddings[self.num_users :]


@dataclass
class LightGCNEmbeddings:
    user_to_index: Dict[str, int]
    item_to_index: Dict[str, int]
    index_to_item: List[str]
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    @classmethod
    def load(cls, path: Path | str) -> "LightGCNEmbeddings":
        data = torch.load(str(path), map_location="cpu")
        user_to_index = data["user_to_index"]
        item_to_index = data["item_to_index"]
        index_to_item = data["index_to_item"]
        user_embeddings = data["user_embeddings"].numpy()
        item_embeddings = data["item_embeddings"].numpy()
        return cls(
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            index_to_item=index_to_item,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
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


def build_normalized_graph(
    user_indices: List[int],
    item_indices: List[int],
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    total_nodes = num_users + num_items
    row = []
    col = []
    values = []

    for user_idx, item_idx in zip(user_indices, item_indices):
        item_node = num_users + item_idx
        row.extend([user_idx, item_node])
        col.extend([item_node, user_idx])
        values.extend([1.0, 1.0])

    if not row:
        indices = torch.empty((2, 0), dtype=torch.int64)
        values_tensor = torch.empty((0,), dtype=torch.float32)
    else:
        indices = torch.tensor([row, col], dtype=torch.int64)
        values_tensor = torch.tensor(values, dtype=torch.float32)

    adjacency = torch.sparse_coo_tensor(
        indices,
        values_tensor,
        (total_nodes, total_nodes),
        dtype=torch.float32,
    )

    deg = torch.sparse.sum(adjacency, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    values_norm = deg_inv_sqrt[row] * values_tensor * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(indices, values_norm, (total_nodes, total_nodes), dtype=torch.float32)


def save_lightgcn(
    user_to_index: Dict[str, int],
    item_to_index: Dict[str, int],
    index_to_item: List[str],
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    path: Path | str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
            "index_to_item": index_to_item,
            "user_embeddings": user_embeddings.detach().cpu(),
            "item_embeddings": item_embeddings.detach().cpu(),
        },
        str(path),
    )
