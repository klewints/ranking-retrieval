from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class RankingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class RankingModelWrapper:
    model: RankingModel
    input_dim: int

    def score(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(features.astype(np.float32))
            scores = self.model(tensor)
        return scores.cpu().numpy()

    @classmethod
    def load(cls, path: Path | str) -> "RankingModelWrapper":
        data = torch.load(str(path), map_location="cpu")
        input_dim = data["input_dim"]
        model = RankingModel(input_dim)
        model.load_state_dict(data["model_state"])
        model.eval()
        return cls(model=model, input_dim=input_dim)


def save_ranking_model(model: RankingModel, input_dim: int, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": input_dim,
            "model_state": model.state_dict(),
        },
        str(path),
    )
