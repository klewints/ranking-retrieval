"""Train a two-tower retrieval model using user-song interactions."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import Config
from backend.retrieval.two_tower import TwoTowerModel, save_two_tower


class InteractionDataset(Dataset):
    def __init__(self, positives: List[tuple[int, int]], num_items: int):
        self.positives = positives
        self.num_items = num_items

    def __len__(self) -> int:
        return len(self.positives)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        user_idx, item_idx = self.positives[index]
        negative_idx = random.randrange(self.num_items)
        return user_idx, item_idx, negative_idx


def load_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracks_path = processed_dir / 'tracks_cleaned.csv'
    interactions_path = processed_dir / 'user_interactions.csv'
    if not tracks_path.exists() or not interactions_path.exists():
        raise FileNotFoundError('Processed dataset not found. Run scripts/process_data.py first.')
    tracks = pd.read_csv(tracks_path)
    interactions = pd.read_csv(interactions_path)
    return tracks, interactions


def build_indices(interactions: pd.DataFrame, tracks: pd.DataFrame) -> tuple[Dict[str, int], Dict[str, int], List[str]]:
    user_ids = sorted(interactions['user'].astype(str).unique())
    item_ids = sorted(tracks['track_id'].astype(str).unique())
    user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_item = [None] * len(item_ids)
    for item_id, idx in item_to_index.items():
        index_to_item[idx] = item_id
    return user_to_index, item_to_index, index_to_item


def prepare_pairs(interactions: pd.DataFrame, user_to_index: Dict[str, int], item_to_index: Dict[str, int]) -> List[tuple[int, int]]:
    unique_pairs = interactions.groupby(['user', 'track_id'], as_index=False)['playcount'].sum()
    return [
        (user_to_index[str(row.user)], item_to_index[str(row.track_id)])
        for row in unique_pairs.itertuples(index=False)
        if str(row.user) in user_to_index and str(row.track_id) in item_to_index
    ]


def train(
    processed_dir: Path,
    output_path: Path,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    tracks, interactions = load_data(processed_dir)
    user_to_index, item_to_index, index_to_item = build_indices(interactions, tracks)
    positive_pairs = prepare_pairs(interactions, user_to_index, item_to_index)
    if not positive_pairs:
        raise RuntimeError('No interactions available for training Two-Tower model.')

    num_users = len(user_to_index)
    num_items = len(item_to_index)
    dataset = InteractionDataset(positive_pairs, num_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TwoTowerModel(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for user_idx, pos_item_idx, neg_item_idx in loader:
            optimizer.zero_grad()
            pos_scores = model(user_idx, pos_item_idx)
            neg_scores = model(user_idx, neg_item_idx)
            loss = torch.mean(F.softplus(neg_scores - pos_scores))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(user_idx)
        average_loss = total_loss / len(dataset)
        print(f'Epoch {epoch}/{epochs} Two-Tower training loss: {average_loss:.4f}')

    save_two_tower(model, user_to_index, item_to_index, output_path)
    print(f'Two-Tower model saved to {output_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Two-Tower retrieval model')
    parser.add_argument('--processed-dir', default=str(Config.PROCESSED_DATA_DIR), help='Processed data directory')
    parser.add_argument('--output', default=str(Config.TWO_TOWER_MODEL_PATH), help='Output model path')
    parser.add_argument('--embedding-dim', type=int, default=Config.EMBEDDING_DIMENSION, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=6, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        processed_dir=Path(args.processed_dir),
        output_path=Path(args.output),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == '__main__':
    main()
