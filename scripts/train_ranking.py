"""Train a ranking model for candidate reordering."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import Config
from backend.retrieval.ranking_model import RankingModel, save_ranking_model
from backend.retrieval.two_tower import TwoTowerEmbeddings


class RankingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


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
    return user_to_index, item_to_index, item_ids


def prepare_user_genres(interactions: pd.DataFrame, track_genres: Dict[str, List[str]]) -> Dict[str, set[str]]:
    user_genres: Dict[str, set[str]] = {}
    for _, row in interactions.iterrows():
        user = str(row['user'])
        track_id = str(row['track_id'])
        genres = track_genres.get(track_id, [])
        user_genres.setdefault(user, set()).update(g.lower() for g in genres)
    return user_genres


def parse_genres(raw: object) -> List[str]:
    if pd.isna(raw):
        return []
    if isinstance(raw, str):
        try:
            parsed = eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if item]
        except Exception:
            return [token.strip() for token in str(raw).split(',') if token.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if item]
    return [str(raw).strip()]


def build_features(
    two_tower: TwoTowerEmbeddings,
    user_to_index: Dict[str, int],
    item_to_index: Dict[str, int],
    tracks: pd.DataFrame,
    interactions: pd.DataFrame,
    max_samples: int = 25000,
) -> tuple[np.ndarray, np.ndarray]:
    item_features = {str(row.track_id): row for row in tracks.itertuples(index=False)}
    track_genres = {str(row.track_id): parse_genres(row.genres_list) for row in tracks.itertuples(index=False)}
    user_genres = prepare_user_genres(interactions, track_genres)

    positive_pairs = interactions.groupby(['user', 'track_id'], as_index=False)['playcount'].sum()
    samples = []
    labels = []
    all_item_ids = list(item_to_index.keys())
    for _, row in positive_pairs.iterrows():
        user_id = str(row['user'])
        track_id = str(row['track_id'])
        if user_id not in user_to_index or track_id not in item_to_index:
            continue
        user_idx = user_to_index[user_id]
        item_idx = item_to_index[track_id]
        user_emb = two_tower.user_embeddings[user_idx]
        item_emb = two_tower.item_embeddings[item_idx]
        pop = float(item_features[track_id].popularity) if track_id in item_features else 0.0
        genre_overlap = float(len(user_genres.get(user_id, set()) & set(g.lower() for g in track_genres.get(track_id, []))))
        similarity = float(np.dot(user_emb, item_emb) / max(1e-8, np.linalg.norm(user_emb) * np.linalg.norm(item_emb)))
        features = np.concatenate([user_emb, item_emb, np.array([pop / 100.0, genre_overlap, similarity, 0.0], dtype=np.float32)])
        samples.append(features)
        labels.append(1.0)
        for _ in range(1):
            negative_track_id = all_item_ids[random.randrange(len(all_item_ids))]
            if negative_track_id == track_id:
                continue
            negative_item_idx = item_to_index[negative_track_id]
            negative_item_emb = two_tower.item_embeddings[negative_item_idx]
            pop = float(item_features[negative_track_id].popularity) if negative_track_id in item_features else 0.0
            genre_overlap = float(len(user_genres.get(user_id, set()) & set(g.lower() for g in track_genres.get(negative_track_id, []))))
            similarity = float(np.dot(user_emb, negative_item_emb) / max(1e-8, np.linalg.norm(user_emb) * np.linalg.norm(negative_item_emb)))
            negative_features = np.concatenate([user_emb, negative_item_emb, np.array([pop / 100.0, genre_overlap, similarity, 0.0], dtype=np.float32)])
            samples.append(negative_features)
            labels.append(0.0)
        if len(samples) >= max_samples:
            break

    if not samples:
        raise RuntimeError('No ranking training samples could be built.')

    return np.vstack(samples), np.array(labels, dtype=np.float32)


def train(
    processed_dir: Path,
    source_model_path: Path,
    output_path: Path,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    tracks, interactions = load_data(processed_dir)
    user_to_index, item_to_index, item_ids = build_indices(interactions, tracks)
    two_tower = TwoTowerEmbeddings.load(source_model_path)

    features, labels = build_features(two_tower, user_to_index, item_to_index, tracks, interactions)
    dataset = RankingDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = features.shape[1]
    model = RankingModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_features.size(0)
        print(f'Epoch {epoch}/{epochs} ranking training loss: {total_loss / len(dataset):.4f}')

    save_ranking_model(model, input_dim, output_path)
    print(f'Ranking model saved to {output_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train ranking model')
    parser.add_argument('--processed-dir', default=str(Config.PROCESSED_DATA_DIR), help='Processed data directory')
    parser.add_argument('--source', default=str(Config.TWO_TOWER_MODEL_PATH), help='Source Two-Tower model path')
    parser.add_argument('--output', default=str(Config.RANKING_MODEL_PATH), help='Output model path')
    parser.add_argument('--embedding-dim', type=int, default=Config.EMBEDDING_DIMENSION, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=6, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        processed_dir=Path(args.processed_dir),
        source_model_path=Path(args.source),
        output_path=Path(args.output),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == '__main__':
    main()
