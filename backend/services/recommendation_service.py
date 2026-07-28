from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.config import Config
from backend.retrieval.retrieval_service import RetrievalService
from backend.retrieval.ranking_model import RankingModelWrapper
from backend.retrieval.two_tower import TwoTowerEmbeddings
from backend.retrieval.lightgcn import LightGCNEmbeddings

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(
        self,
        search_service: Any,
        retrieval_service: RetrievalService,
        ranking_model: Optional[RankingModelWrapper] = None,
        two_tower: Optional[TwoTowerEmbeddings] = None,
        lightgcn: Optional[LightGCNEmbeddings] = None,
    ):
        self.search_service = search_service
        self.retrieval_service = retrieval_service
        self.ranking_model = ranking_model
        self.two_tower = two_tower
        self.lightgcn = lightgcn
        self.tracks = self._load_tracks()
        self.track_map = self.tracks.set_index('track_id')
        self.user_embedding_dim = self.two_tower.user_embeddings.shape[1] if self.two_tower is not None else (
            self.lightgcn.user_embeddings.shape[1] if self.lightgcn is not None else 0
        )
        self.popular_tracks = self._build_popular_tracks()

    def _load_tracks(self) -> pd.DataFrame:
        if not Path(Config.TRACKS_CLEANED_PATH).exists():
            raise FileNotFoundError(f"Dataset not found: {Config.TRACKS_CLEANED_PATH}")
        df = pd.read_csv(Config.TRACKS_CLEANED_PATH)
        if 'track_id' in df.columns:
            df['track_id'] = df['track_id'].astype(str)
        return df

    def _build_popular_tracks(self) -> List[str]:
        if 'track_id' not in self.tracks.columns:
            return []
        ordered = self.tracks.sort_values('popularity', ascending=False)
        return ordered['track_id'].astype(str).tolist()

    def _get_embeddings(self, user_id: Optional[str]) -> np.ndarray:
        if user_id and self.two_tower and self.two_tower.has_user(user_id):
            emb = self.two_tower.get_user_embedding(user_id)
            if emb is not None:
                return emb
        if user_id and self.lightgcn and self.lightgcn.has_user(user_id):
            emb = self.lightgcn.get_user_embedding(user_id)
            if emb is not None:
                return emb
        if self.two_tower is not None:
            return np.mean(self.two_tower.user_embeddings, axis=0)
        if self.lightgcn is not None:
            return np.mean(self.lightgcn.user_embeddings, axis=0)
        return np.zeros((Config.EMBEDDING_DIMENSION,), dtype=np.float32)

    def _score_candidates(
        self,
        candidate_ids: List[str],
        user_id: Optional[str],
        search_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        user_embedding = self._get_embeddings(user_id)
        search_names = {item['name'].lower(): item['score'] for item in search_results}
        scored = []
        for track_id in candidate_ids:
            if track_id not in self.track_map.index:
                continue
            row = self.track_map.loc[track_id]
            item_embedding = None
            if self.two_tower is not None:
                item_embedding = self.two_tower.get_item_embedding(track_id)
            if item_embedding is None and self.lightgcn is not None:
                item_embedding = self.lightgcn.get_item_embedding(track_id)
            if item_embedding is None:
                item_embedding = np.zeros_like(user_embedding)

            pop = float(row.get('popularity', 0.0))
            genre_overlap = self._compute_genre_overlap(row, search_names)
            similarity = self._cosine_similarity(user_embedding, item_embedding)
            search_score = max(search_names.get(str(row.get('track_display', '')).lower(), 0.0), 0.0)
            features = self._build_feature_vector(user_embedding, item_embedding, pop, genre_overlap, similarity, search_score)
            score = self._rank_score(features)
            scored.append(
                {
                    'track_id': track_id,
                    'score': float(score),
                    'track': row.get('track', ''),
                    'artist': row.get('artist', ''),
                    'album': row.get('album', ''),
                    'genres': self._parse_genres(row.get('genres_list', '')),  # type: ignore[arg-type]
                    'popularity': pop,
                    'search_relevance': float(search_score),
                }
            )
        scored.sort(key=lambda item: item['score'], reverse=True)
        return scored

    def _build_feature_vector(
        self,
        user_embedding: np.ndarray,
        item_embedding: np.ndarray,
        popularity: float,
        genre_overlap: float,
        similarity: float,
        search_score: float,
    ) -> np.ndarray:
        pop_norm = float(popularity) / 100.0 if popularity is not None else 0.0
        features = np.concatenate(
            [user_embedding.astype(np.float32), item_embedding.astype(np.float32), np.array([pop_norm, genre_overlap, similarity, search_score], dtype=np.float32)]
        )
        return features

    def _rank_score(self, features: np.ndarray) -> float:
        if self.ranking_model is None:
            return float(np.dot(features[: len(features) // 2], features[len(features) // 2 :]))
        if features.ndim == 1:
            features = features.reshape(1, -1)
        scores = self.ranking_model.score(features)
        return float(scores[0])

    def _compute_genre_overlap(self, row: pd.Series, search_names: Dict[str, float]) -> float:
        genres = self._parse_genres(row.get('genres_list', ''))
        if not genres or not search_names:
            return 0.0
        overlap = sum(1 for genre in genres if genre.lower() in search_names)
        return float(overlap) / max(1.0, len(genres))

    def _parse_genres(self, raw: Any) -> List[str]:
        if pd.isna(raw):
            return []
        try:
            parsed = raw
            if isinstance(raw, str):
                parsed = eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if item]
        except Exception:
            return [token.strip() for token in str(raw).split(',') if token.strip()]
        return [str(item).strip() for item in parsed if item]

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        if left is None or right is None:
            return 0.0
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def recommend(
        self,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = Config.RECOMMENDATION_TOP_K,
    ) -> Dict[str, Any]:
        search_results = self.search_service.search(query, limit=Config.SEARCH_MAX_RESULTS) if query else {'corrected_query': '', 'results': []}
        candidate_ids: List[str] = []
        if user_id:
            candidate_ids.extend(self.retrieval_service.retrieve_by_user(user_id, limit=Config.RETRIEVAL_CANDIDATE_LIMIT))
        if query:
            candidate_ids.extend(self.retrieval_service.retrieve_by_search(search_results['results'], limit=Config.RETRIEVAL_CANDIDATE_LIMIT))
        if not candidate_ids:
            candidate_ids = self.popular_tracks[: Config.RETRIEVAL_CANDIDATE_LIMIT]

        candidate_ids = [str(value) for value in candidate_ids if value]
        candidate_ids = list(dict.fromkeys(candidate_ids))
        scored = self._score_candidates(candidate_ids, user_id, search_results['results'])
        top_items = scored[:limit]
        return {
            'user_id': user_id,
            'query': query,
            'corrected_query': search_results.get('corrected_query', ''),
            'results': top_items,
        }

    def similar(self, track_id: str, limit: int = 20) -> Dict[str, Any]:
        if track_id not in self.track_map.index:
            return {'track_id': track_id, 'results': []}
        track_embedding = None
        if self.two_tower is not None:
            track_embedding = self.two_tower.get_item_embedding(track_id)
        if track_embedding is None and self.lightgcn is not None:
            track_embedding = self.lightgcn.get_item_embedding(track_id)
        if track_embedding is None or not self.retrieval_service.faiss_index.is_loaded():
            return {
                'track_id': track_id,
                'results': [
                    {
                        'track_id': tid,
                        'track': self.track_map.loc[tid].get('track', ''),
                        'artist': self.track_map.loc[tid].get('artist', ''),
                        'album': self.track_map.loc[tid].get('album', ''),
                        'genres': self._parse_genres(self.track_map.loc[tid].get('genres_list', '')),
                        'score': 0.0,
                    }
                    for tid in self.popular_tracks[: limit]
                ],
            }

        ids, distances = self.retrieval_service.faiss_index.query(track_embedding, top_k=limit + 1)
        results = []
        for tid, distance in zip(ids, distances):
            if str(tid) == str(track_id):
                continue
            if tid not in self.track_map.index:
                continue
            row = self.track_map.loc[tid]
            results.append(
                {
                    'track_id': tid,
                    'track': row.get('track', ''),
                    'artist': row.get('artist', ''),
                    'album': row.get('album', ''),
                    'genres': self._parse_genres(row.get('genres_list', '')),
                    'score': float(distance),
                }
            )
            if len(results) >= limit:
                break
        return {'track_id': track_id, 'results': results}

    def is_ready(self) -> bool:
        return self.retrieval_service.is_ready() and self.ranking_model is not None
