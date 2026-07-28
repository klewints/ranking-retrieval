from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config import Config
from backend.retrieval.faiss_index import FaissIndex
from backend.retrieval.lightgcn import LightGCNEmbeddings
from backend.retrieval.two_tower import TwoTowerEmbeddings


class CandidateGenerator:
    def retrieve_by_user(self, user_id: str, limit: int = 20) -> List[str]:
        raise NotImplementedError

    def retrieve_by_search(self, search_results: List[Dict[str, Any]], limit: int = 20) -> List[str]:
        raise NotImplementedError


class DefaultCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        tracks: pd.DataFrame,
        faiss_index: FaissIndex,
        two_tower: Optional[TwoTowerEmbeddings] = None,
        lightgcn: Optional[LightGCNEmbeddings] = None,
    ):
        self.tracks = tracks
        self.faiss_index = faiss_index
        self.two_tower = two_tower
        self.lightgcn = lightgcn
        self.popular_track_ids = self._build_popular_tracks()

    def _build_popular_tracks(self) -> List[str]:
        if 'track_id' not in self.tracks.columns:
            return []
        ordered = self.tracks.sort_values('popularity', ascending=False)
        return ordered['track_id'].astype(str).tolist()

    def _normalize_name(self, value: str) -> str:
        return value.strip().lower() if value else ''

    def _parse_genres(self, raw: str) -> List[str]:
        if not raw or raw == 'nan':
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip().lower() for item in parsed if item]
        except (SyntaxError, ValueError):
            pass
        return [token.strip().lower() for token in str(raw).split(',') if token.strip()]

    def _track_ids_by_search(self, name: str, category: str) -> List[str]:
        if not name:
            return []

        normalized = self._normalize_name(name)
        if category == 'track':
            matched = self.tracks[self.tracks['track_display'].str.lower() == normalized]
            if matched.empty:
                matched = self.tracks[self.tracks['track_clean'].str.contains(normalized, case=False, na=False)]
        elif category == 'artist':
            matched = self.tracks[self.tracks['artist_display'].str.lower() == normalized]
            if matched.empty:
                matched = self.tracks[self.tracks['artist_clean'].str.contains(normalized, case=False, na=False)]
        elif category == 'album':
            matched = self.tracks[self.tracks['album_display'].str.lower() == normalized]
            if matched.empty:
                matched = self.tracks[self.tracks['album_clean'].str.contains(normalized, case=False, na=False)]
        elif category == 'genre':
            matched = self.tracks[self.tracks['genres_list'].apply(lambda raw: normalized in self._parse_genres(str(raw)))]
        else:
            matched = self.tracks[self.tracks['track_display'].str.lower().str.contains(normalized, na=False)]

        return matched['track_id'].astype(str).tolist()

    def _query_faiss_by_embeddings(self, embeddings: List[Optional[List[float]]], limit: int) -> List[str]:
        candidate_ids = []
        if not self.faiss_index.is_loaded():
            return candidate_ids

        for embedding in embeddings:
            if embedding is None:
                continue
            indices, _ = self.faiss_index.query(embedding, top_k=limit)
            candidate_ids.extend(self.faiss_index.get_track_ids(indices))
            if len(candidate_ids) >= limit:
                break
        return candidate_ids

    def retrieve_by_user(self, user_id: str, limit: int = 20) -> List[str]:
        candidates: List[str] = []

        if self.two_tower and self.two_tower.has_user(user_id):
            embedding = self.two_tower.get_user_embedding(user_id)
            candidates = self._query_faiss_by_embeddings([embedding], limit)
        elif self.lightgcn and self.lightgcn.has_user(user_id):
            embedding = self.lightgcn.get_user_embedding(user_id)
            candidates = self._query_faiss_by_embeddings([embedding], limit)

        if not candidates:
            candidates = self.popular_track_ids[: limit]

        unique_candidates = []
        for value in candidates + self.popular_track_ids:
            if value not in unique_candidates:
                unique_candidates.append(value)
            if len(unique_candidates) >= limit:
                break
        return unique_candidates

    def retrieve_by_search(self, search_results: List[Dict[str, Any]], limit: int = 20) -> List[str]:
        candidate_ids = []
        if not search_results:
            return self.popular_track_ids[:limit]

        for result in search_results:
            candidate_ids.extend(self._track_ids_by_search(result.get('name', ''), result.get('category', '')))

        candidate_ids = [str(value) for value in candidate_ids if value]
        candidate_ids = list(dict.fromkeys(candidate_ids))

        if self.faiss_index.is_loaded() and candidate_ids:
            embeddings = []
            for track_id in candidate_ids[:5]:
                try:
                    int(track_id)
                    pass
                except ValueError:
                    continue
                item_embedding = None
                if self.two_tower:
                    item_embedding = self.two_tower.get_item_embedding(track_id)
                if item_embedding is None and self.lightgcn:
                    item_embedding = self.lightgcn.get_item_embedding(track_id)
                embeddings.append(item_embedding)
            candidate_ids.extend(self._query_faiss_by_embeddings(embeddings, limit))

        if len(candidate_ids) < limit:
            candidate_ids.extend(self.popular_track_ids)

        unique_candidates = []
        for value in candidate_ids:
            if value not in unique_candidates:
                unique_candidates.append(value)
            if len(unique_candidates) >= limit:
                break
        return unique_candidates
