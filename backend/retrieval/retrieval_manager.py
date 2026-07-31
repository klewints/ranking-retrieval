from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config import Config
from backend.retrieval.embedding_store import EmbeddingStore
from backend.retrieval.faiss_index import FaissIndex
from backend.retrieval.candidate_generator import DefaultCandidateGenerator

logger = logging.getLogger(__name__)


class RetrievalManager:
    """Orchestrates EmbeddingStore, FaissIndex and CandidateGenerator to provide a simple retrieval API.

    Responsibilities:
    - initialize and coordinate embeddings and the faiss index
    - expose retrieval methods used by RecommendationService
    - provide status and diagnostics
    """

    def __init__(
        self,
        tracks_df: Optional[pd.DataFrame] = None,
        embedding_store: Optional[EmbeddingStore] = None,
        faiss_index: Optional[FaissIndex] = None,
    ):
        self.tracks_df = tracks_df
        self.embedding_store = embedding_store or EmbeddingStore()
        self.faiss_index = faiss_index or FaissIndex()
        self.candidate_generator: Optional[DefaultCandidateGenerator] = None

    def load(self, load_models: bool = True) -> None:
        # Load embeddings (non-fatal)
        try:
            if load_models and Config.AUTO_LOAD_MODELS:
                self.embedding_store.load(load_all=True)
            else:
                self.embedding_store.load(load_all=False)
        except Exception as exc:
            logger.warning("RetrievalManager: embedding store failed to initialize: %s", exc)

        # Load faiss index (non-fatal)
        try:
            if Config.ENABLE_FAISS and Config.AUTO_LOAD_FAISS:
                self.faiss_index.load()
            else:
                # still attempt a light load for diagnostics
                self.faiss_index.load()
        except Exception as exc:
            logger.warning("RetrievalManager: FAISS failed to initialize: %s", exc)

        # Load tracks
        if self.tracks_df is None:
            try:
                self.tracks_df = pd.read_csv(Config.TRACKS_CLEANED_PATH)
            except Exception as exc:
                logger.warning("RetrievalManager: could not load tracks dataframe: %s", exc)
                self.tracks_df = pd.DataFrame()

        # Build candidate generator using embedding_store
        try:
            self.candidate_generator = DefaultCandidateGenerator(
                self.tracks_df, self.faiss_index, embedding_store=self.embedding_store
            )
        except Exception as exc:
            logger.warning("RetrievalManager: failed to initialize CandidateGenerator: %s", exc)
            self.candidate_generator = None

        # validate compatibility between faiss index and item embeddings if both present
        compatible = True
        try:
            faiss_info = self.faiss_index.get_index_info() if hasattr(self.faiss_index, 'get_index_info') else {}
            emb_mat = self.embedding_store.get_item_matrix() if self.embedding_store else None
            faiss_dim = faiss_info.get('dim') if isinstance(faiss_info, dict) else None
            emb_dim = int(emb_mat.shape[1]) if (emb_mat is not None and getattr(emb_mat, 'ndim', 0) == 2) else None
            if faiss_dim is not None and emb_dim is not None and faiss_dim != emb_dim:
                compatible = False
                logger.warning(
                    "FAISS index dimension (%s) does not match embedding dimension (%s). Disabling FAISS.",
                    faiss_dim,
                    emb_dim,
                )
                # disable FAISS to avoid runtime errors
                try:
                    self.faiss_index.index = None
                    self.faiss_index.track_ids = []
                except Exception:
                    pass
        except Exception:
            logger.exception("Error while validating FAISS <-> embeddings compatibility")

        logger.info("RetrievalManager loaded. FAISS loaded=%s, embeddings_loaded=%s, compatible=%s", self.faiss_index.is_loaded(), self.embedding_store.is_loaded(), compatible)

    def is_ready(self) -> bool:
        return bool(self.faiss_index.is_loaded())

    def retrieve_by_user(self, user_id: str, limit: int = Config.RETRIEVAL_TOP_K) -> List[str]:
        if self.candidate_generator is None:
            raise RuntimeError("CandidateGenerator not initialized")
        return self.candidate_generator.retrieve_by_user(user_id, limit=limit)

    def retrieve_by_search(self, search_results: List[Dict[str, Any]], limit: int = Config.RETRIEVAL_TOP_K) -> List[str]:
        if self.candidate_generator is None:
            raise RuntimeError("CandidateGenerator not initialized")
        return self.candidate_generator.retrieve_by_search(search_results, limit=limit)

    def similar_items(self, track_id: str, limit: int = Config.RETRIEVAL_TOP_K) -> List[str]:
        # Uses FAISS + embeddings to find similar tracks; fallback to popular
        if self.faiss_index.is_loaded():
            emb = self.embedding_store.get_item_embedding(track_id) if self.embedding_store else None
            if emb is not None:
                try:
                    ids, _ = self.faiss_index.query(emb, top_k=limit + 1)
                    # filter out the query id
                    return [tid for tid in ids if str(tid) != str(track_id)][:limit]
                except Exception as exc:
                    logger.warning("RetrievalManager.similar_items: faiss query failed: %s", exc)
        # fallback
        if 'track_id' in self.tracks_df.columns:
            return self.tracks_df.sort_values('popularity', ascending=False)['track_id'].astype(str).tolist()[:limit]
        return []

    def get_status(self) -> Dict[str, Any]:
        faiss_info = self.faiss_index.get_index_info() if hasattr(self.faiss_index, 'get_index_info') else {'loaded': False}
        embeddings_info = self.embedding_store.get_model_info() if self.embedding_store else {}
        # compatibility indicator
        compatible = True
        try:
            faiss_dim = faiss_info.get('dim') if isinstance(faiss_info, dict) else None
            emb_mat = self.embedding_store.get_item_matrix() if self.embedding_store else None
            emb_dim = int(emb_mat.shape[1]) if (emb_mat is not None and getattr(emb_mat, 'ndim', 0) == 2) else None
            if faiss_dim is not None and emb_dim is not None and faiss_dim != emb_dim:
                compatible = False
        except Exception:
            compatible = False
        return {
            'faiss': faiss_info,
            'embeddings': embeddings_info,
            'candidate_generator_initialized': self.candidate_generator is not None,
            'compatible': compatible,
        }
