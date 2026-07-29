import pandas as pd

from backend.retrieval.retrieval_manager import RetrievalManager
from backend.retrieval.embedding_store import EmbeddingStore
from backend.retrieval.faiss_index import FaissIndex


def make_tracks_df():
    return pd.DataFrame([
        {'track_id': '1', 'track_display': 'Song A', 'popularity': 10},
        {'track_id': '2', 'track_display': 'Song B', 'popularity': 50},
        {'track_id': '3', 'track_display': 'Song C', 'popularity': 30},
    ])


def test_retrieval_manager_fallbacks(tmp_path):
    tracks = make_tracks_df()
    embedding_store = EmbeddingStore(model_dir=tmp_path)
    faiss = FaissIndex(index_path=tmp_path / 'missing_index.bin', track_ids_path=tmp_path / 'missing_ids.pkl')

    manager = RetrievalManager(tracks_df=tracks, embedding_store=embedding_store, faiss_index=faiss)
    manager.load()

    # FAISS not loaded -> is_ready False
    assert not manager.is_ready()

    # retrieve_by_user should fall back to popular tracks
    user_candidates = manager.retrieve_by_user('unknown_user', limit=2)
    assert isinstance(user_candidates, list)
    assert len(user_candidates) == 2

    # retrieve_by_search should return matching track(s) for search results
    search_results = [{'name': 'Song B', 'category': 'track'}]
    search_candidates = manager.retrieve_by_search(search_results, limit=5)
    assert '2' in search_candidates

    # similar_items should fallback to popular tracks
    similar = manager.similar_items('1', limit=2)
    assert isinstance(similar, list)
    assert len(similar) == 2

    status = manager.get_status()
    assert 'faiss' in status and 'embeddings' in status
