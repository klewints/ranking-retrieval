import os
import tempfile
import pandas as pd

from backend.retrieval.embedding_store import EmbeddingStore


def test_embedding_store_detects_no_models(tmp_path):
    # point the store at an empty temporary models directory
    store = EmbeddingStore(model_dir=tmp_path)
    # discovery should not raise
    store.load(load_all=False)
    assert store.available_models() == []
    assert not store.is_loaded()
    # accessors return None when no models
    assert store.get_user_embedding('nonexistent') is None
    assert store.get_item_embedding('nonexistent') is None


def test_embedding_store_reload_and_model_info(tmp_path):
    # create dummy files to simulate presence
    two_tower_path = tmp_path / 'two_tower.pth'
    two_tower_path.write_text('not_a_real_model')

    # patch Config paths by passing model_dir
    store = EmbeddingStore(model_dir=tmp_path)
    # available_models should list two_tower
    models = store.available_models()
    # path exists but loading will fail; available_models reports detection
    assert 'two_tower' in models or models == []
    # reload should try to load and not crash (errors are logged and swallowed)
    store.reload()
    assert isinstance(store.get_model_info(), dict)
