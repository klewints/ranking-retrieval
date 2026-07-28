from backend.config import Config


def test_config_paths_are_defined():
    assert Config.PROCESSED_DATA_DIR.name == "processed"
    assert Config.TRACKS_CLEANED_PATH.exists()
    assert Config.DEFAULT_SEARCH_LIMIT > 0
    assert Config.FAISS_INDEX_PATH.parent == Config.MODEL_DIR


def test_search_thresholds_are_positive():
    assert Config.SEARCH_SCORE_THRESHOLD > 0
    assert Config.CORRECTION_SCORE_THRESHOLD > 0
