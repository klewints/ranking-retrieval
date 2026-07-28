import pytest

from backend.retrieval.faiss_index import FaissIndex
from backend.retrieval.retrieval_service import RetrievalService
from backend.config import Config


def test_faiss_index_load_raises_file_not_found():
    index = FaissIndex(index_path=Config.FAISS_INDEX_PATH)

    with pytest.raises((FileNotFoundError, RuntimeError)):
        index.load()


def test_retrieval_service_reports_unavailable():
    service = RetrievalService()
    assert not service.is_ready()

    with pytest.raises(RuntimeError, match="Retrieval models are unavailable"):
        service.retrieve_by_user("user-123")

    with pytest.raises(RuntimeError, match="Retrieval models are unavailable"):
        service.retrieve_by_search([{"name": "Taylor Swift", "category": "artist"}])
