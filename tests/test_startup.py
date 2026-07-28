from fastapi.testclient import TestClient

from backend.api.main import app


def test_api_startup_loads_services():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"

        search_service = client.app.state.search_service
        assert search_service is not None
        assert search_service.engine.index.categories()

        retrieval_service = client.app.state.retrieval_service
        assert retrieval_service is not None
        assert isinstance(retrieval_service.is_ready(), bool)
