from fastapi.testclient import TestClient

from backend.api.main import app


def test_search_endpoint_returns_real_catalog_results():
    client = TestClient(app)
    response = client.get('/search', params={'q': 'taylor'})

    assert response.status_code == 200
    body = response.json()
    assert 'results' in body
    assert isinstance(body['results'], list)
    assert len(body['results']) <= 10
    assert body['results']
    assert body['results'][0]['title']
    assert 'score' in body['results'][0]
