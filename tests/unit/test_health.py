from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_info_lists_rag():
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    assert "rag" in response.json()["modules"]
