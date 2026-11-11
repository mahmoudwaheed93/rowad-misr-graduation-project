from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import rag as rag_route

client = TestClient(app)


def test_invalid_department_rejected():
    response = client.post("/api/v1/rag/ask", json={"question":"Leave policy?", "department":"all"})
    assert response.status_code == 422


def test_ask_route_with_mocked_retrieval(monkeypatch):
    monkeypatch.setattr(rag_route, "retrieve", lambda question, department, top_k: [])
    monkeypatch.setattr(rag_route, "generate_answer", lambda q, evidence: ("Not indexed", "no_evidence"))
    response = client.post("/api/v1/rag/ask", json={"question":"Leave policy?", "department":"hr"})
    assert response.status_code == 200
    assert response.json()["mode"] == "no_evidence"
