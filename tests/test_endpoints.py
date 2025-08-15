from fastapi.testclient import TestClient
from services.coder_api.app import app

client = TestClient(app)


def test_codes_endpoint():
    r = client.get("/codes?limit=3")
    assert r.status_code == 200
    body = r.json()
    assert "count" in body and body["count"] >= 3
    assert "sample" in body and len(body["sample"]) == 3


def test_predict_safe_empty_note():
    r = client.post("/predict_safe", json={"note_text": "   ", "top_k": 5})
    assert r.status_code == 422
    assert r.json()["detail"] == "note_text is empty"
