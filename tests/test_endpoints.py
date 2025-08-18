from fastapi.testclient import TestClient
from services.coder_api.app import app

client = TestClient(app)


def test_codes_endpoint():
    r = client.get("/codes?limit=3")
    assert r.status_code == 200
    body = r.json()
    assert "count" in body and body["count"] >= 3
    assert "sample" in body and len(body["sample"]) == 3


def test_predict_empty_note_returns_422():
    r = client.post("/predict", json={"note_text": "   ", "top_k": 5})
    assert r.status_code == 422
    body = r.json()
    # Pydantic v2 validation style: detail is a list of error dicts with "msg"
    assert isinstance(body.get("detail"), list) and len(body["detail"]) >= 1
    msgs = [e.get("msg", "") for e in body["detail"] if isinstance(e, dict)]
    # Message typically like "Value error, note_text is empty"
    assert any("note_text is empty" in m for m in msgs)
