from fastapi.testclient import TestClient
from services.coder_api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_has_candidates():
    note = "Assessment: Type 2 diabetes and essential hypertension. Denies pneumonia."
    r = client.post("/predict", json={"note_text": note, "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "candidates" in data and len(data["candidates"]) >= 1
