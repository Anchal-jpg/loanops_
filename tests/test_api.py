from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_invalid_input():
    assert client.post("/predict", json={"ApplicantIncome": -1}).status_code == 422
