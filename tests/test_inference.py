import os
import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app


class DummyModel:
    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 10))
        probs[:, 7] = 1.0
        return probs

    def predict(self, X):
        return np.full(X.shape[0], 7)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict(client):
    payload = {"features": list(np.random.rand(64).tolist())}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == 7
    assert len(body["probabilities"]) == 10
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Prepare dummy artifacts
    from sklearn.preprocessing import StandardScaler
    from src import model_utils

    scaler = StandardScaler()
    scaler.fit(np.random.randn(50, 64))
    scaler_path = tmp_path / "scaler.pkl"
    model_utils.save_model(scaler, str(scaler_path))

    model_path = tmp_path / "model.pkl"
    model_utils.save_model(DummyModel(), str(model_path))

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("SCALER_PATH", str(scaler_path))

    with TestClient(app) as c:
        yield c
