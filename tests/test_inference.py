import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import model_utils, preprocess


class DummyModel:
    """Always predicts 'dog' with a fixed probability vector."""

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        probs[:, 0] = 0.25
        probs[:, 1] = 0.75
        return probs

    def predict(self, X):
        return np.ones(X.shape[0], dtype=int)


def _jpeg_bytes(size=(256, 256), color=(130, 100, 80)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    n_features = preprocess.MODEL_INPUT_SIZE ** 2 * 3
    scaler = StandardScaler()
    scaler.fit(np.random.default_rng(0).random((20, n_features)))

    scaler_path = tmp_path / "scaler.pkl"
    model_path = tmp_path / "model.pkl"
    model_utils.save_model(scaler, str(scaler_path))
    model_utils.save_model(DummyModel(), str(model_path))

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("SCALER_PATH", str(scaler_path))

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classes"] == ["cat", "dog"]


def test_predict_returns_label_and_probabilities(client):
    resp = client.post(
        "/predict", files={"file": ("pet.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "dog"
    assert body["label_index"] == 1
    assert set(body["probabilities"]) == {"cat", "dog"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_accepts_png_and_odd_dimensions(client):
    buf = io.BytesIO()
    Image.new("RGB", (37, 400), (10, 200, 10)).save(buf, format="PNG")
    resp = client.post(
        "/predict", files={"file": ("pet.png", buf.getvalue(), "image/png")}
    )
    assert resp.status_code == 200
    assert resp.json()["label"] in ("cat", "dog")


def test_predict_rejects_empty_upload(client):
    resp = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert resp.status_code == 400


def test_predict_rejects_non_image(client):
    resp = client.post(
        "/predict", files={"file": ("evil.jpg", b"definitely not an image", "image/jpeg")}
    )
    assert resp.status_code == 400
    assert "decode" in resp.json()["detail"].lower()


def test_predict_requires_file(client):
    assert client.post("/predict").status_code == 422


def test_metrics_counts_requests(client):
    client.get("/health")
    body = client.get("/metrics").json()
    assert body["request_count"] >= 1
    assert body["average_latency_ms"] >= 0.0


def test_evaluate_model_reports_binary_metrics():
    rng = np.random.default_rng(1)
    X = rng.random((20, 4))
    y = np.array([0, 1] * 10)
    metrics = model_utils.evaluate_model(DummyModel(), X, y)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] > 0
    assert "roc_auc" in metrics
    assert set(metrics["classification_report"]) >= {"cat", "dog"}


def test_load_model_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_utils.load_model(str(tmp_path / "absent.pkl"))
