import os
import pytest
import numpy as np
from src import preprocess


def test_fetch_raw_data(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(preprocess, "PROC_DIR", str(tmp_path / "processed"))
    X, y = preprocess.fetch_raw_data(seed=42)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 64
    assert y.ndim == 1
    assert os.path.exists(os.path.join(str(tmp_path), "digits_raw.npz"))


def test_preprocess_data(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(preprocess, "PROC_DIR", str(tmp_path / "processed"))
    X = np.random.rand(100, 64)
    y = np.tile(np.arange(10), 10)
    X_train, X_test, y_train, y_test, scaler = preprocess.preprocess_data(
        X, y, test_size=0.2, seed=42
    )
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert X_train.shape[1] == 64
    assert os.path.exists(os.path.join(str(tmp_path / "processed"), "scaler.pkl"))
    assert hasattr(scaler, "scale_")
