import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_BASE, "data", "raw")
PROC_DIR = os.path.join(_BASE, "data", "processed")


def fetch_raw_data(seed=42):
    """Load the sklearn digits dataset and persist the raw split."""
    os.makedirs(RAW_DIR, exist_ok=True)
    data = load_digits()
    X, y = data.data, data.target
    np.savez(os.path.join(RAW_DIR, "digits_raw.npz"), X=X, y=y)
    return X, y


def preprocess_data(X, y, test_size=0.2, seed=42):
    """Split and scale the raw data; persist processed arrays and scaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    os.makedirs(PROC_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(PROC_DIR, "scaler.pkl"))
    np.savez(
        os.path.join(PROC_DIR, "digits_processed.npz"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    return X_train, X_test, y_train, y_test, scaler


def load_processed_data():
    path = os.path.join(PROC_DIR, "digits_processed.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("Processed data not found. Run fetch_data.py or train.py first.")
    d = np.load(path, allow_pickle=True)
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"]


def load_scaler(path=None):
    path = path or os.path.join(PROC_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)
