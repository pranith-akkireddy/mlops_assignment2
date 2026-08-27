import os

import joblib
import matplotlib
import mlflow
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE, "models", "cats_dogs_model.pkl")
ARTIFACTS_DIR = os.path.join(_BASE, "artifacts")

CLASS_NAMES = ("cat", "dog")


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def evaluate_model(model, X, y):
    """Binary classification metrics for the cats-vs-dogs baseline."""
    preds = model.predict(X)
    probs = model.predict_proba(X)
    positive = probs[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "log_loss": log_loss(y, probs, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, positive),
        "predictions": preds,
        "probabilities": probs,
        "classification_report": classification_report(
            y, preds, target_names=list(CLASS_NAMES), output_dict=True, zero_division=0
        ),
    }


def log_confusion_matrix(y_true, y_pred, label_map=None, filename="confusion_matrix.png"):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    label_map = label_map or list(CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_map, yticklabels=label_map,
    )
    plt.title("Confusion Matrix (Cats vs Dogs)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    path = os.path.join(ARTIFACTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    _log_artifact(path)
    return path


def log_curve(train_sizes, train_scores, val_scores, metric="accuracy", invert=False):
    """Plot a learning curve. Set invert=True for negated loss scorers."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    train_scores = np.asarray(train_scores)
    val_scores = np.asarray(val_scores)
    sign = -1.0 if invert else 1.0
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, sign * train_scores.mean(axis=1), marker="o", label="train")
    plt.plot(train_sizes, sign * val_scores.mean(axis=1), marker="s", label="validation")
    plt.xlabel("Training set size")
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.title(f"Learning curve ({metric})")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(ARTIFACTS_DIR, f"{metric}_curve.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    _log_artifact(path)
    return path


def log_roc_curve(y_true, positive_scores):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, positive_scores)
    auc = roc_auc_score(y_true, positive_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve (dog = positive)")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    _log_artifact(path)
    return path


def _log_artifact(path):
    """Log to MLflow when a run is active; stay usable outside a run."""
    if mlflow.active_run() is not None:
        mlflow.log_artifact(path)
