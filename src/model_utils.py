import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import mlflow

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE, "models", "baseline_model.pkl")
ARTIFACTS_DIR = os.path.join(_BASE, "artifacts")


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "predictions": preds,
        "probabilities": probs,
        "classification_report": report,
    }


def log_confusion_matrix(y_true, y_pred, label_map=None):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_map,
        yticklabels=label_map,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    mlflow.log_artifact(path)
    return path


def log_learning_curve(train_sizes, train_scores, val_scores, metric="accuracy"):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    train_scores = np.asarray(train_scores)
    val_scores = np.asarray(val_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="train")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="validation")
    plt.xlabel("Training set size")
    plt.ylabel(metric.capitalize())
    plt.title(f"Learning curve ({metric})")
    plt.legend()
    path = os.path.join(ARTIFACTS_DIR, f"{metric}_curve.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    mlflow.log_artifact(path)
    return path
