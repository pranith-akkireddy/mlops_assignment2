import argparse
import os
import sys

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import model_utils  # noqa: E402
import preprocess  # noqa: E402


def train(seed=42, C=1.0, max_iter=1000, curve_points=5):
    mlflow.set_experiment("cats_vs_dogs_baseline")
    with mlflow.start_run():
        data = preprocess.load_processed_data()
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "C": C,
                "max_iter": max_iter,
                "seed": seed,
                "image_size": preprocess.IMAGE_SIZE,
                "model_input_size": preprocess.MODEL_INPUT_SIZE,
                "n_features": int(X_train.shape[1]),
                "n_train": int(X_train.shape[0]),
                "n_val": int(X_val.shape[0]),
                "n_test": int(X_test.shape[0]),
                "classes": ",".join(preprocess.CLASS_NAMES),
            }
        )

        model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=seed, solver="lbfgs", n_jobs=-1
        )
        model.fit(X_train, y_train)

        val_metrics = model_utils.evaluate_model(model, X_val, y_val)
        test_metrics = model_utils.evaluate_model(model, X_test, y_test)
        for split, m in (("val", val_metrics), ("test", test_metrics)):
            mlflow.log_metric(f"{split}_accuracy", m["accuracy"])
            mlflow.log_metric(f"{split}_log_loss", m["log_loss"])
            mlflow.log_metric(f"{split}_roc_auc", m["roc_auc"])
        for label, scores in test_metrics["classification_report"].items():
            if isinstance(scores, dict):
                for k, v in scores.items():
                    mlflow.log_metric(f"test_{label}_{k}", v)

        cm_path = model_utils.log_confusion_matrix(y_test, test_metrics["predictions"])
        roc_path = model_utils.log_roc_curve(y_test, test_metrics["probabilities"][:, 1])

        # Loss and accuracy learning curves (assignment asks for loss curves).
        sizes = np.linspace(0.2, 1.0, curve_points)
        cv = min(3, int(np.bincount(y_train).min()))
        loss_path = accuracy_path = None
        if cv >= 2:
            tr_sizes, tr_loss, va_loss = learning_curve(
                model, X_train, y_train, cv=cv, scoring="neg_log_loss",
                train_sizes=sizes, n_jobs=-1,
            )
            loss_path = model_utils.log_curve(
                tr_sizes, tr_loss, va_loss, metric="log_loss", invert=True
            )
            tr_sizes, tr_acc, va_acc = learning_curve(
                model, X_train, y_train, cv=cv, scoring="accuracy",
                train_sizes=sizes, n_jobs=-1,
            )
            accuracy_path = model_utils.log_curve(
                tr_sizes, tr_acc, va_acc, metric="accuracy"
            )

        model_path = model_utils.save_model(model)
        mlflow.log_artifact(model_path, artifact_path="model")

        print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Test accuracy:       {test_metrics['accuracy']:.4f}")
        print(f"Test ROC-AUC:        {test_metrics['roc_auc']:.4f}")
        print(f"Test log loss:       {test_metrics['log_loss']:.4f}")
        print(f"Model saved to {model_path}")
        for p in (cm_path, roc_path, loss_path, accuracy_path):
            if p:
                print(f"Artifact: {p}")
        return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cats-vs-dogs baseline")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curve-points", type=int, default=5)
    args = parser.parse_args()
    train(
        seed=args.seed,
        C=args.C,
        max_iter=args.max_iter,
        curve_points=args.curve_points,
    )
