import os
import argparse
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

import preprocess
import model_utils


def train(test_size=0.2, seed=42, C=1.0, max_iter=1000):
    mlflow.set_experiment("digits_baseline")
    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("seed", seed)
        mlflow.log_param("test_size", test_size)

        X, y = preprocess.fetch_raw_data(seed=seed)
        mlflow.log_param("n_samples", int(X.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))
        mlflow.log_param("n_classes", int(len(np.unique(y))))

        X_train, X_test, y_train, y_test, scaler = preprocess.preprocess_data(
            X, y, test_size=test_size, seed=seed
        )

        model = LogisticRegression(
            max_iter=max_iter, C=C, random_state=seed, n_jobs=5, solver="lbfgs"
        )
        model.fit(X_train, y_train)

        metrics = model_utils.evaluate_model(model, X_test, y_test)
        mlflow.log_metric("accuracy", metrics["accuracy"])

        for label, scores in metrics["classification_report"].items():
            if isinstance(scores, dict):
                for k, v in scores.items():
                    mlflow.log_metric(f"{label}_{k}", v)

        preds = metrics["predictions"]
        cm_path = model_utils.log_confusion_matrix(
            y_test, preds, label_map=[str(i) for i in range(10)]
        )

        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=seed,
            n_jobs=5,
        )
        curve_path = model_utils.log_learning_curve(
            train_sizes, train_scores, val_scores, metric="accuracy"
        )

        model_path = model_utils.save_model(model)
        mlflow.log_artifact(model_path, artifact_path="model")

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Model saved to {model_path}")
        print(f"Confusion matrix: {cm_path}")
        print(f"Learning curve: {curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train(test_size=args.test_size, seed=args.seed, C=args.C, max_iter=args.max_iter)
