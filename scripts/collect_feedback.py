import json
import urllib.request
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

BASE_URL = "http://localhost:8000"


def main(n=50):
    data = load_digits()
    indices = np.random.default_rng(42).choice(len(data.data), n, replace=False)
    true_labels = []
    pred_labels = []
    for idx in indices:
        features = data.data[idx].tolist()
        true = int(data.target[idx])
        req = urllib.request.Request(
            f"{BASE_URL}/predict",
            data=json.dumps({"features": features}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        body = json.loads(resp.read())
        pred = int(body["label"])
        true_labels.append(true)
        pred_labels.append(pred)
    acc = accuracy_score(true_labels, pred_labels)
    report = {"samples": n, "accuracy": round(acc, 4)}
    print(f"Simulated feedback accuracy over {n} samples: {acc:.4f}")
    with open("feedback_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
