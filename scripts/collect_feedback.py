"""Post-deployment model performance tracking.

Sends a batch of held-out images to the running service, compares the returned
labels against ground truth, and writes feedback_report.json.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import preprocess  # noqa: E402

BASE_URL = os.environ.get("FEEDBACK_BASE_URL", "http://localhost:8000")


def _post_image(path):
    with open(path, "rb") as fh:
        payload = fh.read()
    boundary = f"----feedback{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            'Content-Disposition: form-data; name="file"; '
            f'filename="{os.path.basename(path)}"\r\n'.encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def load_holdout():
    """Return the held-out test split recorded at preprocessing time.

    Scoring against images the model trained on would report leaked accuracy,
    so only the test split is ever used here.
    """
    data = preprocess.load_processed_data()
    if "test_paths" not in data:
        raise SystemExit(
            "Processed data predates held-out path tracking. "
            "Re-run scripts/fetch_data.py to regenerate it."
        )
    return data["test_paths"].astype(str), data["y_test"]


def main(n=50, seed=7):
    paths, labels = load_holdout()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)

    true_labels, pred_labels, failures = [], [], 0
    for i in idx:
        try:
            body = _post_image(paths[i])
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"Request failed for {paths[i]}: {exc}")
            failures += 1
            continue
        true_labels.append(int(labels[i]))
        pred_labels.append(int(body["label_index"]))

    if not true_labels:
        print("No successful predictions; is the service running?")
        sys.exit(1)

    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    accuracy = float((true_arr == pred_arr).mean())
    per_class = {
        name: round(float((pred_arr[true_arr == i] == i).mean()), 4)
        for i, name in enumerate(preprocess.CLASS_NAMES)
        if (true_arr == i).any()
    }
    report = {
        "split": "test (held-out)",
        "samples": len(true_labels),
        "failed_requests": failures,
        "accuracy": round(accuracy, 4),
        "per_class_recall": per_class,
    }
    out = os.path.join(_ROOT, "feedback_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Post-deployment accuracy over {len(true_labels)} samples: {accuracy:.4f}")
    print(f"Per-class recall: {per_class}")
    print(f"Report written to {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(n=args.samples, seed=args.seed)
