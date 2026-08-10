import json
import urllib.request
import urllib.error
import sys

BASE_URL = "http://localhost:8000"


def check_health():
    try:
        resp = urllib.request.urlopen(f"{BASE_URL}/health", timeout=10)
    except urllib.error.URLError as exc:
        print(f"Health check failed: {exc}")
        sys.exit(1)
    assert resp.status == 200, f"Expected 200, got {resp.status}"
    body = json.loads(resp.read())
    assert body.get("status") == "ok", f"Unexpected body: {body}"
    print("[smoke] /health OK")


def check_predict():
    payload = json.dumps({"features": [0.0] * 64}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
    except urllib.error.URLError as exc:
        print(f"Prediction failed: {exc}")
        sys.exit(1)
    assert resp.status == 200, f"Expected 200, got {resp.status}"
    body = json.loads(resp.read())
    assert "label" in body, f"Missing label in response: {body}"
    assert "probabilities" in body, f"Missing probabilities in response: {body}"
    print("[smoke] /predict OK")


if __name__ == "__main__":
    check_health()
    check_predict()
    print("All smoke tests passed.")
