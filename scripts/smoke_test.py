"""Post-deploy smoke test: health check plus one real prediction.

Exits non-zero on any failure so the CD pipeline fails loudly.
"""

import io
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://localhost:8000")


def _get(path):
    return urllib.request.urlopen(f"{BASE_URL}{path}", timeout=15)


def check_health():
    try:
        resp = _get("/health")
    except urllib.error.URLError as exc:
        print(f"[smoke] FAIL /health unreachable: {exc}")
        sys.exit(1)
    if resp.status != 200:
        print(f"[smoke] FAIL /health status {resp.status}")
        sys.exit(1)
    body = json.loads(resp.read())
    if body.get("status") != "ok" or not body.get("model_loaded"):
        print(f"[smoke] FAIL /health body {body}")
        sys.exit(1)
    print("[smoke] /health OK")


def _sample_image_bytes():
    """Use a real test image when available, else synthesise a valid JPEG."""
    candidate_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw", "PetImages", "Dog",
    )
    if os.path.isdir(candidate_dir):
        for name in sorted(os.listdir(candidate_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(candidate_dir, name)
                if os.path.getsize(path) > 0:
                    with open(path, "rb") as fh:
                        return fh.read(), name
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (120, 90, 70)).save(buf, format="JPEG")
    return buf.getvalue(), "synthetic.jpg"


def _multipart(field, filename, payload):
    boundary = f"----smoke{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; '
            f'filename="{filename}"\r\n'.encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def check_predict():
    payload, filename = _sample_image_bytes()
    body, content_type = _multipart("file", filename, payload)
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as exc:
        print(f"[smoke] FAIL /predict HTTP {exc.code}: {exc.read()!r}")
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"[smoke] FAIL /predict unreachable: {exc}")
        sys.exit(1)
    if resp.status != 200:
        print(f"[smoke] FAIL /predict status {resp.status}")
        sys.exit(1)
    result = json.loads(resp.read())
    for key in ("label", "label_index", "probabilities"):
        if key not in result:
            print(f"[smoke] FAIL /predict missing '{key}': {result}")
            sys.exit(1)
    if result["label"] not in ("cat", "dog"):
        print(f"[smoke] FAIL unexpected label: {result['label']}")
        sys.exit(1)
    print(f"[smoke] /predict OK -> {result['label']} ({filename})")


if __name__ == "__main__":
    print(f"[smoke] target {BASE_URL}")
    check_health()
    check_predict()
    print("All smoke tests passed.")
