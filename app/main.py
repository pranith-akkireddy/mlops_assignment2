import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

# Make src imports available without package install
_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model_utils import load_model  # noqa: E402
from preprocess import (  # noqa: E402
    CLASS_NAMES,
    image_to_features,
    load_image,
    load_scaler,
)

_ROOT = os.path.dirname(os.path.dirname(__file__))
_DEFAULT_MODEL_PATH = os.path.join(_ROOT, "models", "cats_dogs_model.pkl")
_DEFAULT_SCALER_PATH = os.path.join(_ROOT, "data", "processed", "scaler.pkl")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

logger = logging.getLogger("uvicorn")
request_count = 0
total_latency = 0.0


class PredictResponse(BaseModel):
    label: str
    label_index: int
    probabilities: Dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
    scaler_path = os.environ.get("SCALER_PATH", _DEFAULT_SCALER_PATH)
    app.state.model = load_model(model_path)
    app.state.scaler = load_scaler(scaler_path)
    yield


app = FastAPI(
    title="Cats vs Dogs Inference API",
    description="Binary image classification service for a pet adoption platform.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global request_count, total_latency
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    request_count += 1
    total_latency += latency
    # Only request metadata is logged -- never the uploaded image payload.
    logger.info(
        "method=%s path=%s status_code=%s latency_ms=%.2f total_requests=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency * 1000,
        request_count,
    )
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "classes": list(CLASS_NAMES),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    try:
        img = load_image(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image: {exc}"
        ) from exc

    try:
        features = image_to_features(img).reshape(1, -1)
        scaled = app.state.scaler.transform(features)
        model = app.state.model
        probs = model.predict_proba(scaled)[0]
        index = int(model.predict(scaled)[0])
        return {
            "label": CLASS_NAMES[index],
            "label_index": index,
            "probabilities": {
                name: float(p) for name, p in zip(CLASS_NAMES, probs)
            },
        }
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
async def metrics():
    avg = (total_latency / request_count) * 1000 if request_count else 0.0
    return {
        "request_count": request_count,
        "average_latency_ms": round(avg, 2),
    }
