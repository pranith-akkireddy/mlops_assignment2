import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

# Make src imports available without package install
_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model_utils import load_model  # noqa: E402
from preprocess import load_scaler  # noqa: E402

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "baseline_model.pkl"
)
_DEFAULT_SCALER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "processed", "scaler.pkl"
)

logger = logging.getLogger("uvicorn")
request_count = 0
total_latency = 0.0


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=64, max_length=64)


class PredictResponse(BaseModel):
    label: int
    probabilities: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
    scaler_path = os.environ.get("SCALER_PATH", _DEFAULT_SCALER_PATH)
    app.state.model = load_model(model_path)
    app.state.scaler = load_scaler(scaler_path)
    yield


app = FastAPI(
    title="Digits Baseline Inference API",
    version="0.1.0",
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
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        arr = np.asarray(req.features).reshape(1, -1)
        scaled = app.state.scaler.transform(arr)
        model = app.state.model
        probs = model.predict_proba(scaled)[0]
        label = int(model.predict(scaled)[0])
        return {
            "label": label,
            "probabilities": {str(i): float(p) for i, p in enumerate(probs)},
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
