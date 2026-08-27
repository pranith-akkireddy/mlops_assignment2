FROM python:3.11-slim

WORKDIR /app

# Optional escape hatch for networks with a TLS-inspecting proxy, e.g.
#   podman build --build-arg PIP_EXTRA_ARGS="--trusted-host pypi.org --trusted-host files.pythonhosted.org" .
# Left empty by default so certificate verification stays enabled.
ARG PIP_EXTRA_ARGS=""

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip ${PIP_EXTRA_ARGS} \
    && pip install --no-cache-dir ${PIP_EXTRA_ARGS} -r requirements.txt

COPY src ./src
COPY app ./app
COPY models/cats_dogs_model.pkl ./models/cats_dogs_model.pkl
COPY data/processed/scaler.pkl ./data/processed/scaler.pkl

ENV MODEL_PATH=/app/models/cats_dogs_model.pkl \
    SCALER_PATH=/app/data/processed/scaler.pkl \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Drop root privileges for the service process.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
