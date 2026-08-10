# Digits Baseline MLOps Pipeline

A minimal end-to-end MLOps example covering model development, experiment tracking, containerized inference, CI/CD, and deployment.

## Modules Covered

- **M1** – Data/code versioning with Git + DVC/Git-LFS, baseline `LogisticRegression` model, MLflow experiment tracking.
- **M2** – FastAPI inference service with `/health` and `/predict`, pinned `requirements.txt`, Dockerfile + docker-compose.
- **M3** – Unit tests (`pytest`) and GitHub Actions CI to train, test, build, and push the Docker image.
- **M4** – CD workflow with Docker Compose deployment and post-deploy smoke tests; Kubernetes manifests also provided.
- **M5** – Request/response logging, latency/request-count metrics, and simulated post-deployment feedback collection.

## Project Structure

```text
.
├── app/                      # FastAPI inference service
│   └── main.py
├── src/                      # Training & preprocessing scripts
│   ├── preprocess.py
│   ├── model_utils.py
│   └── train.py
├── scripts/
│   ├── fetch_data.py
│   ├── smoke_test.py
│   └── collect_feedback.py
├── tests/                    # pytest unit tests
│   ├── test_preprocess.py
│   └── test_inference.py
├── k8s/                      # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/        # CI/CD pipelines
│   ├── ci.yml
│   └── cd.yml
├── data/                     # Raw & processed data (ignored by Git/DVC)
├── models/                   # Trained .pkl model (ignored by Git/DVC)
├── artifacts/                # MLflow artifacts
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── .gitattributes
└── .dvc/
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Fetch data & train

```bash
python scripts/fetch_data.py
python src/train.py --seed 42 --C 1.0 --max-iter 1000
```

Training logs parameters, metrics, and artifacts to `mlruns/` via MLflow.

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Run the inference service locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

- `GET  /health` – service health
- `POST /predict` – accepts `{"features": [64 floats]}` and returns `{"label": int, "probabilities": {...}}`
- `GET  /metrics` – request count and average latency

### 5. Build & run Docker container

```bash
docker build -t digits-baseline-api:latest .
docker run -p 8000:8000 digits-baseline-api:latest
```

Or via Docker Compose:

```bash
docker compose up --build -d
python scripts/smoke_test.py
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`): installs dependencies, runs `fetch_data` + `train`, executes `pytest`, builds the Docker image, and pushes to Docker Hub on pushes to `main`.
- **CD** (`.github/workflows/cd.yml`): triggered after a successful `main` CI run, pulls the latest image, deploys via Docker Compose, and runs `scripts/smoke_test.py`.

Set the secrets `DOCKER_USERNAME` and `DOCKER_PASSWORD` in your GitHub repository settings.

## Kubernetes (Optional)

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Post-Deployment Monitoring

After deploying, collect simulated feedback with:

```bash
python scripts/collect_feedback.py
```

This posts requests to the running service, compares predictions to true labels, and writes `feedback_report.json`.

## Notes

- Raw/processed data, trained models, and artifacts are tracked with DVC/Git-LFS and excluded from direct Git commits.
- The baseline model is intentionally simple (`LogisticRegression` on flattened 8×8 digit pixels) to keep the pipeline reproducible and lightweight.
