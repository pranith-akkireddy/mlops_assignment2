# Cats vs Dogs — End-to-End MLOps Pipeline

Binary image classification (cats vs dogs) for a pet adoption platform, wired
end to end: data/code versioning, experiment tracking, a containerised
inference service, CI/CD, deployment manifests, and post-deployment monitoring.

## Modules

| Module | Scope | Where |
|---|---|---|
| **M1** | Git + DVC versioning, baseline model, MLflow tracking | `dvc.yaml`, `src/` |
| **M2** | FastAPI service, pinned deps, Dockerfile | `app/`, `requirements.txt`, `Dockerfile` |
| **M3** | pytest unit tests + GitHub Actions CI, image push | `tests/`, `.github/workflows/ci.yml` |
| **M4** | CD with Compose deploy, k8s manifests, smoke tests | `.github/workflows/cd.yml`, `k8s/`, `scripts/smoke_test.py` |
| **M5** | Request logging, latency/count metrics, feedback loop | `app/main.py`, `scripts/collect_feedback.py` |

## Dataset & preprocessing

Kaggle Cats vs Dogs (25k images, 2 classes). Images are decoded to the
canonical **224×224 RGB** representation and split **80/10/10**
train/validation/test with stratification. **Augmentation** (horizontal flip,
±15° rotation, brightness/contrast jitter) is applied to the training split
only, so validation and test remain clean.

> **Design note.** The baseline is a `LogisticRegression` (explicitly permitted
> by the assignment as "logistic regression on flattened pixels"). A flattened
> 224×224×3 image is 150,528 features, which is intractable for a linear model
> at this sample size, so after the mandated 224×224 decode each image is
> downscaled to `MODEL_INPUT_SIZE` (default 32 → 3,072 features) before
> flattening. Both stages are explicit in `src/preprocess.py` and
> `MODEL_INPUT_SIZE` is configurable. The API applies the identical two-stage
> transform at inference time, so training and serving cannot drift.

The corpus contains a few truncated/zero-byte JPEGs; `discover_images()`
verifies every file and skips the bad ones.

## Getting the data

The archive is ~800 MB. `scripts/fetch_data.py` resolves it in this order:

1. `--zip /path/to/kagglecatsanddogs_5340.zip` (or `DATA_ZIP` env var)
2. an already-extracted `data/raw/PetImages/` directory
3. any `kagglecatsanddogs*.zip` in `data/raw/`
4. download from `DATA_URL` (Microsoft's public mirror of the Kaggle corpus)

```bash
# Recommended: drop the zip in data/raw/ yourself, then
python scripts/fetch_data.py --max-per-class 3000

# Or point at it explicitly
python scripts/fetch_data.py --zip "D:/downloads/kagglecatsanddogs_5340.zip"
```

`--max-per-class` caps images per class (default 3000, so 6000 total) to keep
the linear baseline and the processed `.npz` a sane size.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python scripts/fetch_data.py     # fetch + preprocess -> data/processed/
python src/train.py              # train + log to MLflow -> models/
pytest tests/ -v
uvicorn app.main:app --port 8000
```

Or run the whole pipeline through DVC:

```bash
dvc repro          # runs fetch -> train, tracking outputs
dvc push           # publish artifacts to the configured remote
```

On Windows, if the checkout sits in a deeply nested directory (e.g. under
OneDrive), DVC's cache paths can exceed the 260-character `MAX_PATH` limit and
`dvc repro` fails writing to `.dvc/cache`. Point the cache somewhere short —
`--local` keeps this out of version control:

```bash
dvc cache dir --local C:/dvc_cache
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + whether the model loaded |
| `/predict` | POST | `multipart/form-data` image upload → label + class probabilities |
| `/metrics` | GET | Request count and average latency |

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -F "file=@data/raw/PetImages/Dog/1.jpg"
# {"label":"dog","label_index":1,"probabilities":{"cat":0.18,"dog":0.82}}

curl http://localhost:8000/metrics
```

Interactive docs (handy for the demo recording): <http://localhost:8000/docs>

## Container

The container runtime is **Podman** (daemonless and rootless). The build file
is still named `Dockerfile`, which Podman reads natively.

```bash
podman build --format docker -t cats-dogs-api:latest .
podman run -p 8000:8000 cats-dogs-api:latest
python scripts/smoke_test.py
```

`--format docker` matters: Podman defaults to the OCI image format, which has
no `HEALTHCHECK` field and silently drops the one declared in the `Dockerfile`.

Docker is a drop-in substitute if you prefer it — the same `Dockerfile` and
`docker-compose.yml` work with `docker build` / `docker run`.

Behind a TLS-inspecting corporate proxy, pip may fail to verify PyPI's
certificate. Certificate verification is **on** by default; opt out only for a
local build:

```bash
podman build --build-arg \
  PIP_EXTRA_ARGS="--trusted-host pypi.org --trusted-host files.pythonhosted.org" \
  -t cats-dogs-api:latest .
```

The cleaner fix is to install your corporate root CA into the image and set
`PIP_CERT`.

Compose — Podman has no built-in Compose engine, so this uses
[`podman-compose`](https://github.com/containers/podman-compose):

```bash
pip install podman-compose==1.6.0

podman-compose up --build -d                              # local build
IMAGE=ghcr.io/<owner>/cats-dogs-api:latest \
  podman-compose up -d --no-build                         # registry image

podman-compose logs
podman-compose down
```

## CI/CD

**CI** (`.github/workflows/ci.yml`) — on every push/PR: checkout, install deps,
fetch data (archive cached via `actions/cache`), train, run pytest, `podman
build` the image, start it and assert it serves a real prediction, then
`podman push` to GHCR on `main`/`master`.

**CD** (`.github/workflows/cd.yml`) — after a successful CI run: `podman login`,
`podman pull` the new image, deploy with `podman-compose up -d --no-build` (so
the *pulled* image is deployed, not a local rebuild), wait for health, run
smoke tests, and fail the pipeline if they fail.

Podman ships preinstalled on GitHub's `ubuntu-latest` runners, so no setup
action is needed; `podman-compose` is pip-installed in the CD job.

**No repository secrets are required.** The image is published to GitHub
Container Registry, and both jobs authenticate with the `GITHUB_TOKEN` that
Actions injects automatically — CI requests `packages: write`, CD requests
`packages: read`. The token is piped via `--password-stdin` so it never
appears in the process list or the run log.

The published image is `ghcr.io/<owner>/cats-dogs-api`. New GHCR packages are
**private** by default; to let anyone (or a Kubernetes cluster without an
`imagePullSecret`) pull it, set the package to public under
*Profile → Packages → cats-dogs-api → Package settings*.

Pushing to GHCR from a workstation needs a GitHub PAT with the `write:packages`
scope:

```bash
echo $GITHUB_PAT | podman login ghcr.io -u <your-github-username> --password-stdin
```

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods -l app=cats-dogs-api
```

Two replicas, liveness/readiness probes on `/health`, and resource
requests/limits. Reachable on NodePort `30080`.

The manifest references `ghcr.io/pranith-akkireddy/cats-dogs-api:latest`. If
the GHCR package is still private, the cluster cannot pull it anonymously —
either make the package public or create a pull secret:

```bash
kubectl create secret docker-registry ghcr-creds \
  --docker-server=ghcr.io \
  --docker-username=<your-github-username> \
  --docker-password=$GITHUB_PAT
```

then add `imagePullSecrets: [{name: ghcr-creds}]` to the pod spec.

## Monitoring

The service logs method, path, status code, latency, and a running request
count for every request — metadata only, never the uploaded image bytes.
`/metrics` exposes in-app counters.

Post-deployment performance tracking:

```bash
python scripts/collect_feedback.py -n 100
```

Sends held-out images to the live service, compares predictions against ground
truth, and writes `feedback_report.json` with overall accuracy and per-class
recall.

## Layout

```text
├── app/main.py              # FastAPI inference service
├── src/
│   ├── preprocess.py        # 224x224 RGB decode, split, augmentation, scaling
│   ├── model_utils.py       # persistence, metrics, plots
│   └── train.py             # baseline training + MLflow
├── scripts/
│   ├── fetch_data.py        # archive resolution, extract, preprocess
│   ├── smoke_test.py        # post-deploy health + prediction check
│   └── collect_feedback.py  # post-deployment accuracy tracking
├── tests/                   # pytest: preprocessing + inference
├── k8s/                     # Deployment + Service
├── .github/workflows/       # ci.yml, cd.yml
├── dvc.yaml                 # fetch -> train pipeline
├── Dockerfile
└── docker-compose.yml
```

Note: `data/`, `models/`, and `artifacts/` are DVC-tracked and excluded from
direct Git commits. The raw 800 MB archive is fetched/cached rather than
version-controlled; the preprocessed `.npz`, the scaler, and the trained model
are the DVC-tracked outputs.
