FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for numpy/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY models ./models
COPY data/processed ./data/processed

ENV MODEL_PATH=/app/models/baseline_model.pkl
ENV SCALER_PATH=/app/data/processed/scaler.pkl
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
