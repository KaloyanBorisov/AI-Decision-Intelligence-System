FROM python:3.11-slim

WORKDIR /app

# libgomp1: LightGBM's Linux wheel needs the system OpenMP runtime at import
# time (unlike xgboost's wheel, which bundles its own) — required for FLAML,
# which tunes LightGBM among its candidate models.
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt

RUN pip install --no-cache-dir -r backend/requirements.txt \
    # xgboost's PyPI wheel depends on nvidia-nccl-cu12 (GPU multi-node training
    # support) regardless of whether GPUs are available; this backend is
    # CPU-only (torch is pinned to the +cpu build too), so NCCL is ~450MB of
    # dead weight that CPU training/inference never touches. Safe to drop.
    && pip uninstall -y --no-input nvidia-nccl-cu12 || true

COPY . .

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/backend/models /app/backend/experiments /app/backend/logs /app/uploads

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/backend:.

CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]