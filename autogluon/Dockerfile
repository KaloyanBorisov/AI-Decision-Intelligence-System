FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by LightGBM / CatBoost; curl is needed for Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY autogluon/requirements.txt requirements.txt

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && pip uninstall -y --no-input nvidia-nccl-cu12 || true

COPY autogluon/main.py /app/main.py

RUN mkdir -p /app/models /app/uploads /app/mlops

EXPOSE 8010

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
