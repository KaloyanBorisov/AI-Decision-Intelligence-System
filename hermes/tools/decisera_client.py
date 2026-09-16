"""
Decisera Client Helper for Hermes Agent
Provides Python functions to interact directly with the Decisera API gateway.
"""

import os
import time
import requests
from typing import Dict, Any, Optional

BACKEND_URL = os.getenv("DECISERA_BACKEND_URL", "http://backend:8000")
API_BASE = f"{BACKEND_URL}/api/v1"


def list_datasets() -> Dict[str, Any]:
    """Retrieve list of all available datasets."""
    resp = requests.get(f"{API_BASE}/datasets", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_dataset_profile(dataset_id: str) -> Dict[str, Any]:
    """Get automated quality and statistical profile of a dataset."""
    resp = requests.get(f"{API_BASE}/datasets/{dataset_id}/profile", timeout=15)
    resp.raise_for_status()
    return resp.json()


def train_model(
    dataset_id: str,
    target_column: str,
    task_type: str = "auto",
    use_h2o: bool = True,
    use_flaml: bool = False,
    runtime_secs: int = 180,
    experiment_name: str = "Hermes_AutoML"
) -> Dict[str, Any]:
    """
    Dispatch an asynchronous AutoML training job to Decisera.
    """
    payload = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "task_type": task_type,
        "use_celery": True,
        "use_h2o": use_h2o,
        "h2o_max_runtime_secs": runtime_secs,
        "use_flaml": use_flaml,
        "flaml_time_budget_secs": runtime_secs,
        "experiment_name": experiment_name,
    }
    resp = requests.post(f"{API_BASE}/models/train", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Check status of a running Celery training task."""
    resp = requests.get(f"{API_BASE}/models/tasks/{task_id}/status", timeout=10)
    resp.raise_for_status()
    return resp.json()


def wait_for_task(task_id: str, poll_interval: int = 5, timeout: int = 600) -> Dict[str, Any]:
    """Poll task until completion or failure."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_task_status(task_id)
        current_state = status.get("status")
        if current_state in ("completed", "failed"):
            return status
        time.sleep(poll_interval)
    raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")


def get_model_explainability(model_id: str) -> Dict[str, Any]:
    """Fetch SHAP feature importance values and insights for a trained model."""
    resp = requests.get(f"{API_BASE}/models/{model_id}/explainability", timeout=15)
    resp.raise_for_status()
    return resp.json()
