---
name: decisera-ml
description: Interact with Decisera AutoML platform to profile datasets, dispatch model training (H2O, FLAML, Scikit-Learn), monitor Celery tasks, compare metrics via MLflow, and fetch SHAP explainability insights.
---

# Decisera ML Skill for Hermes Agent

This skill allows Hermes Agent to orchestrate machine learning workflows on the Decisera platform.

## Available Capabilities

### 1. List and Inspect Datasets
- **API**: `GET http://backend:8000/api/v1/datasets`
- **Volume Inspection**: Read CSV directly from `/app/uploads/<dataset_id>.csv`
- **Data Quality Profiling**: `GET http://backend:8000/api/v1/datasets/{dataset_id}/profile`

### 2. Dispatch Model Training
- **API**: `POST http://backend:8000/api/v1/train`
- **Payload Example**:
  ```json
  {
    "dataset_id": "churn.csv",
    "target_column": "Churn",
    "task_type": "classification",
    "use_celery": true,
    "use_h2o": true,
    "h2o_max_runtime_secs": 180,
    "use_flaml": false,
    "flaml_time_budget_secs": 180,
    "experiment_name": "Hermes_AutoML_Benchmarking"
  }
  ```
- **Returns**: `task_id` (e.g. `202Accepted`)

### 3. Monitor Task Status
- **API**: `GET http://backend:8000/api/v1/tasks/{task_id}/status`
- Poll until `status == "completed"` or `status == "failed"`.
- When complete, returns `model_id`, `best_model`, `best_score`, and validation metrics.

### 4. Query MLflow Leaderboard
- **Endpoint**: `http://mlflow:5000`
- Query run metrics (ROC-AUC, F1, Accuracy, LogLoss, RMSE, R2) to compare algorithms.

### 5. Fetch SHAP Explainability & Insights
- **API**: `GET http://backend:8000/api/v1/models/{model_id}/explainability`
- Uncover top feature importance rankings and per-prediction attribution.

## Autonomous Workflow Instructions

When requested to run or compare models:
1. Identify the target variable and dataset from user request.
2. Check dataset profile if needed to detect class imbalances or missing values.
3. Submit asynchronous training request with appropriate engine flags (`use_h2o=true` or `use_flaml=true`).
4. Monitor task status periodically without blocking other tools.
5. Format the final output into a concise markdown table with key metrics, the winning algorithm, and top SHAP feature drivers.
