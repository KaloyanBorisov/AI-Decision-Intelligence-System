"""
Celery tasks for async processing integrated with new AutoML engine
"""

from .celery_app import celery_app
from .services.dataset_service import dataset_service
from .services.model_service import model_service
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="tasks.train_model")
def train_model_task(
    self,
    task_id: str,
    dataset_id: str,
    target_column: str,
    task_type: str = "auto",
    test_size: float = 0.2,
    experiment_name: str = "AutoML",
    use_h2o: bool = True,
    h2o_max_runtime_secs: int = 180,
    use_flaml: bool = False,
    flaml_time_budget_secs: int = 180,
):
    """
    Async Celery task for model training. Runs in a separate `celery_worker`
    container/process from the FastAPI API server (see docker-compose.yml),
    dispatched via .delay() from the /train endpoint when the client asks for
    Celery-backed execution instead of the default in-process BackgroundTasks.

    This is a thin delegate to model_service.train_model_async — the same,
    already-tested training path used by the in-process (BackgroundTasks)
    execution mode, so there's exactly one place that does data prep, AutoML
    fitting, explainability, and model persistence. Task-status updates go
    through model_service._set_task_status, which mirrors them to Redis so
    the API process's GET /tasks/{id}/status can see progress written by this
    worker process, not just tasks run in its own process.

    Args:
        task_id: Unique task identifier
        dataset_id: ID of the dataset to train on
        target_column: Name of the target variable
        task_type: 'classification', 'regression', or 'auto'
        test_size: Proportion for test set
        experiment_name: MLflow experiment name
        use_h2o: Whether to try the H2O cluster first when available
        h2o_max_runtime_secs: Search budget when H2O is used for training
        use_flaml: Whether to use FLAML's in-process AutoML search
        flaml_time_budget_secs: Search budget when FLAML is used for training
    """
    try:
        logger.info(
            f"[Task {task_id}] Starting AutoML training (celery worker) for dataset {dataset_id}"
        )

        dataset_df = model_service.get_dataset(dataset_id)

        if target_column not in dataset_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        model_service.train_model_async(
            task_id=task_id,
            dataset_df=dataset_df,
            target_column=target_column,
            dataset_id=dataset_id,
            task_type=task_type,
            test_size=test_size,
            experiment_name=experiment_name,
            use_h2o=use_h2o,
            h2o_max_runtime_secs=h2o_max_runtime_secs,
            use_flaml=use_flaml,
            flaml_time_budget_secs=flaml_time_budget_secs,
        )

        status = model_service.get_task_status(task_id) or {}
        if status.get("status") == "failed":
            raise RuntimeError(status.get("error", "Training failed"))

        results = status.get("results", {})
        logger.info(f"[Task {task_id}] Training completed successfully")

        return {
            "status": "completed",
            "task_id": task_id,
            "model_id": status.get("model_id"),
            "best_model": results.get("best_model"),
            "best_score": results.get("best_score"),
            "message": status.get("message"),
        }

    except Exception as e:
        logger.error(f"[Task {task_id}] Training failed: {e}")
        # train_model_async already writes a "failed" task status on its own
        # exceptions; this covers failures before that point (e.g. bad dataset_id).
        existing = model_service.get_task_status(task_id)
        if existing is None or existing.get("status") != "failed":
            model_service._set_task_status(task_id, {
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "progress": 0,
                "error": str(e),
            })
        raise


@celery_app.task(bind=True, name="tasks.batch_predict")
def batch_predict_task(self, model_id: str, data_list: list):
    """
    Async task for batch predictions

    Args:
        model_id: ID of the trained model
        data_list: List of dictionaries with input data
    """
    try:
        logger.info(
            f"Batch prediction task for model {model_id}, {len(data_list)} samples"
        )

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 20, "message": "Loading model..."}
        )

        # Make predictions using model service
        results = model_service.predict_batch(model_id, data_list)

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 100, "message": "Predictions completed"}
        )

        logger.info(f"Batch predictions completed for model {model_id}")

        return {"status": "completed", "predictions": results, "count": len(results)}

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise


@celery_app.task(bind=True, name="tasks.profile_dataset")
def profile_dataset_task(self, dataset_id: str):
    """
    Async task for dataset profiling and quality analysis

    Args:
        dataset_id: ID of the dataset to profile
    """
    try:
        logger.info(f"Profiling dataset {dataset_id}")

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 10, "message": "Loading dataset..."}
        )

        # Load dataset
        dataset_df = model_service.get_dataset(dataset_id)

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 30, "message": "Analyzing data..."}
        )

        # Generate profile
        profile = {
            "rows": len(dataset_df),
            "columns": len(dataset_df.columns),
            "memory_usage": dataset_df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "columns_info": [],
            "missing_values": {},
            "duplicates": int(dataset_df.duplicated().sum()),
            "data_types": {},
        }

        # Analyze each column
        for col in dataset_df.columns:
            col_data = dataset_df[col]
            col_info = {
                "name": col,
                "dtype": str(col_data.dtype),
                "unique_values": int(col_data.nunique()),
                "missing_count": int(col_data.isna().sum()),
                "missing_percent": float(col_data.isna().sum() / len(col_data) * 100),
            }

            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                col_info.update(
                    {
                        "mean": (
                            float(col_data.mean())
                            if not col_data.isna().all()
                            else None
                        ),
                        "std": (
                            float(col_data.std()) if not col_data.isna().all() else None
                        ),
                        "min": (
                            float(col_data.min()) if not col_data.isna().all() else None
                        ),
                        "max": (
                            float(col_data.max()) if not col_data.isna().all() else None
                        ),
                        "median": (
                            float(col_data.median())
                            if not col_data.isna().all()
                            else None
                        ),
                    }
                )

            profile["columns_info"].append(col_info)

            if col_info["missing_count"] > 0:
                profile["missing_values"][col] = col_info["missing_count"]

            profile["data_types"][col] = str(col_data.dtype)

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 100, "message": "Profiling completed"}
        )

        logger.info(f"Dataset profile completed for {dataset_id}")

        return {"status": "completed", "profile": profile}

    except Exception as e:
        logger.error(f"Dataset profiling failed: {e}")
        raise


@celery_app.task(bind=True, name="tasks.generate_explanations")
def generate_explanations_task(self, model_id: str, num_samples: int = 100):
    """
    Async task for generating SHAP explanations

    Args:
        model_id: ID of the trained model
        num_samples: Number of samples to use for explanation
    """
    try:
        logger.info(f"Generating explanations for model {model_id}")

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 20, "message": "Loading model..."}
        )

        # Get global explanations
        importance = model_service.get_global_explanation(model_id, top_n=20)

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 70, "message": "Generating plots..."}
        )

        # Generate plots
        summary_plot = model_service.get_explanation_plot(model_id, plot_type="summary")
        importance_plot = model_service.get_explanation_plot(
            model_id, plot_type="importance"
        )

        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={"progress": 100, "message": "Explanations generated"},
        )

        logger.info(f"Explanations generated for model {model_id}")

        return {
            "status": "completed",
            "importance": importance,
            "plots": {"summary": summary_plot, "importance": importance_plot},
        }

    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise
