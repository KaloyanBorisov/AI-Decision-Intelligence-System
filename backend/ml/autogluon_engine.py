"""
AutoGluon AutoML Engine Integration
Delegates model training, stacking, and hyperparameter optimization to the dedicated
AutoGluon microservice container while exposing a scikit-learn compatible wrapper.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import httpx

from ..utils.config import settings

logger = logging.getLogger(__name__)


def is_autogluon_available(timeout_secs: float = 2.0) -> bool:
    """Check if the remote AutoGluon microservice container is healthy and reachable."""
    ag_url = getattr(settings, "autogluon_url", "http://localhost:8010")
    try:
        with httpx.Client(timeout=timeout_secs) as client:
            resp = client.get(f"{ag_url}/health")
            return resp.status_code == 200
    except Exception as exc:
        logger.debug(f"AutoGluon service check failed ({ag_url}): {exc}")
        return False


class AutoGluonModelWrapper:
    """
    Scikit-learn compatible wrapper around a trained AutoGluon TabularPredictor.
    Allows seamless integration with ModelService, /predict, and explainability endpoints.
    """

    def __init__(
        self,
        model_id: str,
        model_path: str,
        task_type: str,
        feature_names: List[str],
        classes: Optional[List[Any]] = None,
        variable_importance: Optional[Dict[str, float]] = None,
        leaderboard: Optional[List[Dict[str, Any]]] = None,
        best_model_name: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.task_type = task_type
        self.feature_names = feature_names
        self.classes_ = classes or []
        self.variable_importance = variable_importance or {}
        self.leaderboard = leaderboard or []
        self.best_model_name = best_model_name
        self.ag_url = getattr(settings, "autogluon_url", "http://localhost:8010")

    def predict(self, X: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]) -> np.ndarray:
        """Generate predictions via AutoGluon microservice or local fallback."""
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_names[: X.shape[1]])
            records = df.to_dict(orient="records")
        elif isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        else:
            records = list(X)

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.ag_url}/predict",
                    json={"model_path": self.model_path, "data": records},
                )
                resp.raise_for_status()
                res = resp.json()
                return np.array(res["predictions"])
        except Exception as exc:
            logger.warning(f"Remote AutoGluon predict failed ({exc}), attempting local load...")
            try:
                from autogluon.tabular import TabularPredictor
                predictor = TabularPredictor.load(self.model_path)
                df = pd.DataFrame(records)
                preds = predictor.predict(df)
                return np.array(preds)
            except Exception as local_exc:
                logger.error(f"Local AutoGluon predict failed: {local_exc}")
                raise RuntimeError(f"Failed to generate predictions with AutoGluon: {exc}") from exc

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]) -> np.ndarray:
        """Generate prediction probabilities for classification tasks."""
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_names[: X.shape[1]])
            records = df.to_dict(orient="records")
        elif isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        else:
            records = list(X)

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.ag_url}/predict",
                    json={"model_path": self.model_path, "data": records},
                )
                resp.raise_for_status()
                res = resp.json()
                proba_records = res.get("probabilities", [])
                if proba_records:
                    proba_df = pd.DataFrame(proba_records)
                    return proba_df.values
                # Fallback to 1-hot from point predictions
                preds = np.array(res["predictions"])
                return preds
        except Exception as exc:
            logger.warning(f"Remote predict_proba failed ({exc}); attempting local fallback...")
            try:
                from autogluon.tabular import TabularPredictor
                predictor = TabularPredictor.load(self.model_path)
                df = pd.DataFrame(records)
                proba_df = predictor.predict_proba(df)
                return proba_df.values
            except Exception as local_exc:
                logger.error(f"Local AutoGluon predict_proba failed: {local_exc}")
                raise RuntimeError(f"Failed to predict probabilities with AutoGluon: {exc}") from exc

    def predict_contributions(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """
        Feature contribution / importance fallback for single-instance or global explanation.
        """
        if self.variable_importance:
            return self.variable_importance
        return {feat: 1.0 / max(1, len(self.feature_names)) for feat in self.feature_names}


class AutoGluonEngine:
    """
    AutoML Engine client that dispatches training to the AutoGluon microservice container.
    """

    def __init__(
        self,
        task_type: str = "auto",
        time_limit: int = 180,
        presets: str = "medium_quality",
        eval_metric: Optional[str] = None,
    ):
        self.task_type = task_type
        self.time_limit = time_limit
        self.presets = presets
        self.eval_metric = eval_metric
        self.ag_url = getattr(settings, "autogluon_url", "http://localhost:8010")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "auto",
        dataset_id: str = "",
        experiment_name: str = "AutoML",
        dataset_path: Optional[str] = None,
        log_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Dispatch training request to the AutoGluon container.
        """
        # Auto-detect task type if needed
        if task_type == "auto":
            if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
                resolved_task = "classification"
            else:
                resolved_task = "regression"
        else:
            resolved_task = task_type

        target_name = y.name if getattr(y, "name", None) else "target"
        
        # Prepare payload: if dataset_path is accessible to the container, pass path;
        # otherwise pass combined DataFrame records.
        payload: Dict[str, Any] = {
            "target_column": str(target_name),
            "task_type": resolved_task,
            "time_limit": self.time_limit,
            "presets": self.presets,
            "eval_metric": self.eval_metric,
            "dataset_id": dataset_id,
            "experiment_name": experiment_name,
        }

        if dataset_path and os.path.exists(dataset_path):
            payload["dataset_path"] = str(dataset_path)
        else:
            df_combined = X.copy()
            df_combined[target_name] = y.values
            payload["data"] = df_combined.to_dict(orient="records")

        logger.info(
            f"Dispatching AutoGluon fit to {self.ag_url}/fit (time_limit={self.time_limit}s, "
            f"presets={self.presets}, task={resolved_task})"
        )

        http_timeout = max(300.0, float(self.time_limit + 120))
        with httpx.Client(timeout=http_timeout) as client:
            resp = client.post(f"{self.ag_url}/fit", json=payload)
            resp.raise_for_status()
            res = resp.json()

        # Build leader results map
        leaderboard = res.get("leaderboard", [])
        all_results = {}
        for row in leaderboard:
            m_name = row.get("model", "unknown")
            score = row.get("score_val", 0.0)
            all_results[m_name] = {
                "score": float(score),
                "fit_time": row.get("fit_time"),
                "pred_time_val": row.get("pred_time_val"),
                "stack_level": row.get("stack_level"),
            }

        feature_names = res.get("feature_names", list(X.columns))
        variable_importance = res.get("variable_importance", {})

        wrapper = AutoGluonModelWrapper(
            model_id=res["model_id"],
            model_path=res["model_path"],
            task_type=resolved_task,
            feature_names=feature_names,
            variable_importance=variable_importance,
            leaderboard=leaderboard,
            best_model_name=res.get("best_model"),
        )

        return {
            "engine": "autogluon",
            "model": wrapper,
            "best_model": res.get("best_model", "AutoGluon_Ensemble"),
            "best_score": res.get("best_score", 0.0),
            "all_results": all_results,
            "leaderboard": leaderboard,
            "task_type": resolved_task,
            "model_path": res.get("model_path"),
            "variable_importance": variable_importance,
            "feature_names": feature_names,
            "run_id": res.get("run_id"),
        }
