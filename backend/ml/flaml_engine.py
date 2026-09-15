"""
FLAML AutoML Engine Integration (SKETCH — not wired into automl.py yet)

Alternative to H2OAutoMLEngine (see h2o_engine.py). Mirrors its interface
(is_*_available / *Engine.fit returning the same result-dict shape) so it can
be swapped in with a small change to AutoML.fit(), for side-by-side comparison.

Unlike H2O, FLAML runs entirely in-process: no separate cluster, no JVM, no
client/server version to keep in lockstep, no docker-compose service. It
searches over the same model families already in requirements.txt (LightGBM,
XGBoost, RandomForest, ...), so the winning model is a plain, already-fitted
scikit-learn-compatible estimator — no wrapper needed, save_model/load_model
and notebook joblib.load() work exactly as they do today for the non-H2O path.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import mlflow

logger = logging.getLogger(__name__)


def is_flaml_available() -> bool:
    """Check that the flaml package is importable. No cluster/network check
    needed — FLAML has no server component."""
    try:
        import flaml  # noqa: F401
        return True
    except ImportError:
        logger.warning("flaml is not installed (pip install flaml)")
        return False


class FLAMLEngine:
    """
    AutoML engine using FLAML's in-process hyperparameter search over
    LightGBM / XGBoost / RandomForest / extra_trees / etc.
    """

    def __init__(
        self,
        task_type: str = "auto",
        time_budget_secs: int = 180,
        seed: int = 42,
    ):
        self.task_type = task_type
        self.time_budget_secs = time_budget_secs
        self.seed = seed
        self.best_estimator_name = None
        self.best_score = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "auto",
        dataset_id: str = "",
        experiment_name: str = "AutoML",
        log_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Run FLAML's AutoML search, logging to MLflow, and return a result dict
        shaped like H2OAutoMLEngine.fit()'s so callers (automl.py) don't need
        to special-case which engine ran.
        """
        from flaml import AutoML as FLAMLAutoML

        # Auto-detect task type the same way h2o_engine.py does
        if task_type == "auto":
            if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
                self.task_type = "classification"
            else:
                self.task_type = "regression"
        else:
            self.task_type = task_type

        flaml_task = "classification" if self.task_type == "classification" else "regression"

        automl = FLAMLAutoML()

        try:
            mlflow.set_experiment(experiment_name)
        except Exception as exc:
            logger.warning(f"Could not set MLflow experiment: {exc}")

        with mlflow.start_run(run_name=f"FLAML_AutoML_{self.task_type}") as run:
            mlflow.log_param("engine", "FLAML")
            mlflow.log_param("task_type", self.task_type)
            mlflow.log_param("time_budget_secs", self.time_budget_secs)
            if dataset_id:
                mlflow.log_param("dataset_id", dataset_id)

            automl.fit(
                X_train=X,
                y_train=y,
                task=flaml_task,
                time_budget=self.time_budget_secs,
                seed=self.seed,
                verbose=0,
                # FLAML has its own built-in MLflow auto-logging that activates
                # whenever it detects an active run (which we're inside of
                # here) and tries to log every candidate model via skops
                # serialization — this crashes on FLAML's wrapped estimators
                # ("could not be serialized in the skops serialization
                # format"). We do our own explicit logging below instead.
                mlflow_logging=False,
            )

            self.best_estimator_name = automl.best_estimator
            self.best_score = 1 - automl.best_loss  # FLAML tracks loss, not score

            # FLAML's leaderboard equivalent: per-estimator best config/loss
            all_results = {
                name: {"model_name": name, "metrics": {"loss": float(loss)}}
                for name, loss in (automl.best_loss_per_estimator or {}).items()
            }

            metrics = {"best_loss": float(automl.best_loss), "score": float(self.best_score)}
            mlflow.log_metrics(metrics)

            # The winning model is a real, plain scikit-learn-compatible object —
            # automl.model.estimator — not a remote handle. mlflow.sklearn (or
            # mlflow.lightgbm/xgboost as appropriate) can log it directly, and
            # joblib.dump(model, path) later produces a fully self-contained file.
            model = automl.model.estimator
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception as exc:
                logger.warning(f"Could not log FLAML model to MLflow: {exc}")

            varimp = {}
            try:
                importances = getattr(model, "feature_importances_", None)
                if importances is not None:
                    varimp = dict(zip(X.columns, (float(v) for v in importances)))
            except Exception as exc:
                logger.debug(f"Feature importance not available: {exc}")

            results = {
                "engine": "flaml",
                "best_model": self.best_estimator_name,
                "best_score": float(self.best_score),
                "task_type": self.task_type,
                "metrics": metrics,
                "all_results": all_results,
                "feature_names": list(X.columns),
                "variable_importance": varimp,
                "model": model,          # <- plain estimator, no wrapper needed
                "mojo_path": None,       # <- no equivalent needed; joblib is enough
                "run_id": run.info.run_id,
            }

        return results
