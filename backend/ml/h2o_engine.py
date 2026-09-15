"""
H2O-3 AutoML Engine Integration
Delegates model training, hyperparameter optimization, and ensemble building
to the dedicated H2O cluster while maintaining MLflow tracking and low-latency MOJO exports.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.h2o

from ..utils.config import settings

logger = logging.getLogger(__name__)

# Track H2O connection state
_h2o_initialized = False


def is_h2o_available() -> bool:
    """Check if the remote H2O cluster is reachable."""
    global _h2o_initialized
    try:
        import h2o
        if not _h2o_initialized:
            h2o_url = getattr(settings, "h2o_url", "http://localhost:54321")
            h2o.init(url=h2o_url, verbose=False, max_mem_size=None)
            _h2o_initialized = True
        # Verify cluster is responsive
        return h2o.connection().connected
    except Exception as exc:
        logger.warning(f"H2O cluster check failed: {exc}")
        _h2o_initialized = False
        return False


class H2OModelWrapper:
    """
    Scikit-learn compatible wrapper around trained H2O Model / MOJO
    so it integrates seamlessly into the existing ModelService and FastAPI endpoints.
    """

    def __init__(
        self,
        model_id: str,
        task_type: str,
        feature_names: List[str],
        classes: Optional[List[str]] = None,
        mojo_path: Optional[str] = None,
        h2o_model_id: Optional[str] = None,
        all_model_ids: Optional[List[str]] = None,
    ):
        self.model_id = model_id
        self.task_type = task_type
        self.feature_names = feature_names
        self.classes_ = classes or []
        self.mojo_path = mojo_path
        self.h2o_model_id = h2o_model_id
        # Every AutoML run trains a whole leaderboard (GBM/DRF/GLM/XGBoost/
        # StackedEnsemble variants), not just the leader — all of them stay
        # resident in cluster memory. Keep every id so deletion can clean up
        # the full run, not just the one model this wrapper serves.
        self.all_model_ids = all_model_ids or ([h2o_model_id] if h2o_model_id else [])

    def _get_model(self):
        """
        Look up this model in the H2O cluster, auto-healing if it's missing —
        e.g. the h2o container was restarted, wiping all in-memory models.
        H2O never persists models on its own (that's a deliberate design
        choice, not a gap in H2O); training already exports a MOJO to durable
        storage for exactly this situation, so re-import it on demand instead
        of failing every prediction until someone retrains from scratch.
        """
        import h2o

        try:
            return h2o.get_model(self.h2o_model_id)
        except Exception as exc:
            if not self.mojo_path or not os.path.exists(self.mojo_path):
                raise

            logger.warning(
                f"H2O model {self.h2o_model_id} not found in cluster ({exc}); "
                f"re-importing from MOJO at {self.mojo_path}"
            )
            imported = h2o.import_mojo(self.mojo_path)
            self.h2o_model_id = imported.model_id
            if self.h2o_model_id not in self.all_model_ids:
                self.all_model_ids.append(self.h2o_model_id)
            logger.info(f"Re-imported H2O model as {self.h2o_model_id}")
            return imported

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate point predictions for input data."""
        import h2o

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names[: X.shape[1]])

        hf = h2o.H2OFrame(X)
        model = self._get_model()
        preds_hf = model.predict(hf)
        preds_df = preds_hf.as_data_frame()

        if self.task_type == "classification":
            predict_col = preds_df["predict"]
            return predict_col.values
        else:
            return preds_df["predict"].values.astype(float)

    def predict_contributions(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """
        Per-instance feature contributions (H2O's SHAP-equivalent for tree
        models: GBM/DRF/XGBoost), used as the local-explanation fallback since
        the SHAP library itself can't introspect an H2O model.
        """
        import h2o

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names[: X.shape[1]])

        hf = h2o.H2OFrame(X)
        model = self._get_model()
        try:
            contrib_hf = model.predict_contributions(hf)
        except Exception as exc:
            # GLM (and some other non-tree algorithms) only support contribution
            # calculation relative to a background frame, unlike GBM/DRF/XGBoost
            # which work without one. Retry supplying the instance itself as its
            # own background before giving up.
            logger.debug(f"predict_contributions without background_frame failed ({exc}); retrying with one")
            contrib_hf = model.predict_contributions(hf, background_frame=hf)
        contrib_df = contrib_hf.as_data_frame()

        # First row only (single-instance explanation); drop the bias/intercept
        # column H2O appends (named "BiasTerm").
        row = contrib_df.iloc[0]
        return {
            col: float(row[col]) for col in contrib_df.columns if col != "BiasTerm"
        }

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate prediction probabilities for classification."""
        import h2o

        if self.task_type != "classification":
            raise ValueError("predict_proba is only supported for classification tasks")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names[: X.shape[1]])

        hf = h2o.H2OFrame(X)
        model = self._get_model()
        preds_hf = model.predict(hf)
        preds_df = preds_hf.as_data_frame()

        prob_cols = [c for c in preds_df.columns if c != "predict"]
        if prob_cols:
            return preds_df[prob_cols].values
        return np.column_stack([1 - preds_df["predict"].values, preds_df["predict"].values])


class H2OAutoMLEngine:
    """
    Automated Machine Learning engine delegating training to H2O-3 cluster.
    """

    def __init__(
        self,
        task_type: str = "auto",
        max_models: int = 10,
        max_runtime_secs: int = 180,
        seed: int = 42,
    ):
        self.task_type = task_type
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.leader = None
        self.leaderboard = None
        self.best_model_name = None
        self.best_score = 0.0
        self.wrapper = None

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "auto",
        dataset_id: str = "",
        experiment_name: str = "AutoML",
        log_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute H2O AutoML training, logging run into MLflow and exporting MOJO artifact.
        """
        import h2o
        from h2o.automl import H2OAutoML

        # 1. Connect to H2O Cluster
        h2o_url = getattr(settings, "h2o_url", "http://localhost:54321")
        h2o.init(url=h2o_url, verbose=False)

        # 2. Prepare Data
        clean_df = df.dropna(subset=[target_column]).copy()
        y_series = clean_df[target_column]

        # Auto-detect task type
        if task_type == "auto":
            if not pd.api.types.is_numeric_dtype(y_series) or y_series.nunique() <= 20:
                self.task_type = "classification"
            else:
                self.task_type = "regression"
        else:
            self.task_type = task_type

        # Convert to H2OFrame
        hf = h2o.H2OFrame(clean_df)

        if self.task_type == "classification":
            hf[target_column] = hf[target_column].asfactor()

        features = [col for col in clean_df.columns if col != target_column]

        # Split into train/test
        train_hf, test_hf = hf.split_frame(ratios=[0.8], seed=self.seed)

        # 3. Configure and Run H2O AutoML
        logger.info(
            f"Starting H2O AutoML ({self.task_type}) on cluster {h2o_url} "
            f"with max_models={self.max_models}, max_runtime_secs={self.max_runtime_secs}"
        )

        aml = H2OAutoML(
            max_models=self.max_models,
            max_runtime_secs=self.max_runtime_secs,
            seed=self.seed,
            sort_metric="AUTO",
            stopping_metric="AUTO",
            stopping_rounds=3,
        )

        try:
            mlflow.set_experiment(experiment_name)
        except Exception as exc:
            logger.warning(f"Could not set MLflow experiment: {exc}")

        results: Dict[str, Any] = {}
        mojo_path = None

        with mlflow.start_run(run_name=f"H2O_AutoML_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.log_param("engine", "H2O-3")
            mlflow.log_param("task_type", self.task_type)
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("max_models", self.max_models)
            mlflow.log_param("max_runtime_secs", self.max_runtime_secs)
            if dataset_id:
                mlflow.log_param("dataset_id", dataset_id)

            # Train models in H2O
            aml.train(x=features, y=target_column, training_frame=train_hf, leaderboard_frame=test_hf)

            self.leader = aml.leader
            self.best_model_name = self.leader.model_id if self.leader else "H2O_Model"

            # 4. Extract Leaderboard and Metrics
            lb_df = aml.leaderboard.as_data_frame()
            all_results = {}

            for _, row in lb_df.iterrows():
                m_id = str(row["model_id"])
                all_results[m_id] = {
                    "model_name": m_id,
                    "metrics": {k: float(v) for k, v in row.items() if k != "model_id" and pd.notna(v)},
                }

            # Evaluate performance on test frame
            perf = self.leader.model_performance(test_hf)
            metrics: Dict[str, float] = {}

            if self.task_type == "classification":
                try:
                    metrics["auc"] = float(perf.auc()) if hasattr(perf, "auc") and perf.auc() is not None else 0.0
                    metrics["accuracy"] = float(perf.accuracy()[0][1]) if hasattr(perf, "accuracy") and perf.accuracy() else 0.0
                    metrics["logloss"] = float(perf.logloss()) if hasattr(perf, "logloss") and perf.logloss() is not None else 0.0
                except Exception as exc:
                    # Do NOT fabricate a plausible-looking score here — a made-up
                    # number is worse than a visibly missing one. Surface the
                    # failure so the caller can decide whether to trust this run.
                    logger.error(f"Metrics extraction failed for H2O model {self.leader.model_id if self.leader else '?'}: {exc}")
                    raise RuntimeError(f"Could not extract evaluation metrics from H2O model: {exc}") from exc
                self.best_score = metrics.get("auc") or metrics.get("accuracy", 0.0)
            else:
                try:
                    metrics["rmse"] = float(perf.rmse()) if hasattr(perf, "rmse") else 0.0
                    metrics["mae"] = float(perf.mae()) if hasattr(perf, "mae") else 0.0
                    metrics["r2_score"] = float(perf.r2()) if hasattr(perf, "r2") else 0.0
                except Exception as exc:
                    logger.error(f"Metrics extraction failed for H2O model {self.leader.model_id if self.leader else '?'}: {exc}")
                    raise RuntimeError(f"Could not extract evaluation metrics from H2O model: {exc}") from exc
                self.best_score = metrics.get("r2_score", 0.0)

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # 5. Export MOJO Model Artifact
            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)
            try:
                mojo_path = self.leader.download_mojo(path=str(models_dir), get_genmodel_jar=True)
                if log_artifacts and mojo_path:
                    mlflow.log_artifact(mojo_path)
            except Exception as exc:
                logger.warning(f"Could not download MOJO: {exc}")

            # 6. Extract Variable Importances if available
            varimp = {}
            try:
                varimp_list = self.leader.varimp(use_pandas=True)
                if varimp_list is not None and not varimp_list.empty:
                    for _, vrow in varimp_list.head(10).iterrows():
                        varimp[str(vrow["variable"])] = float(vrow.get("relative_importance", 0.0))
            except Exception as exc:
                logger.debug(f"Varimp not available for this model type: {exc}")

            # Every model AutoML trained during this run (leaderboard, incl. the
            # leader) — all of it stays resident in the cluster and needs to be
            # cleaned up together, not just the leader, when the model is deleted.
            all_model_ids = list(all_results.keys())
            if self.leader.model_id not in all_model_ids:
                all_model_ids.append(self.leader.model_id)

            # Build wrapper
            classes = [str(c) for c in clean_df[target_column].unique()] if self.task_type == "classification" else []
            self.wrapper = H2OModelWrapper(
                model_id=self.leader.model_id,
                task_type=self.task_type,
                feature_names=features,
                classes=classes,
                mojo_path=mojo_path,
                h2o_model_id=self.leader.model_id,
                all_model_ids=all_model_ids,
            )

            results = {
                "engine": "h2o",
                "best_model": self.best_model_name,
                "best_score": float(self.best_score),
                "task_type": self.task_type,
                "metrics": metrics,
                "all_results": all_results,
                "all_model_ids": all_model_ids,
                "feature_names": features,
                "variable_importance": varimp,
                "model": self.wrapper,
                "mojo_path": mojo_path,
                "run_id": run.info.run_id,
            }

        return results
