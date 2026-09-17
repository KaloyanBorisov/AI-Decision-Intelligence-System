"""
AutoGluon Tabular AutoML Microservice
Exposes REST endpoints for fitting TabularPredictors, scoring, and computing feature importance.
"""

import os
import uuid
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("autogluon_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AutoGluon AutoML Microservice",
    description="Containerized AutoGluon Tabular training and inference engine",
    version="1.0.0",
)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class FitRequest(BaseModel):
    target_column: str
    dataset_path: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    task_type: str = "auto"
    time_limit: int = 180
    eval_metric: Optional[str] = None
    presets: str = "medium_quality"
    dataset_id: str = ""
    experiment_name: str = "AutoML"
    model_id: Optional[str] = None
    holdout_frac: Optional[float] = 0.2


class PredictRequest(BaseModel):
    model_path: str
    data: List[Dict[str, Any]]


class FeatureImportanceRequest(BaseModel):
    model_path: str
    dataset_path: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    target_column: Optional[str] = None


@app.get("/health")
def health_check():
    """Health check endpoint confirming AutoGluon availability."""
    try:
        import autogluon.tabular
        ag_version = getattr(autogluon.tabular, "__version__", "unknown")
    except ImportError:
        ag_version = "not_installed"

    return {
        "status": "healthy",
        "service": "autogluon",
        "autogluon_version": ag_version,
    }


@app.post("/fit")
def fit_model(req: FitRequest):
    """
    Train an AutoGluon TabularPredictor on the provided dataset.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"AutoGluon not installed: {exc}")

    # 1. Load dataset
    if req.dataset_path:
        path = Path(req.dataset_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset file not found at {req.dataset_path}")
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    elif req.data:
        df = pd.DataFrame(req.data)
    else:
        raise HTTPException(status_code=400, detail="Either dataset_path or data must be provided")

    if req.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' not found in dataset columns: {list(df.columns)}"
        )

    # 2. Map task type to AutoGluon problem_type
    problem_type = None
    if req.task_type in ("binary", "multiclass", "regression", "quantile"):
        problem_type = req.task_type
    elif req.task_type == "classification":
        n_unique = df[req.target_column].nunique(dropna=True)
        problem_type = "binary" if n_unique == 2 else "multiclass"
    elif req.task_type == "regression":
        problem_type = "regression"

    # 3. Setup model storage directory
    model_id = req.model_id or f"ag_{uuid.uuid4().hex[:12]}"
    save_path = str(MODELS_DIR / model_id)

    logger.info(
        f"Starting AutoGluon fit: model_id={model_id}, time_limit={req.time_limit}s, "
        f"presets={req.presets}, problem_type={problem_type}, rows={len(df)}"
    )

    try:
        predictor = TabularPredictor(
            label=req.target_column,
            problem_type=problem_type,
            eval_metric=req.eval_metric,
            path=save_path,
        )

        predictor.fit(
            train_data=df,
            time_limit=req.time_limit,
            presets=req.presets,
            holdout_frac=req.holdout_frac,
        )

        # 4. Extract leader, leaderboard, and feature importance
        best_model = predictor.model_best
        leaderboard_df = predictor.leaderboard(silent=True)
        leaderboard_records = leaderboard_df.to_dict(orient="records")

        # Extract best score
        best_row = leaderboard_df[leaderboard_df["model"] == best_model]
        best_score = float(best_row["score_val"].values[0]) if not best_row.empty else 0.0

        # Feature importance
        fi_dict = {}
        try:
            fi_df = predictor.feature_importance(df, silent=True)
            if fi_df is not None and not fi_df.empty and "importance" in fi_df.columns:
                fi_dict = {
                    feat: float(fi_df.loc[feat, "importance"])
                    for feat in fi_df.index
                }
        except Exception as fi_exc:
            logger.warning(f"Could not compute feature importance: {fi_exc}")

        # Model names
        all_models = predictor.model_names()

        # MLflow logging (if configured)
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        run_id = None
        if mlflow_uri:
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(req.experiment_name)
                with mlflow.start_run(run_name=f"AutoGluon_{model_id}") as run:
                    run_id = run.info.run_id
                    mlflow.log_param("engine", "AutoGluon")
                    mlflow.log_param("model_id", model_id)
                    mlflow.log_param("problem_type", predictor.problem_type)
                    mlflow.log_param("best_model", best_model)
                    mlflow.log_param("presets", req.presets)
                    mlflow.log_metric("val_score", best_score)
                    if req.dataset_id:
                        mlflow.log_param("dataset_id", req.dataset_id)
            except Exception as mlf_exc:
                logger.warning(f"MLflow logging failed: {mlf_exc}")

        return {
            "status": "success",
            "model_id": model_id,
            "model_path": save_path,
            "best_model": best_model,
            "best_score": best_score,
            "problem_type": predictor.problem_type,
            "eval_metric": predictor.eval_metric,
            "feature_names": [c for c in df.columns if c != req.target_column],
            "leaderboard": leaderboard_records,
            "all_models": all_models,
            "variable_importance": fi_dict,
            "run_id": run_id,
        }

    except Exception as exc:
        logger.exception("AutoGluon training failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Generate predictions using a saved TabularPredictor.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"AutoGluon not installed: {exc}")

    if not os.path.exists(req.model_path):
        raise HTTPException(status_code=404, detail=f"Model path '{req.model_path}' does not exist")

    df = pd.DataFrame(req.data)
    if df.empty:
        return {"predictions": [], "probabilities": []}

    try:
        predictor = TabularPredictor.load(req.model_path)
        preds = predictor.predict(df)
        preds_list = preds.tolist() if isinstance(preds, (pd.Series, np.ndarray)) else list(preds)

        proba_list = []
        if predictor.problem_type in ("binary", "multiclass"):
            try:
                proba_df = predictor.predict_proba(df)
                proba_list = proba_df.to_dict(orient="records")
            except Exception as proba_exc:
                logger.debug(f"predict_proba unavailable: {proba_exc}")

        return {
            "predictions": preds_list,
            "probabilities": proba_list,
            "problem_type": predictor.problem_type,
        }
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feature_importance")
def feature_importance(req: FeatureImportanceRequest):
    """
    Compute feature importance on demand for a trained predictor.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"AutoGluon not installed: {exc}")

    if not os.path.exists(req.model_path):
        raise HTTPException(status_code=404, detail=f"Model path '{req.model_path}' does not exist")

    predictor = TabularPredictor.load(req.model_path)

    if req.dataset_path and os.path.exists(req.dataset_path):
        df = pd.read_parquet(req.dataset_path) if req.dataset_path.endswith(".parquet") else pd.read_csv(req.dataset_path)
    elif req.data:
        df = pd.DataFrame(req.data)
    else:
        df = None

    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail="Either 'data' (records) or valid 'dataset_path' must be provided to compute permutation feature importance.",
        )

    try:
        fi_df = predictor.feature_importance(data=df, silent=True)
        fi_dict = {}
        if fi_df is not None and not fi_df.empty:
            fi_dict = {
                feat: float(fi_df.loc[feat, "importance"])
                for feat in fi_df.index
            }
        return {"feature_importance": fi_dict}
    except Exception as exc:
        logger.exception("Feature importance calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))
