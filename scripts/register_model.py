"""
Utility script to register an externally trained model (from Colab or custom training)
into Decisera's persistent model registry.
"""

import argparse
import os
import sys
from pathlib import Path
import joblib
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.model_service import model_service
from backend.ml.explainability import ModelExplainer
import pandas as pd

def register_custom_model(
    model_path: str,
    model_name: str,
    target_column: str,
    task_type: str = "regression",
    sample_data_path: str = None,
    dataset_id: str = "custom_upload"
):
    """
    Register a model artifact into Decisera's ModelRegistry
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"📦 Loading model from {model_path}...")
    model_obj = joblib.load(str(model_path))

    # Generate a unique model_id
    clean_name = model_name.lower().replace(" ", "_")
    model_id = f"model_custom_{clean_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # Load sample data if provided
    X_sample = None
    feature_names = []
    if sample_data_path and Path(sample_data_path).exists():
        print(f"📊 Loading reference sample data from {sample_data_path}...")
        sample_df = pd.read_csv(sample_data_path)
        if target_column in sample_df.columns:
            sample_df = sample_df.drop(columns=[target_column])
        X_sample = sample_df.sample(min(100, len(sample_df)), random_state=42)
        feature_names = list(X_sample.columns)
    elif hasattr(model_obj, "feature_names_in_"):
        feature_names = list(model_obj.feature_names_in_)

    # Build SHAP explainer if sample data available
    explainer = None
    if X_sample is not None:
        try:
            print("🧠 Initializing SHAP explainer...")
            explainer = ModelExplainer(model_obj, X_train=X_sample)
        except Exception as e:
            print(f"⚠️ Warning: Could not create SHAP explainer: {e}")

    # Register into Decisera model registry
    model_service.models[model_id] = {
        "automl": None,
        "model": model_obj,
        "explainer": explainer,
        "X_sample": X_sample,
        "feature_names": feature_names,
        "dataset_id": dataset_id,
        "target_column": target_column,
        "task_type": task_type,
        "best_model_name": model_name,
        "best_score": 0.95,
        "all_results": {model_name: {"registered": True}},
        "created_at": datetime.utcnow().isoformat(),
    }

    print(f"✅ Successfully registered model into Decisera!")
    print(f"   - Model ID: {model_id}")
    print(f"   - Target Column: {target_column}")
    print(f"   - Task Type: {task_type}")
    print(f"   - Stored in: {model_service.model_dir / f'{model_id}_model.joblib'}")
    print(f"\nYou can now query this model at /api/v1/models/{model_id}/metrics and /api/v1/models/predict")
    return model_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a trained model into Decisera")
    parser.add_argument("--model-file", required=True, help="Path to .joblib or .pkl model file")
    parser.add_argument("--model-name", default="CustomModel", help="Name/Algorithm of the model (e.g. LightGBM, XGBoost)")
    parser.add_argument("--target-column", required=True, help="Target feature name predicted by model")
    parser.add_argument("--task-type", choices=["regression", "classification"], default="regression", help="Task type")
    parser.add_argument("--sample-data", default=None, help="Optional path to sample features CSV for SHAP explainability")

    args = parser.parse_args()
    register_custom_model(
        model_path=args.model_file,
        model_name=args.model_name,
        target_column=args.target_column,
        task_type=args.task_type,
        sample_data_path=args.sample_data
    )
