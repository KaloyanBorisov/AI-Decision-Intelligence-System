"""
Smart City Digital Twin Ecosystem - Model Training Script
Can be run locally or converted for remote execution / Google Colab.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib

def load_and_merge_dataset(data_dir: str = "./data"):
    """
    Load relational tables and merge them into a unified feature table
    """
    data_path = Path(data_dir)
    print(f"Loading tables from {data_path}...")
    
    df_power = pd.read_csv(data_path / "power_grid.csv") if (data_path / "power_grid.csv").exists() else None
    df_weather = pd.read_csv(data_path / "weather.csv") if (data_path / "weather.csv").exists() else None
    df_traffic = pd.read_csv(data_path / "traffic.csv") if (data_path / "traffic.csv").exists() else None
    df_districts = pd.read_csv(data_path / "districts.csv") if (data_path / "districts.csv").exists() else None

    # Base table
    base_df = df_power.copy() if df_power is not None else df_traffic.copy()
    if base_df is None:
        raise FileNotFoundError("Could not find power_grid.csv or traffic.csv")

    if "timestamp" in base_df.columns:
        base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
        base_df["hour"] = base_df["timestamp"].dt.hour
        base_df["dayofweek"] = base_df["timestamp"].dt.dayofweek
        base_df["month"] = base_df["timestamp"].dt.month
        base_df["is_weekend"] = base_df["dayofweek"].isin([5, 6]).astype(int)

    if df_weather is not None and "timestamp" in df_weather.columns and "district_id" in df_weather.columns:
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])
        base_df = base_df.merge(df_weather, on=["timestamp", "district_id"], how="left", suffixes=("", "_weather"))

    if df_districts is not None and "district_id" in df_districts.columns:
        base_df = base_df.merge(df_districts, on="district_id", how="left")

    print(f"Master merged dataset shape: {base_df.shape}")
    return base_df

def preprocess_and_train(df: pd.DataFrame, output_dir: str = "./storage/models"):
    """
    Clean, impute missing values, train models, and save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_df = df.drop(columns=["timestamp"], errors="ignore").copy()

    # Identify numeric candidate targets
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_targets = [col for col in numeric_cols if any(k in col.lower() for k in ["load", "demand", "power", "speed", "congestion", "aqi", "pm2_5"])]
    target_col = candidate_targets[0] if candidate_targets else numeric_cols[0]
    print(f"Target variable: {target_col}")

    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    # Impute missing values
    for col in X.columns:
        if X[col].dtype in ["float64", "int64"]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype(str).fillna("Unknown")

    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LightGBM": lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    }

    best_score = -float("inf")
    best_model = None
    best_name = ""

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        print(f"{name} -> R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

    # Save model artifact
    artifact_path = os.path.join(output_dir, f"smart_city_{best_name.lower()}_best.joblib")
    joblib.dump(best_model, artifact_path)
    print(f"Saved best model ({best_name}) to: {artifact_path}")

if __name__ == "__main__":
    import kagglehub
    dataset_path = kagglehub.dataset_download("razanihababdellatif/smart-city-digital-twin-ecosystem-dataset")
    df = load_and_merge_dataset(dataset_path)
    preprocess_and_train(df)
