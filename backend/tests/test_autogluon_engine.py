"""
Tests for AutoGluon AutoML Engine Integration
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from backend.ml.autogluon_engine import (
    is_autogluon_available,
    AutoGluonEngine,
    AutoGluonModelWrapper,
)
from backend.ml.automl import AutoML


class TestAutoGluonEngine:
    @pytest.fixture
    def classification_data(self):
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 4,
                "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 4,
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 4, name="target")
        return X, y

    def test_autogluon_healthcheck_returns_bool(self):
        """Verify is_autogluon_available returns a boolean without throwing unhandled exceptions."""
        available = is_autogluon_available(timeout_secs=0.1)
        assert isinstance(available, bool)

    def test_model_wrapper_predict_remote(self):
        """Verify AutoGluonModelWrapper delegates predict to REST API."""
        wrapper = AutoGluonModelWrapper(
            model_id="test_model_1",
            model_path="/app/models/test_model_1",
            task_type="classification",
            feature_names=["feature1", "feature2"],
            classes=[0, 1],
            variable_importance={"feature1": 0.8, "feature2": 0.2},
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "predictions": [1, 0],
            "probabilities": [{"0": 0.2, "1": 0.8}, {"0": 0.9, "1": 0.1}],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client.post", return_value=mock_resp):
            df_test = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [0.1, 0.2]})
            preds = wrapper.predict(df_test)
            assert len(preds) == 2
            assert preds[0] == 1
            assert preds[1] == 0

            proba = wrapper.predict_proba(df_test)
            assert proba.shape == (2, 2)

            contrib = wrapper.predict_contributions(df_test)
            assert contrib == {"feature1": 0.8, "feature2": 0.2}

    def test_autogluon_engine_fit_mocked(self, classification_data):
        """Verify AutoGluonEngine dispatches to /fit and returns structured result."""
        X, y = classification_data
        engine = AutoGluonEngine(task_type="classification", time_limit=30, presets="medium_quality")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "model_id": "ag_12345",
            "model_path": "/app/models/ag_12345",
            "best_model": "WeightedEnsemble_L2",
            "best_score": 0.95,
            "problem_type": "binary",
            "feature_names": ["feature1", "feature2"],
            "leaderboard": [
                {"model": "WeightedEnsemble_L2", "score_val": 0.95, "fit_time": 1.2, "pred_time_val": 0.01, "stack_level": 2},
                {"model": "LightGBM_BAG_L1", "score_val": 0.91, "fit_time": 0.5, "pred_time_val": 0.005, "stack_level": 1},
            ],
            "variable_importance": {"feature1": 0.75, "feature2": 0.25},
            "run_id": "mlflow_run_ag123",
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client.post", return_value=mock_resp):
            result = engine.fit(X, y, dataset_id="ds_test", experiment_name="test_ag")
            assert result["engine"] == "autogluon"
            assert result["best_model"] == "WeightedEnsemble_L2"
            assert result["best_score"] == 0.95
            assert "WeightedEnsemble_L2" in result["all_results"]
            assert len(result["leaderboard"]) == 2
            assert isinstance(result["model"], AutoGluonModelWrapper)

    def test_automl_dispatches_to_autogluon(self, classification_data):
        """Verify AutoML correctly routes to AutoGluon when use_autogluon=True and service is available."""
        X, y = classification_data
        automl = AutoML(task_type="classification", test_size=0.2, use_autogluon=True, use_h2o=False)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "model_id": "ag_99999",
            "model_path": "/app/models/ag_99999",
            "best_model": "WeightedEnsemble_L2",
            "best_score": 0.98,
            "problem_type": "binary",
            "feature_names": ["feature1", "feature2"],
            "leaderboard": [{"model": "WeightedEnsemble_L2", "score_val": 0.98}],
            "variable_importance": {"feature1": 0.6, "feature2": 0.4},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("backend.ml.autogluon_engine.is_autogluon_available", return_value=True):
            with patch("httpx.Client.post", return_value=mock_resp):
                result = automl.fit(X, y, dataset_id="ds_ag", experiment_name="test_automl_ag", log_artifacts=False)
                assert result["engine"] == "autogluon"
                assert result["best_model"] == "WeightedEnsemble_L2"
                assert result["best_score"] == 0.98
                assert automl.best_model is not None
