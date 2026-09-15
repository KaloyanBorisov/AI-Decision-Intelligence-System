"""
Tests for H2O-3 AutoML Engine Integration
"""

import pytest
import pandas as pd
import numpy as np
from backend.ml.h2o_engine import is_h2o_available, H2OAutoMLEngine, H2OModelWrapper
from backend.ml.automl import AutoML


class TestH2OEngine:
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

    @pytest.fixture
    def regression_data(self):
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 4,
                "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 4,
            }
        )
        y = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] * 4, name="target")
        return X, y

    def test_h2o_connection_check(self):
        # Verify connection check returns a boolean without throwing unhandled exceptions
        available = is_h2o_available()
        assert isinstance(available, bool)

    def test_automl_fallback_when_h2o_disabled(self, classification_data):
        X, y = classification_data
        automl = AutoML(task_type="classification", test_size=0.2, use_h2o=False)
        result = automl.fit(X, y, experiment_name="test_fallback", log_artifacts=False)
        assert result["best_model"] is not None
        assert result["task_type"] == "classification"
        assert automl.best_model is not None
        preds = automl.predict(X.head(5))
        assert len(preds) == 5

    def test_automl_training_with_h2o_or_fallback(self, classification_data):
        X, y = classification_data
        automl = AutoML(task_type="classification", test_size=0.2, use_h2o=True)
        result = automl.fit(X, y, experiment_name="test_h2o_cls", log_artifacts=False)
        assert result["best_model"] is not None
        assert result["best_score"] >= 0
        assert automl.best_model is not None
        preds = automl.predict(X.head(5))
        assert len(preds) == 5

        # If the H2O cluster is actually reachable, training MUST have gone through
        # it rather than silently falling back to sklearn/XGBoost (e.g. due to a
        # client/server version mismatch). This asserts loudly instead of letting
        # a regression pass silently under the "or_fallback" name.
        if is_h2o_available():
            assert result.get("engine") == "h2o", (
                "H2O cluster is reachable but training fell back to the non-H2O "
                "path (check h2o-py client version vs the H2O cluster image version)"
            )
            assert result.get("variable_importance") is not None
