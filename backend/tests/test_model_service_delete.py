"""
Tests for ModelService.delete_model's MLflow run/experiment cleanup.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.services.model_service import ModelService


@pytest.fixture
def service():
    svc = ModelService.__new__(ModelService)  # skip __init__ (avoids touching real disk/DB)
    svc.models = MagicMock()
    svc.tasks = {}
    return svc


def _model_info(mlflow_run_id="run-123", mlflow_experiment_name="exp-A"):
    return {
        "model": MagicMock(all_model_ids=None, h2o_model_id=None),
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment_name": mlflow_experiment_name,
    }


class TestDeleteModelMlflowCleanup:
    def test_deletes_orphaned_experiment_when_no_models_reference_it(self, service):
        service.models.__contains__ = MagicMock(return_value=True)
        service.models.__getitem__ = MagicMock(return_value=_model_info())
        service.models.__delitem__ = MagicMock()

        mock_client = MagicMock()
        mock_experiment = MagicMock(experiment_id="42", lifecycle_stage="active")
        mock_client.get_experiment_by_name.return_value = mock_experiment

        with patch("backend.utils.storage.models_storage.all", return_value=[]), patch(
            "backend.services.model_service.cache_delete"
        ), patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            service.delete_model("model_1")

        mock_client.delete_run.assert_called_once_with("run-123")
        mock_client.get_experiment_by_name.assert_called_once_with("exp-A")
        mock_client.delete_experiment.assert_called_once_with("42")

    def test_keeps_experiment_when_another_model_still_references_it(self, service):
        service.models.__contains__ = MagicMock(return_value=True)
        service.models.__getitem__ = MagicMock(return_value=_model_info())
        service.models.__delitem__ = MagicMock()

        mock_client = MagicMock()

        remaining_rows = [{"model_id": "model_2", "mlflow_experiment_name": "exp-A"}]
        with patch("backend.utils.storage.models_storage.all", return_value=remaining_rows), patch(
            "backend.services.model_service.cache_delete"
        ), patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            service.delete_model("model_1")

        mock_client.delete_run.assert_called_once_with("run-123")
        mock_client.get_experiment_by_name.assert_not_called()
        mock_client.delete_experiment.assert_not_called()

    def test_skips_experiment_lookup_when_experiment_name_missing(self, service):
        service.models.__contains__ = MagicMock(return_value=True)
        service.models.__getitem__ = MagicMock(
            return_value=_model_info(mlflow_run_id="", mlflow_experiment_name="")
        )
        service.models.__delitem__ = MagicMock()

        mock_client = MagicMock()

        with patch("backend.utils.storage.models_storage.all", return_value=[]), patch(
            "backend.services.model_service.cache_delete"
        ), patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            service.delete_model("model_1")

        mock_client.delete_run.assert_not_called()
        mock_client.get_experiment_by_name.assert_not_called()
        mock_client.delete_experiment.assert_not_called()

    def test_does_not_delete_already_deleted_experiment(self, service):
        service.models.__contains__ = MagicMock(return_value=True)
        service.models.__getitem__ = MagicMock(return_value=_model_info())
        service.models.__delitem__ = MagicMock()

        mock_client = MagicMock()
        mock_experiment = MagicMock(experiment_id="42", lifecycle_stage="deleted")
        mock_client.get_experiment_by_name.return_value = mock_experiment

        with patch("backend.utils.storage.models_storage.all", return_value=[]), patch(
            "backend.services.model_service.cache_delete"
        ), patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            service.delete_model("model_1")

        mock_client.get_experiment_by_name.assert_called_once_with("exp-A")
        mock_client.delete_experiment.assert_not_called()
