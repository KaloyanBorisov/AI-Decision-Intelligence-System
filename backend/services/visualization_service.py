import logging
from typing import Dict, Any, Optional
import pandas as pd
from ..visualizations import (
    CorrelationHeatmap,
    FeatureImportancePlot,
    TrendAnalysisChart,
    ForecastPlot,
    InteractiveFilters,
)
from ..ml.data_ingestion import DataIngestion
from .dataset_service import dataset_service

logger = logging.getLogger(__name__)


class DatasetNotFoundError(Exception):
    """Raised when the requested dataset does not exist."""


class NotTimeSeriesError(Exception):
    """Raised when a dataset exists but has no usable date/target columns for a trend chart."""


class VisualizationService:
    def __init__(self):
        pass

    def get_correlation_heatmap(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        dataset = next(
            (d for d in dataset_service.datasets if d.id == dataset_id), None
        )
        if not dataset:
            return None
        df = DataIngestion.load_data(dataset.file_path)
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[float, int])
        if numeric_df.empty:
            return None
        corr = numeric_df.corr()
        return {
            "matrix": corr.round(4).values.tolist(),
            "columns": corr.columns.tolist(),
        }

    def get_feature_importance(self, model_id: str) -> Optional[Dict[str, Any]]:
        from .model_service import model_service

        info = model_service.models.get(model_id)
        if not info:
            return None
        model = info.get("model")
        if model is None:
            return None
        feature_names = info.get("feature_names") or []
        n_features = getattr(model, "n_features_in_", len(feature_names))
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        plot = FeatureImportancePlot(model, feature_names)
        importance = plot._get_importance()
        return {"feature_importance": importance.to_dict()}

    def get_trend_analysis(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Build a trend chart for a dataset.

        Raises:
            DatasetNotFoundError: no dataset with this id exists.
            NotTimeSeriesError: the dataset exists but has no date column
                and/or no usable numeric target for a trend chart.
        """
        dataset = next(
            (d for d in dataset_service.datasets if d.id == dataset_id), None
        )
        if not dataset:
            raise DatasetNotFoundError(dataset_id)
        try:
            df = DataIngestion.load_data(dataset.file_path)
            profile = getattr(dataset, "profile", {}) or {}
            target = profile.get("target_variable")
            if not target or target not in df.columns:
                num_cols = df.select_dtypes(include=[float, int]).columns
                target = num_cols[0] if not num_cols.empty else None

            # Find date col
            date_col = None
            for col in df.columns:
                if (
                    pd.api.types.is_datetime64_any_dtype(df[col])
                    or "date" in str(col).lower()
                    or "time" in str(col).lower()
                ):
                    date_col = col
                    break
            if not date_col or not target:
                raise NotTimeSeriesError(
                    f"Dataset {dataset_id} has no date column and/or no numeric "
                    f"target (date_col={date_col!r}, target={target!r})"
                )
            df_sorted = df[[date_col, target]].dropna().sort_values(date_col)
            # Downsample very large series so the chart stays responsive
            max_points = 2000
            if len(df_sorted) > max_points:
                step = len(df_sorted) // max_points
                df_sorted = df_sorted.iloc[::step]
            x_values = df_sorted[date_col].astype(str)
            return {
                "trends": [
                    {
                        "name": target,
                        "x": x_values.tolist(),
                        "y": df_sorted[target].tolist(),
                    }
                ]
            }
        except (DatasetNotFoundError, NotTimeSeriesError):
            raise
        except Exception:
            logger.exception(
                "Unexpected error generating trend analysis for dataset %s",
                dataset_id,
            )
            raise

    def get_forecast_plot(
        self, model_id: str, dataset_id: str
    ) -> Optional[Dict[str, Any]]:
        from .model_service import model_service

        model = model_service.models.get(model_id)
        dataset = next(
            (d for d in dataset_service.datasets if d.id == dataset_id), None
        )
        if not model or not dataset:
            return None
        try:
            df = DataIngestion.load_data(dataset.file_path)
            profile = getattr(dataset, "profile", {}) or {}
            target = profile.get("target_variable")
            if not target or target not in df.columns:
                num_cols = df.select_dtypes(include=[float, int]).columns
                target = num_cols[0] if not num_cols.empty else None
            date_col = None
            for col in df.columns:
                if (
                    pd.api.types.is_datetime64_any_dtype(df[col])
                    or "date" in str(col).lower()
                ):
                    date_col = col
                    break
            if not date_col or not target:
                return None
            plot = ForecastPlot(df, model, date_col, target)
            return plot.generate_plot()
        except Exception as e:
            return None

    def get_interactive_filters(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        dataset = next(
            (d for d in dataset_service.datasets if d.id == dataset_id), None
        )
        if not dataset:
            return None
        df = DataIngestion.load_data(dataset.file_path)
        filters = InteractiveFilters(df)
        return filters.generate_filters()


visualization_service = VisualizationService()
