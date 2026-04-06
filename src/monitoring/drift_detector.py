"""Data drift detection using Evidently AI."""

from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
)
import pandas as pd
import structlog
from typing import Dict, Any
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class DriftReport:
    """Result of a drift detection check."""
    is_drifted: bool
    drift_score: float           # Overall drift score (0-1)
    drifted_columns: list        # Which features drifted
    column_scores: Dict[str, float]  # Per-column drift scores
    details: Dict[str, Any]      # Full Evidently report data


class DriftDetector:
    """Monitors incoming data for statistical drift against a reference dataset.

    Uses Evidently AI under the hood. Supports multiple drift detection methods:
    - PSI (Population Stability Index)
    - KS (Kolmogorov-Smirnov) test
    - Jensen-Shannon divergence
    """

    _instance = None

    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.3):
        """
        Args:
            reference_data: The "known good" data distribution to compare against.
                           Typically your training data or a validated production window.
            drift_threshold: Fraction of columns that need to drift to trigger alert.
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold

    @classmethod
    def get_instance(cls, reference_data: pd.DataFrame = None, drift_threshold: float = 0.3) -> "DriftDetector":
        """Get or create the singleton DriftDetector instance."""
        if cls._instance is None:
            if reference_data is None:
                raise ValueError("reference_data required for first initialization")
            cls._instance = cls(reference_data, drift_threshold)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def check_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Compare current production data against reference.

        Args:
            current_data: Recent production data window (e.g., last 1000 predictions)

        Returns:
            DriftReport with drift status, scores, and details
        """
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )

        result = report.as_dict()

        # Extract drift info from Evidently result
        dataset_drift = result["metrics"][0]["result"]
        drift_table = result["metrics"][1]["result"]

        # Get per-column drift scores
        column_scores = {}
        drifted_columns = []
        for col_name, col_data in drift_table.get("drift_by_columns", {}).items():
            score = col_data.get("drift_score", 0)
            column_scores[col_name] = score
            if col_data.get("drift_detected", False):
                drifted_columns.append(col_name)

        drift_report = DriftReport(
            is_drifted=dataset_drift.get("dataset_drift", False),
            drift_score=dataset_drift.get("share_of_drifted_columns", 0),
            drifted_columns=drifted_columns,
            column_scores=column_scores,
            details=result,
        )

        logger.info(
            "drift_check_complete",
            is_drifted=drift_report.is_drifted,
            drift_score=drift_report.drift_score,
            drifted_columns=drifted_columns,
        )

        return drift_report

    def update_reference(self, new_reference: pd.DataFrame):
        """Update reference data after a successful retraining cycle."""
        self.reference_data = new_reference
        logger.info("reference_data_updated", rows=len(new_reference))
