"""Data drift detection using Evidently AI (v0.7+ API)."""

from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import structlog
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class DriftReport:
    """Result of a drift detection check."""

    is_drifted: bool
    drift_score: float  # Share of drifted columns (0-1)
    drifted_columns: list  # Which features drifted
    column_scores: Dict[str, float]  # Per-column drift p-values
    details: Dict[str, Any]  # Full Evidently report data


class DriftDetector:
    """Monitors incoming data for statistical drift against a reference dataset.

    Uses Evidently AI under the hood. In v0.7+, uses DataDriftPreset which
    applies appropriate statistical tests per column type:
    - KS test for numerical features
    - Chi-squared / Jensen-Shannon for categorical features
    """

    _instance: Optional["DriftDetector"] = None

    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.5):
        """
        Args:
            reference_data: The "known good" data distribution to compare against.
            drift_threshold: Share of columns that need to drift to flag dataset-level drift.
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold

    @classmethod
    def get_instance(cls, reference_data: pd.DataFrame = None, drift_threshold: float = 0.5) -> "DriftDetector":
        """Get or create the singleton DriftDetector instance."""
        if cls._instance is None:
            if reference_data is None:
                reference_data = pd.DataFrame()
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
        if self.reference_data.empty or current_data.empty:
            return DriftReport(
                is_drifted=False,
                drift_score=0.0,
                drifted_columns=[],
                column_scores={},
                details={},
            )

        # Align columns between reference and current data
        common_columns = list(set(self.reference_data.columns) & set(current_data.columns))
        if not common_columns:
            return DriftReport(
                is_drifted=False,
                drift_score=0.0,
                drifted_columns=[],
                column_scores={},
                details={"error": "No common columns"},
            )

        ref_aligned = self.reference_data[common_columns]
        cur_aligned = current_data[common_columns]

        # Evidently v0.7+ API
        report = Report([DataDriftPreset(drift_share=self.drift_threshold)])
        snapshot = report.run(ref_aligned, cur_aligned)
        result = snapshot.dict()

        # Parse the v0.7 output structure
        column_scores = {}
        drifted_columns = []
        drifted_count = 0
        total_columns = 0

        for metric in result.get("metrics", []):
            metric_name = metric.get("metric_name", "")
            config = metric.get("config", {})

            # DriftedColumnsCount gives overall drift summary
            if "DriftedColumnsCount" in metric_name:
                value = metric.get("value", {})
                drifted_count = value.get("count", 0)
                drift_share = value.get("share", 0)

            # ValueDrift gives per-column drift scores
            elif "ValueDrift" in metric_name:
                col_name = config.get("column", "")
                threshold = config.get("threshold", 0.05)
                p_value = metric.get("value", 1.0)
                total_columns += 1

                column_scores[col_name] = p_value
                if p_value < threshold:
                    drifted_columns.append(col_name)

        # Determine overall drift
        drift_share = len(drifted_columns) / max(total_columns, 1)
        is_drifted = drift_share >= self.drift_threshold

        drift_report = DriftReport(
            is_drifted=is_drifted,
            drift_score=drift_share,
            drifted_columns=drifted_columns,
            column_scores=column_scores,
            details=result,
        )

        logger.info(
            "drift_check_complete",
            is_drifted=drift_report.is_drifted,
            drift_score=round(drift_report.drift_score, 3),
            drifted_columns=drifted_columns,
        )

        return drift_report

    def update_reference(self, new_reference: pd.DataFrame):
        """Update reference data after a successful retraining cycle."""
        self.reference_data = new_reference
        logger.info("reference_data_updated", rows=len(new_reference))
