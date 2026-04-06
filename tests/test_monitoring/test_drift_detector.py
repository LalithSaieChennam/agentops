"""Tests for drift detection."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.monitoring.drift_detector import DriftDetector, DriftReport


class TestDriftDetector:
    """Test the DriftDetector class."""

    def setup_method(self):
        """Reset singleton before each test."""
        DriftDetector.reset_instance()

    def test_drift_report_dataclass(self):
        """Test DriftReport dataclass."""
        report = DriftReport(
            is_drifted=True,
            drift_score=0.5,
            drifted_columns=["col1"],
            column_scores={"col1": 0.8},
            details={},
        )
        assert report.is_drifted is True
        assert report.drift_score == 0.5
        assert report.drifted_columns == ["col1"]

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        ref_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        instance1 = DriftDetector.get_instance(reference_data=ref_data)
        instance2 = DriftDetector.get_instance()
        assert instance1 is instance2

    def test_singleton_without_ref_data_creates_empty(self):
        """Test that first call without reference_data creates with empty DataFrame."""
        instance = DriftDetector.get_instance()
        assert instance is not None
        assert instance.reference_data.empty

    def test_reset_instance(self):
        """Test that reset clears the singleton."""
        ref_data = pd.DataFrame({"a": [1, 2, 3]})
        instance1 = DriftDetector.get_instance(reference_data=ref_data)
        DriftDetector.reset_instance()
        instance2 = DriftDetector.get_instance(reference_data=ref_data)
        assert instance1 is not instance2

    def test_update_reference(self):
        """Test updating reference data."""
        ref_data = pd.DataFrame({"a": [1, 2, 3]})
        detector = DriftDetector(ref_data)

        new_ref = pd.DataFrame({"a": [4, 5, 6]})
        detector.update_reference(new_ref)
        assert len(detector.reference_data) == 3

    def test_empty_reference_returns_no_drift(self):
        """Test that empty reference data returns no drift."""
        detector = DriftDetector(pd.DataFrame())
        result = detector.check_drift(pd.DataFrame({"a": [1, 2, 3]}))
        assert result.is_drifted is False
        assert result.drift_score == 0.0

    def test_empty_current_returns_no_drift(self):
        """Test that empty current data returns no drift."""
        detector = DriftDetector(pd.DataFrame({"a": [1, 2, 3]}))
        result = detector.check_drift(pd.DataFrame())
        assert result.is_drifted is False
        assert result.drift_score == 0.0

    def test_no_common_columns_returns_no_drift(self):
        """Test that no common columns returns no drift."""
        ref = pd.DataFrame({"a": [1, 2, 3]})
        cur = pd.DataFrame({"b": [4, 5, 6]})
        detector = DriftDetector(ref)
        result = detector.check_drift(cur)
        assert result.is_drifted is False

    def test_detects_significant_drift(self):
        """Test that actual data drift is detected with real Evidently."""
        np.random.seed(42)
        ref = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 200),
            "feature2": np.random.normal(0, 1, 200),
        })
        # Heavily shifted data — should trigger drift
        cur = pd.DataFrame({
            "feature1": np.random.normal(5, 1, 200),  # Shifted by 5 std devs
            "feature2": np.random.normal(0, 1, 200),  # Same distribution
        })

        detector = DriftDetector(ref, drift_threshold=0.3)
        result = detector.check_drift(cur)

        assert "feature1" in result.drifted_columns
        assert result.drift_score > 0  # At least some columns drifted

    def test_no_drift_with_same_distribution(self):
        """Test that identical distributions show no drift."""
        np.random.seed(42)
        ref = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 300),
            "feature2": np.random.normal(0, 1, 300),
        })
        # Same distribution
        np.random.seed(43)
        cur = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 300),
            "feature2": np.random.normal(0, 1, 300),
        })

        detector = DriftDetector(ref, drift_threshold=0.5)
        result = detector.check_drift(cur)

        # With same distribution, drift should not be detected
        assert result.is_drifted is False
