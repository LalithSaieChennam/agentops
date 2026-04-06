"""Tests for drift detection."""

import pytest
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

    def test_singleton_requires_reference_data(self):
        """Test that first call requires reference_data."""
        with pytest.raises(ValueError, match="reference_data required"):
            DriftDetector.get_instance()

    def test_update_reference(self):
        """Test updating reference data."""
        ref_data = pd.DataFrame({"a": [1, 2, 3]})
        detector = DriftDetector(ref_data)

        new_ref = pd.DataFrame({"a": [4, 5, 6]})
        detector.update_reference(new_ref)
        assert len(detector.reference_data) == 3
