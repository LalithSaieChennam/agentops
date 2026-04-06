"""Tests for performance tracking."""

import pytest
from src.monitoring.performance_tracker import PerformanceTracker


class TestPerformanceTracker:
    """Test the PerformanceTracker class."""

    def setup_method(self):
        """Reset singleton before each test."""
        PerformanceTracker.reset_instance()

    def test_log_prediction(self):
        """Test logging a single prediction."""
        tracker = PerformanceTracker(window_size=100, baseline_f1=0.85)
        tracker.log_prediction(predicted=1, actual=1, confidence=0.95)
        assert len(tracker.predictions) == 1

    def test_insufficient_data(self):
        """Test that metrics return None with insufficient data."""
        tracker = PerformanceTracker(window_size=100, baseline_f1=0.85)
        for i in range(10):
            tracker.log_prediction(predicted=1, actual=1, confidence=0.9)
        result = tracker.compute_metrics()
        assert result is None

    def test_compute_metrics_with_enough_data(self):
        """Test metric computation with sufficient data."""
        tracker = PerformanceTracker(window_size=100, baseline_f1=0.85)
        # Log 60 correct predictions
        for i in range(60):
            tracker.log_prediction(predicted=1, actual=1, confidence=0.9)

        snapshot = tracker.compute_metrics()
        assert snapshot is not None
        assert snapshot.accuracy == 1.0
        assert snapshot.f1_weighted == 1.0
        assert snapshot.sample_count == 60

    def test_degradation_detection(self):
        """Test that degradation is detected when F1 drops."""
        tracker = PerformanceTracker(window_size=100, baseline_f1=0.90)

        # Log mixed predictions (some wrong) to simulate degradation
        for i in range(60):
            predicted = 1 if i % 3 != 0 else 0  # ~33% error rate
            tracker.log_prediction(predicted=predicted, actual=1, confidence=0.7)

        result = tracker.is_degraded()
        assert result["degraded"] is True
        assert result["f1_drop"] > 0.05

    def test_no_degradation(self):
        """Test that no degradation is detected when model is performing well."""
        tracker = PerformanceTracker(window_size=100, baseline_f1=0.85)

        # Log mostly correct predictions
        for i in range(60):
            tracker.log_prediction(predicted=1, actual=1, confidence=0.95)

        result = tracker.is_degraded()
        assert result["degraded"] is False

    def test_update_baseline(self):
        """Test baseline update clears predictions."""
        tracker = PerformanceTracker(window_size=100, baseline_f1=0.85)
        tracker.log_prediction(predicted=1, actual=1, confidence=0.9)
        assert len(tracker.predictions) == 1

        tracker.update_baseline(0.90)
        assert tracker.baseline_f1 == 0.90
        assert len(tracker.predictions) == 0

    def test_sliding_window(self):
        """Test that the sliding window limits size."""
        tracker = PerformanceTracker(window_size=10, baseline_f1=0.85)
        for i in range(20):
            tracker.log_prediction(predicted=1, actual=1, confidence=0.9)
        assert len(tracker.predictions) == 10
