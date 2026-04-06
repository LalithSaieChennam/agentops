"""Tests for the Data Quality Agent."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.agents.state import AgentState


def _base_state(**overrides) -> AgentState:
    """Create a base state for data quality agent tests."""
    state: AgentState = {
        "trigger_reason": "manual",
        "triggered_at": "2024-01-01T00:00:00",
        "drift_detected": False,
        "drift_score": 0.0,
        "drifted_features": [],
        "drift_report_summary": "",
        "errors": [],
    }
    state.update(overrides)
    return state


class TestDataQualityAgent:
    """Test data quality agent behavior."""

    def test_handles_empty_data(self):
        """Test agent handles empty production data gracefully."""
        from src.agents.data_quality_agent import data_quality_agent

        with patch("src.agents.data_quality_agent.get_recent_predictions") as mock_get:
            mock_get.return_value = pd.DataFrame()

            state = _base_state()
            result = data_quality_agent(state)

            assert result["drift_detected"] is False
            assert result["drift_score"] == 0.0
            assert "No recent data" in result["drift_report_summary"]

    def test_detects_drift_when_present(self):
        """Test agent correctly reports drift when detected by Evidently."""
        from src.agents.data_quality_agent import data_quality_agent

        mock_drift_report = MagicMock()
        mock_drift_report.is_drifted = True
        mock_drift_report.drift_score = 0.65
        mock_drift_report.drifted_columns = ["confidence", "predicted_label_id"]
        mock_drift_report.column_scores = {"confidence": 0.8, "predicted_label_id": 0.7}

        with patch("src.agents.data_quality_agent.get_recent_predictions") as mock_get, \
             patch("src.agents.data_quality_agent.DriftDetector") as mock_detector_cls, \
             patch("src.agents.data_quality_agent.llm") as mock_llm, \
             patch("src.agents.data_quality_agent.DATA_DRIFT_SCORE"), \
             patch("src.agents.data_quality_agent.DATA_DRIFT_DETECTED"), \
             patch("src.agents.data_quality_agent.DRIFTED_FEATURES_COUNT"):

            mock_get.return_value = pd.DataFrame({"confidence": [0.5, 0.6], "predicted_label_id": [1, 2]})
            mock_detector = MagicMock()
            mock_detector.check_drift.return_value = mock_drift_report
            mock_detector_cls.get_instance.return_value = mock_detector
            mock_llm.invoke.return_value = MagicMock(content="Significant drift detected in confidence and label distribution.")

            state = _base_state()
            result = data_quality_agent(state)

            assert result["drift_detected"] is True
            assert result["drift_score"] == 0.65
            assert "confidence" in result["drifted_features"]
            assert "predicted_label_id" in result["drifted_features"]

    def test_no_drift_reported_when_clean(self):
        """Test agent reports no drift when data is stable."""
        from src.agents.data_quality_agent import data_quality_agent

        mock_drift_report = MagicMock()
        mock_drift_report.is_drifted = False
        mock_drift_report.drift_score = 0.05
        mock_drift_report.drifted_columns = []
        mock_drift_report.column_scores = {"confidence": 0.02}

        with patch("src.agents.data_quality_agent.get_recent_predictions") as mock_get, \
             patch("src.agents.data_quality_agent.DriftDetector") as mock_detector_cls, \
             patch("src.agents.data_quality_agent.llm") as mock_llm, \
             patch("src.agents.data_quality_agent.DATA_DRIFT_SCORE"), \
             patch("src.agents.data_quality_agent.DATA_DRIFT_DETECTED"), \
             patch("src.agents.data_quality_agent.DRIFTED_FEATURES_COUNT"):

            mock_get.return_value = pd.DataFrame({"confidence": [0.9, 0.91]})
            mock_detector = MagicMock()
            mock_detector.check_drift.return_value = mock_drift_report
            mock_detector_cls.get_instance.return_value = mock_detector
            mock_llm.invoke.return_value = MagicMock(content="No significant drift detected.")

            state = _base_state()
            result = data_quality_agent(state)

            assert result["drift_detected"] is False
            assert result["drift_score"] == 0.05
            assert result["drifted_features"] == []

    def test_updates_prometheus_metrics(self):
        """Test that Prometheus metrics are updated during drift check."""
        from src.agents.data_quality_agent import data_quality_agent

        mock_drift_report = MagicMock()
        mock_drift_report.is_drifted = True
        mock_drift_report.drift_score = 0.4
        mock_drift_report.drifted_columns = ["col1"]
        mock_drift_report.column_scores = {"col1": 0.8}

        with patch("src.agents.data_quality_agent.get_recent_predictions") as mock_get, \
             patch("src.agents.data_quality_agent.DriftDetector") as mock_detector_cls, \
             patch("src.agents.data_quality_agent.llm") as mock_llm, \
             patch("src.agents.data_quality_agent.DATA_DRIFT_SCORE") as mock_drift_gauge, \
             patch("src.agents.data_quality_agent.DATA_DRIFT_DETECTED") as mock_detected_gauge, \
             patch("src.agents.data_quality_agent.DRIFTED_FEATURES_COUNT") as mock_count_gauge:

            mock_get.return_value = pd.DataFrame({"col1": [1, 2]})
            mock_detector = MagicMock()
            mock_detector.check_drift.return_value = mock_drift_report
            mock_detector_cls.get_instance.return_value = mock_detector
            mock_llm.invoke.return_value = MagicMock(content="Drift found.")

            state = _base_state()
            data_quality_agent(state)

            mock_drift_gauge.set.assert_called_once_with(0.4)
            mock_detected_gauge.set.assert_called_once_with(1)
            mock_count_gauge.set.assert_called_once_with(1)

    def test_error_handling(self):
        """Test agent handles errors gracefully and logs them to state."""
        from src.agents.data_quality_agent import data_quality_agent

        with patch("src.agents.data_quality_agent.get_recent_predictions") as mock_get:
            mock_get.side_effect = Exception("Database connection failed")

            state = _base_state()
            result = data_quality_agent(state)

            assert result["drift_detected"] is False
            assert len(result["errors"]) > 0
            assert "Data quality agent" in result["errors"][0]

    def test_llm_interprets_drift_results(self):
        """Test that the LLM is called to interpret drift results."""
        from src.agents.data_quality_agent import data_quality_agent

        mock_drift_report = MagicMock()
        mock_drift_report.is_drifted = True
        mock_drift_report.drift_score = 0.5
        mock_drift_report.drifted_columns = ["col1"]
        mock_drift_report.column_scores = {"col1": 0.9}

        with patch("src.agents.data_quality_agent.get_recent_predictions") as mock_get, \
             patch("src.agents.data_quality_agent.DriftDetector") as mock_detector_cls, \
             patch("src.agents.data_quality_agent.llm") as mock_llm, \
             patch("src.agents.data_quality_agent.DATA_DRIFT_SCORE"), \
             patch("src.agents.data_quality_agent.DATA_DRIFT_DETECTED"), \
             patch("src.agents.data_quality_agent.DRIFTED_FEATURES_COUNT"):

            mock_get.return_value = pd.DataFrame({"col1": [1, 2]})
            mock_detector = MagicMock()
            mock_detector.check_drift.return_value = mock_drift_report
            mock_detector_cls.get_instance.return_value = mock_detector
            mock_llm.invoke.return_value = MagicMock(content="Technical ticket surge detected.")

            state = _base_state()
            result = data_quality_agent(state)

            # LLM was called exactly once
            mock_llm.invoke.assert_called_once()
            # The summary came from the LLM
            assert result["drift_report_summary"] == "Technical ticket surge detected."
