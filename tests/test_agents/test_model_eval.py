"""Tests for the Model Evaluation Agent."""

import pytest
from unittest.mock import patch, MagicMock
from src.agents.state import AgentState
from src.agents.model_eval_agent import model_eval_agent


def _base_state(**overrides) -> AgentState:
    """Create a base state for model eval agent tests."""
    state: AgentState = {
        "trigger_reason": "manual",
        "triggered_at": "2024-01-01T00:00:00",
        "drift_detected": False,
        "drift_score": 0.0,
        "drifted_features": [],
        "drift_report_summary": "",
        "performance_degraded": False,
        "current_f1": 0.0,
        "baseline_f1": 0.0,
        "f1_drop": 0.0,
        "eval_summary": "",
        "retraining_triggered": False,
        "new_model_f1": 0.0,
        "new_model_path": "",
        "retraining_summary": "",
        "mlflow_run_id": "",
        "deployment_action": "none",
        "deployed_model_version": "",
        "deployment_summary": "",
        "pipeline_decision": "",
        "final_summary": "",
        "errors": [],
    }
    state.update(overrides)
    return state


class TestModelEvalAgent:
    """Test model evaluation agent behavior."""

    def test_state_schema_has_all_eval_fields(self):
        """Test that AgentState has all required eval fields."""
        state = _base_state(current_f1=0.88, baseline_f1=0.85, f1_drop=-0.03)

        assert state["current_f1"] == 0.88
        assert state["performance_degraded"] is False

    def test_detects_degradation(self):
        """Test agent detects model performance degradation."""
        with patch("src.agents.model_eval_agent.PerformanceTracker") as mock_tracker_cls, \
             patch("src.agents.model_eval_agent.llm") as mock_llm, \
             patch("src.agents.model_eval_agent.MODEL_F1_SCORE"), \
             patch("src.agents.model_eval_agent.MODEL_ACCURACY"):

            mock_tracker = MagicMock()
            mock_tracker.is_degraded.return_value = {
                "degraded": True,
                "current_f1": 0.78,
                "baseline_f1": 0.88,
                "f1_drop": 0.10,
                "confidence_mean": 0.65,
                "sample_count": 500,
            }
            mock_tracker_cls.get_instance.return_value = mock_tracker
            mock_llm.invoke.return_value = MagicMock(content="RETRAIN — F1 dropped 10%.")

            state = _base_state(drift_detected=True, drift_score=0.4)
            result = model_eval_agent(state)

            assert result["performance_degraded"] is True
            assert result["current_f1"] == 0.78
            assert result["f1_drop"] == 0.10
            assert "RETRAIN" in result["eval_summary"]

    def test_no_degradation_when_healthy(self):
        """Test agent reports no degradation when model is performing well."""
        with patch("src.agents.model_eval_agent.PerformanceTracker") as mock_tracker_cls, \
             patch("src.agents.model_eval_agent.llm") as mock_llm, \
             patch("src.agents.model_eval_agent.MODEL_F1_SCORE"), \
             patch("src.agents.model_eval_agent.MODEL_ACCURACY"):

            mock_tracker = MagicMock()
            mock_tracker.is_degraded.return_value = {
                "degraded": False,
                "current_f1": 0.89,
                "baseline_f1": 0.88,
                "f1_drop": -0.01,
                "confidence_mean": 0.92,
                "sample_count": 500,
            }
            mock_tracker_cls.get_instance.return_value = mock_tracker
            mock_llm.invoke.return_value = MagicMock(content="NO ACTION — model is healthy.")

            state = _base_state()
            result = model_eval_agent(state)

            assert result["performance_degraded"] is False
            assert result["current_f1"] == 0.89

    def test_cross_references_drift_data(self):
        """Test that the LLM prompt includes drift data from Agent 1."""
        with patch("src.agents.model_eval_agent.PerformanceTracker") as mock_tracker_cls, \
             patch("src.agents.model_eval_agent.llm") as mock_llm, \
             patch("src.agents.model_eval_agent.MODEL_F1_SCORE"), \
             patch("src.agents.model_eval_agent.MODEL_ACCURACY"):

            mock_tracker = MagicMock()
            mock_tracker.is_degraded.return_value = {
                "degraded": False, "current_f1": 0.87, "baseline_f1": 0.88,
                "f1_drop": 0.01, "confidence_mean": 0.9, "sample_count": 300,
            }
            mock_tracker_cls.get_instance.return_value = mock_tracker
            mock_llm.invoke.return_value = MagicMock(content="MONITOR — drift present but F1 stable.")

            state = _base_state(drift_detected=True, drift_score=0.35, drifted_features=["confidence"])
            result = model_eval_agent(state)

            # Check that the LLM was called with drift info in the prompt
            call_args = mock_llm.invoke.call_args[0][0][0].content
            assert "True" in call_args  # drift_detected
            assert "0.35" in call_args  # drift_score

    def test_updates_prometheus_metrics(self):
        """Test that Prometheus metrics are updated."""
        with patch("src.agents.model_eval_agent.PerformanceTracker") as mock_tracker_cls, \
             patch("src.agents.model_eval_agent.llm") as mock_llm, \
             patch("src.agents.model_eval_agent.MODEL_F1_SCORE") as mock_f1, \
             patch("src.agents.model_eval_agent.MODEL_ACCURACY"):

            mock_tracker = MagicMock()
            mock_tracker.is_degraded.return_value = {
                "degraded": False, "current_f1": 0.89, "baseline_f1": 0.88,
                "f1_drop": -0.01, "confidence_mean": 0.91, "sample_count": 500,
            }
            mock_tracker_cls.get_instance.return_value = mock_tracker
            mock_llm.invoke.return_value = MagicMock(content="Healthy.")

            state = _base_state()
            model_eval_agent(state)

            mock_f1.set.assert_called_once_with(0.89)

    def test_error_handling(self):
        """Test agent handles errors gracefully."""
        with patch("src.agents.model_eval_agent.PerformanceTracker") as mock_tracker_cls:
            mock_tracker_cls.get_instance.side_effect = Exception("Tracker unavailable")

            state = _base_state()
            result = model_eval_agent(state)

            assert result["performance_degraded"] is False
            assert len(result["errors"]) > 0
            assert "Model eval agent" in result["errors"][0]
