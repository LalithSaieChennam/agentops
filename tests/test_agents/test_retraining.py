"""Tests for the Retraining Agent."""

import pytest
from unittest.mock import patch, MagicMock
from src.agents.retraining_agent import retraining_agent
from src.agents.state import AgentState


def _base_state(**overrides) -> AgentState:
    """Create a base state for retraining agent tests."""
    state: AgentState = {
        "trigger_reason": "scheduled",
        "triggered_at": "2024-01-01T00:00:00",
        "drift_detected": False,
        "drift_score": 0.0,
        "drifted_features": [],
        "drift_report_summary": "",
        "performance_degraded": False,
        "current_f1": 0.88,
        "baseline_f1": 0.85,
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


class TestRetrainingAgent:
    """Test retraining agent decision logic."""

    def test_skip_when_no_issues(self):
        """Test agent skips retraining when no degradation or drift."""
        state = _base_state()
        result = retraining_agent(state)

        assert result["retraining_triggered"] is False
        assert "No retraining needed" in result["retraining_summary"]

    def test_triggers_when_degraded(self):
        """Test agent triggers retraining when performance is degraded."""
        with patch("src.agents.retraining_agent.TicketDataProcessor") as mock_proc_cls, \
             patch("src.agents.retraining_agent.TicketClassifier") as mock_model_cls, \
             patch("src.agents.retraining_agent.Trainer") as mock_trainer_cls, \
             patch("src.agents.retraining_agent.llm") as mock_llm, \
             patch("src.agents.retraining_agent.RETRAINING_RUNS"):

            # Setup mocks
            mock_proc = MagicMock()
            mock_proc.load_and_prepare.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_proc_cls.return_value = mock_proc

            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {
                "best_f1": 0.90,
                "model_path": "models/best",
                "mlflow_run_id": "run-123",
            }
            mock_trainer_cls.return_value = mock_trainer

            mock_llm.invoke.return_value = MagicMock(content="DEPLOY — new model is better.")

            state = _base_state(performance_degraded=True, current_f1=0.78, baseline_f1=0.88)
            result = retraining_agent(state)

            assert result["retraining_triggered"] is True
            assert result["new_model_f1"] == 0.90
            assert result["new_model_path"] == "models/best"
            assert "DEPLOY" in result["retraining_summary"]

    def test_triggers_when_drifted(self):
        """Test agent triggers retraining when drift is detected."""
        with patch("src.agents.retraining_agent.TicketDataProcessor") as mock_proc_cls, \
             patch("src.agents.retraining_agent.TicketClassifier") as mock_model_cls, \
             patch("src.agents.retraining_agent.Trainer") as mock_trainer_cls, \
             patch("src.agents.retraining_agent.llm") as mock_llm, \
             patch("src.agents.retraining_agent.RETRAINING_RUNS"):

            mock_proc = MagicMock()
            mock_proc.load_and_prepare.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_proc_cls.return_value = mock_proc

            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {
                "best_f1": 0.87,
                "model_path": "models/best",
                "mlflow_run_id": "run-456",
            }
            mock_trainer_cls.return_value = mock_trainer

            mock_llm.invoke.return_value = MagicMock(content="DEPLOY — model adapted to new distribution.")

            state = _base_state(drift_detected=True, drift_score=0.45)
            result = retraining_agent(state)

            assert result["retraining_triggered"] is True
            assert result["new_model_f1"] == 0.87

    def test_loads_existing_model_for_finetuning(self):
        """Test agent tries to load existing model weights before training."""
        with patch("src.agents.retraining_agent.TicketDataProcessor") as mock_proc_cls, \
             patch("src.agents.retraining_agent.TicketClassifier") as mock_model_cls, \
             patch("src.agents.retraining_agent.Trainer") as mock_trainer_cls, \
             patch("src.agents.retraining_agent.llm") as mock_llm, \
             patch("src.agents.retraining_agent.RETRAINING_RUNS"):

            mock_proc = MagicMock()
            mock_proc.load_and_prepare.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_proc_cls.return_value = mock_proc

            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {"best_f1": 0.89, "model_path": "models/best", "mlflow_run_id": "run-789"}
            mock_trainer_cls.return_value = mock_trainer

            mock_llm.invoke.return_value = MagicMock(content="DEPLOY")

            state = _base_state(performance_degraded=True)
            retraining_agent(state)

            # Model.load should have been called to load existing weights
            mock_model.load.assert_called_once_with("models/best")

    def test_falls_back_to_pretrained_if_no_existing(self):
        """Test agent falls back to pretrained when no existing model found."""
        with patch("src.agents.retraining_agent.TicketDataProcessor") as mock_proc_cls, \
             patch("src.agents.retraining_agent.TicketClassifier") as mock_model_cls, \
             patch("src.agents.retraining_agent.Trainer") as mock_trainer_cls, \
             patch("src.agents.retraining_agent.llm") as mock_llm, \
             patch("src.agents.retraining_agent.RETRAINING_RUNS"):

            mock_proc = MagicMock()
            mock_proc.load_and_prepare.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_proc_cls.return_value = mock_proc

            mock_model = MagicMock()
            mock_model.load.side_effect = Exception("No model found")
            mock_model_cls.return_value = mock_model

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {"best_f1": 0.85, "model_path": "models/best", "mlflow_run_id": "run-000"}
            mock_trainer_cls.return_value = mock_trainer

            mock_llm.invoke.return_value = MagicMock(content="DEPLOY")

            state = _base_state(performance_degraded=True)
            result = retraining_agent(state)

            # Should still succeed despite load failure
            assert result["retraining_triggered"] is True
            assert result["new_model_f1"] == 0.85

    def test_error_handling(self):
        """Test agent handles training errors gracefully."""
        with patch("src.agents.retraining_agent.TicketDataProcessor") as mock_proc_cls, \
             patch("src.agents.retraining_agent.RETRAINING_RUNS"):

            mock_proc_cls.side_effect = Exception("Data loading failed")

            state = _base_state(performance_degraded=True)
            result = retraining_agent(state)

            assert result["retraining_triggered"] is False
            assert len(result["errors"]) > 0
            assert "Retraining agent" in result["errors"][0]

    def test_increments_retraining_counter(self):
        """Test that the Prometheus retraining counter is incremented."""
        with patch("src.agents.retraining_agent.TicketDataProcessor") as mock_proc_cls, \
             patch("src.agents.retraining_agent.TicketClassifier") as mock_model_cls, \
             patch("src.agents.retraining_agent.Trainer") as mock_trainer_cls, \
             patch("src.agents.retraining_agent.llm") as mock_llm, \
             patch("src.agents.retraining_agent.RETRAINING_RUNS") as mock_counter:

            mock_proc = MagicMock()
            mock_proc.load_and_prepare.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_proc_cls.return_value = mock_proc
            mock_model_cls.return_value = MagicMock()
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {"best_f1": 0.90, "model_path": "models/best", "mlflow_run_id": "x"}
            mock_trainer_cls.return_value = mock_trainer
            mock_llm.invoke.return_value = MagicMock(content="DEPLOY")

            state = _base_state(performance_degraded=True)
            retraining_agent(state)

            mock_counter.inc.assert_called_once()
