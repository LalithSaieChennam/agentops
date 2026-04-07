"""Tests for the Deployment Agent."""

from unittest.mock import MagicMock, patch

from src.agents.deployment_agent import _generate_final_summary, deployment_agent
from src.agents.state import AgentState


def _base_state(**overrides) -> AgentState:
    """Create a base AgentState with sensible defaults, then apply overrides."""
    state: AgentState = {
        "trigger_reason": "manual",
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


class TestDeploymentAgent:
    """Test deployment agent behavior."""

    def test_no_deployment_when_no_retraining(self):
        """Test agent skips deployment when no retraining occurred."""
        with patch("src.agents.deployment_agent._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(content="No action needed.")

            state = _base_state(retraining_triggered=False)
            result = deployment_agent(state)

            assert result["deployment_action"] == "none"
            assert result["deployment_summary"] == "No deployment needed."

    @patch("src.agents.deployment_agent.shutil")
    @patch("src.agents.deployment_agent.Path")
    def test_deploys_when_new_model_is_better(self, mock_path_cls, mock_shutil):
        """Test agent deploys when retrained model is better than current."""
        with patch("src.agents.deployment_agent._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(content="Model deployed successfully.")

            with patch("src.agents.deployment_agent.PerformanceTracker") as mock_tracker_cls:
                mock_tracker = MagicMock()
                mock_tracker_cls.get_instance.return_value = mock_tracker

                with patch("src.agents.deployment_agent.DEPLOYMENT_SWAPS"), \
                     patch("src.agents.deployment_agent.MODEL_VERSION"):

                    mock_prod_path = MagicMock()
                    mock_prod_path.exists.return_value = True
                    mock_path_cls.return_value = mock_prod_path

                    state = _base_state(
                        retraining_triggered=True,
                        new_model_f1=0.91,
                        current_f1=0.80,
                        baseline_f1=0.85,
                        new_model_path="models/best",
                        retraining_summary="DEPLOY — new model is significantly better.",
                    )

                    result = deployment_agent(state)

                    assert result["deployment_action"] == "swap"
                    assert "deployed_model_version" in result
                    mock_tracker.update_baseline.assert_called_once_with(0.91)

    def test_no_deploy_when_new_model_not_better(self):
        """Test agent skips deploy when new model doesn't meet criteria."""
        with patch("src.agents.deployment_agent._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(content="Not deploying — SKIP.")

            state = _base_state(
                retraining_triggered=True,
                new_model_f1=0.78,
                current_f1=0.80,
                baseline_f1=0.85,
                new_model_path="models/best",
                retraining_summary="SKIP — new model is worse.",
            )

            result = deployment_agent(state)
            assert result["deployment_action"] == "none"

    def test_error_handling_in_deployment(self):
        """Test agent handles errors gracefully during deployment."""
        with patch("src.agents.deployment_agent._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(content="Error occurred.")

            with patch("src.agents.deployment_agent.Path") as mock_path_cls:
                mock_path_cls.side_effect = Exception("Disk full")

                state = _base_state(
                    retraining_triggered=True,
                    new_model_f1=0.91,
                    current_f1=0.80,
                    new_model_path="models/best",
                    retraining_summary="DEPLOY it",
                )

                result = deployment_agent(state)
                assert result["deployment_action"] == "none"
                assert len(result["errors"]) > 0
                assert "Deployment agent" in result["errors"][0]

    def test_generate_final_summary_calls_llm(self):
        """Test that _generate_final_summary uses LLM."""
        with patch("src.agents.deployment_agent._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(
                content="Pipeline completed. No drift. Model healthy."
            )

            state = _base_state()
            _generate_final_summary(state)

            assert state["final_summary"] == "Pipeline completed. No drift. Model healthy."
            mock_llm.return_value.invoke.assert_called_once()

    def test_errors_accumulate(self):
        """Test that new errors append to existing errors list."""
        with patch("src.agents.deployment_agent._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(content="Summary")

            with patch("src.agents.deployment_agent.Path") as mock_path_cls:
                mock_path_cls.side_effect = Exception("Something broke")

                state = _base_state(
                    retraining_triggered=True,
                    new_model_f1=0.91,
                    current_f1=0.80,
                    new_model_path="models/best",
                    retraining_summary="DEPLOY",
                    errors=["Previous error"],
                )

                result = deployment_agent(state)
                assert len(result["errors"]) == 2
                assert result["errors"][0] == "Previous error"
