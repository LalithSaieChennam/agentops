"""Tests for the LangGraph orchestrator."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.agents.state import AgentState
from src.agents.orchestrator import should_retrain, build_pipeline


class TestOrchestratorRouting:
    """Test orchestrator routing logic."""

    def test_should_retrain_when_degraded(self):
        """Test routing to retrain when performance is degraded."""
        state: AgentState = {
            "performance_degraded": True,
            "drift_detected": False,
        }
        assert should_retrain(state) == "retrain"

    def test_should_retrain_when_drifted(self):
        """Test routing to retrain when drift is detected."""
        state: AgentState = {
            "performance_degraded": False,
            "drift_detected": True,
        }
        assert should_retrain(state) == "retrain"

    def test_should_retrain_when_both(self):
        """Test routing to retrain when both degraded and drifted."""
        state: AgentState = {
            "performance_degraded": True,
            "drift_detected": True,
        }
        assert should_retrain(state) == "retrain"

    def test_should_deploy_when_neither(self):
        """Test routing to deploy when no issues detected."""
        state: AgentState = {
            "performance_degraded": False,
            "drift_detected": False,
        }
        assert should_retrain(state) == "deploy"


class TestPipelineConstruction:
    """Test that the LangGraph pipeline is constructed correctly."""

    def test_build_pipeline_returns_compiled_graph(self):
        """Test that build_pipeline returns a compilable graph."""
        pipeline = build_pipeline()
        # A compiled StateGraph should be invokable
        assert pipeline is not None
        assert hasattr(pipeline, "invoke") or hasattr(pipeline, "ainvoke")

    def test_pipeline_has_all_nodes(self):
        """Test that the pipeline contains all 4 agent nodes."""
        pipeline = build_pipeline()
        # The compiled graph's nodes should include our 4 agents + __start__ + __end__
        graph_nodes = pipeline.get_graph().nodes
        node_names = set(graph_nodes.keys())

        assert "data_quality" in node_names
        assert "model_eval" in node_names
        assert "retrain" in node_names
        assert "deploy" in node_names

    def test_pipeline_entry_point_is_data_quality(self):
        """Test that the pipeline starts with the data_quality agent."""
        pipeline = build_pipeline()
        graph = pipeline.get_graph()

        # The __start__ node should connect to data_quality
        start_edges = [e for e in graph.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "data_quality"

    def test_pipeline_ends_at_deploy(self):
        """Test that the pipeline ends after the deploy agent."""
        pipeline = build_pipeline()
        graph = pipeline.get_graph()

        # The deploy node should connect to __end__
        deploy_edges = [e for e in graph.edges if e.source == "deploy"]
        assert len(deploy_edges) == 1
        assert deploy_edges[0].target == "__end__"


class TestRunPipeline:
    """Test the run_pipeline async function."""

    @pytest.mark.asyncio
    async def test_run_pipeline_initializes_state(self):
        """Test that run_pipeline creates proper initial state."""
        mock_result = {
            "trigger_reason": "manual",
            "triggered_at": "2024-01-01",
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
            "final_summary": "All good.",
            "errors": [],
        }

        with patch("src.agents.orchestrator.pipeline") as mock_pipeline, \
             patch("src.agents.orchestrator.AGENT_RUNS_TOTAL") as mock_counter, \
             patch("src.agents.orchestrator.AGENT_PIPELINE_DURATION"), \
             patch("src.agents.orchestrator.log_pipeline_run"):

            mock_pipeline.ainvoke = AsyncMock(return_value=mock_result)

            from src.agents.orchestrator import run_pipeline
            result = await run_pipeline(trigger_reason="manual")

            assert result["trigger_reason"] == "manual"
            mock_counter.labels.assert_called_once_with(trigger_reason="manual")
