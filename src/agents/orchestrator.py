"""LangGraph orchestrator — wires the 4 agents into a pipeline.

This is the brain of AgentOps. It defines the graph of agents,
the conditional edges (which agent runs next based on state),
and the overall execution flow.
"""

from datetime import datetime

import structlog
from langgraph.graph import END, StateGraph

from src.agents.data_quality_agent import data_quality_agent
from src.agents.deployment_agent import deployment_agent
from src.agents.model_eval_agent import model_eval_agent
from src.agents.retraining_agent import retraining_agent
from src.agents.state import AgentState
from src.monitoring.metrics_exporter import AGENT_PIPELINE_DURATION, AGENT_RUNS_TOTAL
from src.storage.database import log_pipeline_run

logger = structlog.get_logger()


def should_retrain(state: AgentState) -> str:
    """After evaluation: retrain if degraded or drifted, else skip to deploy."""
    if state.get("performance_degraded") or state.get("drift_detected"):
        return "retrain"
    return "deploy"


def build_pipeline() -> StateGraph:
    """Construct the LangGraph agent pipeline.

    Flow:
        data_quality -> model_eval -> (retrain if needed) -> deploy -> END

    Returns a compiled LangGraph that can be invoked with initial state.
    """
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("data_quality", data_quality_agent)
    workflow.add_node("model_eval", model_eval_agent)
    workflow.add_node("retrain", retraining_agent)
    workflow.add_node("deploy", deployment_agent)

    # Define the flow
    workflow.set_entry_point("data_quality")
    workflow.add_edge("data_quality", "model_eval")
    workflow.add_conditional_edges("model_eval", should_retrain, {
        "retrain": "retrain",
        "deploy": "deploy",
    })
    workflow.add_edge("retrain", "deploy")
    workflow.add_edge("deploy", END)

    return workflow.compile()


# Build once, reuse
pipeline = build_pipeline()


async def run_pipeline(trigger_reason: str = "scheduled") -> AgentState:
    """Execute the full agent pipeline.

    Args:
        trigger_reason: What triggered this run (scheduled/manual/alert)

    Returns:
        Final AgentState with all agent outputs
    """
    import time
    start = time.time()

    AGENT_RUNS_TOTAL.labels(trigger_reason=trigger_reason).inc()

    initial_state: AgentState = {
        "trigger_reason": trigger_reason,
        "triggered_at": datetime.utcnow().isoformat(),
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

    logger.info("pipeline_started", trigger=trigger_reason)
    result = await pipeline.ainvoke(initial_state)

    duration = time.time() - start
    AGENT_PIPELINE_DURATION.observe(duration)

    # Log to database
    try:
        log_pipeline_run(result, duration=duration)
    except Exception as e:
        logger.error("failed_to_log_pipeline_run", error=str(e))

    logger.info(
        "pipeline_complete",
        duration=round(duration, 2),
        drift_detected=result["drift_detected"],
        retrained=result["retraining_triggered"],
        deployed=result["deployment_action"],
    )

    return result
