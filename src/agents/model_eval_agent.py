"""Agent 2: Tracks model performance and detects degradation.

Computes accuracy, F1, precision, recall over a sliding window
and determines if the model needs retraining.
"""

import structlog
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.agents.state import AgentState
from src.monitoring.metrics_exporter import MODEL_ACCURACY, MODEL_F1_SCORE
from src.monitoring.performance_tracker import PerformanceTracker

logger = structlog.get_logger()

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm


def model_eval_agent(state: AgentState) -> AgentState:
    """Evaluate current model performance against baseline.

    1. Computes metrics over the sliding prediction window
    2. Checks for degradation against baseline thresholds
    3. Cross-references with drift data from Agent 1
    4. Uses LLM to decide if retraining is warranted
    """
    logger.info("model_eval_agent_started")

    try:
        tracker = PerformanceTracker.get_instance()
        degradation = tracker.is_degraded()

        # Update Prometheus
        if degradation.get("current_f1"):
            MODEL_F1_SCORE.set(degradation["current_f1"])
        if degradation.get("current_accuracy"):
            MODEL_ACCURACY.set(degradation["current_accuracy"])

        # LLM decides: is retraining needed?
        decision = _get_llm().invoke([HumanMessage(content=f"""
You are an MLOps evaluation agent. Based on these metrics, decide if
the model needs retraining. Reply with a JSON object.

Model Performance:
- Current F1: {degradation.get('current_f1', 'N/A')}
- Baseline F1: {degradation.get('baseline_f1', 'N/A')}
- F1 Drop: {degradation.get('f1_drop', 'N/A')}
- Confidence Mean: {degradation.get('confidence_mean', 'N/A')}

Data Drift Status (from previous agent):
- Drift Detected: {state.get('drift_detected', False)}
- Drift Score: {state.get('drift_score', 0)}
- Drifted Features: {state.get('drifted_features', [])}

Decision criteria:
- If F1 dropped > 5% AND drift detected -> RETRAIN (urgent)
- If F1 dropped > 5% but no drift -> RETRAIN (model degradation)
- If drift detected but F1 still good -> MONITOR (drift hasn't impacted yet)
- If neither -> NO ACTION

Provide your reasoning in 2-3 sentences and your decision.
""")])

        state["performance_degraded"] = degradation.get("degraded", False)
        state["current_f1"] = degradation.get("current_f1", 0)
        state["baseline_f1"] = degradation.get("baseline_f1", 0)
        state["f1_drop"] = degradation.get("f1_drop", 0)
        state["eval_summary"] = decision.content

        logger.info(
            "model_eval_agent_complete",
            degraded=degradation.get("degraded"),
            current_f1=degradation.get("current_f1"),
        )

    except Exception as e:
        logger.error("model_eval_agent_failed", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Model eval agent: {str(e)}"]
        state["performance_degraded"] = False

    return state
