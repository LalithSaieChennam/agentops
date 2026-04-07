"""Agent 1: Monitors incoming data for drift.

This agent fetches recent production data, compares it against
the reference distribution, and reports whether drift occurred.
"""

import structlog
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.agents.state import AgentState
from src.config import settings
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.metrics_exporter import DATA_DRIFT_DETECTED, DATA_DRIFT_SCORE, DRIFTED_FEATURES_COUNT
from src.storage.database import get_recent_predictions

logger = structlog.get_logger()

_llm = None


def _get_llm():
    """Lazy-initialize the LLM to avoid import-time API key validation."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm


def data_quality_agent(state: AgentState) -> AgentState:
    """Check production data for drift against reference distribution.

    1. Fetches last N predictions from PostgreSQL
    2. Runs Evidently drift detection against training reference
    3. Uses LLM to interpret results and generate summary
    4. Updates Prometheus metrics
    """
    logger.info("data_quality_agent_started")

    try:
        # Fetch recent production data
        recent_data = get_recent_predictions(limit=settings.drift_check_batch_size)

        if recent_data.empty:
            state["drift_detected"] = False
            state["drift_score"] = 0.0
            state["drifted_features"] = []
            state["drift_report_summary"] = "No recent data available for drift analysis."
            return state

        # Run drift detection
        detector = DriftDetector.get_instance()
        drift_report = detector.check_drift(recent_data)

        # Update Prometheus metrics
        DATA_DRIFT_SCORE.set(drift_report.drift_score)
        DATA_DRIFT_DETECTED.set(1 if drift_report.is_drifted else 0)
        DRIFTED_FEATURES_COUNT.set(len(drift_report.drifted_columns))

        # Use LLM to interpret the drift results
        interpretation = _get_llm().invoke([HumanMessage(content=f"""
You are an MLOps monitoring agent. Analyze this data drift report and provide
a concise summary (2-3 sentences) of what's happening:

- Dataset drift detected: {drift_report.is_drifted}
- Overall drift score: {drift_report.drift_score:.3f}
- Drifted columns: {drift_report.drifted_columns}
- Column-level scores: {drift_report.column_scores}

Focus on: Is this concerning? What kind of drift is this?
What might be causing it in a customer support ticket context?
""")])

        state["drift_detected"] = drift_report.is_drifted
        state["drift_score"] = drift_report.drift_score
        state["drifted_features"] = drift_report.drifted_columns
        state["drift_report_summary"] = interpretation.content

        logger.info(
            "data_quality_agent_complete",
            drift_detected=drift_report.is_drifted,
            drift_score=drift_report.drift_score,
        )

    except Exception as e:
        logger.error("data_quality_agent_failed", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Data quality agent: {str(e)}"]
        state["drift_detected"] = False

    return state
