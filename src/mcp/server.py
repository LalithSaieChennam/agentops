"""MCP Server — exposes AgentOps to Claude, Cursor, and any MCP client.

Tools exposed:
- check_model_status: Get current model health
- check_drift: Run drift detection
- trigger_retraining: Manually trigger the pipeline
- predict_ticket: Classify a support ticket
- get_pipeline_history: View past pipeline runs
"""

from fastmcp import FastMCP

from src.agents.orchestrator import run_pipeline
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance_tracker import PerformanceTracker
from src.storage.database import get_pipeline_history as db_get_pipeline_history
from src.storage.database import get_recent_predictions

mcp = FastMCP("AgentOps", description="Agentic MLOps Pipeline — monitor, retrain, and deploy ML models")


@mcp.tool()
async def check_model_status() -> dict:
    """Check the current health and performance of the deployed model.

    Returns F1 score, accuracy, drift status, and whether
    the model is degraded.
    """
    tracker = PerformanceTracker.get_instance()
    degradation = tracker.is_degraded()

    drift_info = {"drift_detected": False, "drift_score": 0}
    try:
        detector = DriftDetector.get_instance()
        recent_data = get_recent_predictions(limit=500)
        if not recent_data.empty:
            drift = detector.check_drift(recent_data)
            drift_info = {
                "drift_detected": drift.is_drifted,
                "drift_score": drift.drift_score,
            }
    except Exception:
        pass

    return {
        "model_health": "degraded" if degradation.get("degraded") else "healthy",
        "current_f1": degradation.get("current_f1"),
        "baseline_f1": degradation.get("baseline_f1"),
        "drift_detected": drift_info["drift_detected"],
        "drift_score": drift_info["drift_score"],
        "prediction_count": degradation.get("sample_count", 0),
    }


@mcp.tool()
async def trigger_retraining(reason: str = "manual") -> dict:
    """Trigger the full agent pipeline: drift check -> evaluation -> retrain -> deploy.

    Args:
        reason: Why you're triggering this (e.g., "manual", "performance_drop")

    Returns the pipeline result including what actions were taken.
    """
    result = await run_pipeline(trigger_reason=reason)
    return {
        "drift_detected": result["drift_detected"],
        "performance_degraded": result["performance_degraded"],
        "retrained": result["retraining_triggered"],
        "deployed": result["deployment_action"],
        "new_f1": result.get("new_model_f1"),
        "summary": result["final_summary"],
    }


@mcp.tool()
async def predict_ticket(text: str) -> dict:
    """Classify a customer support ticket.

    Args:
        text: The ticket text to classify

    Returns predicted category and confidence.
    """
    from src.api.app import model, processor
    from src.ml.data_processor import LABEL_NAMES

    encoded = processor.tokenize_single(text)
    label_id, confidence, probs = model.predict(encoded["input_ids"], encoded["attention_mask"])

    return {
        "category": LABEL_NAMES[label_id],
        "confidence": round(confidence, 4),
        "all_probabilities": {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    }


@mcp.tool()
async def get_pipeline_history(limit: int = 10) -> list:
    """Get history of recent pipeline runs.

    Args:
        limit: Number of recent runs to return

    Returns list of pipeline run summaries.
    """
    return db_get_pipeline_history(limit=limit)


if __name__ == "__main__":
    mcp.run()
