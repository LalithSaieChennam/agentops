"""Metrics info endpoint (Prometheus metrics are mounted at /metrics)."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class MetricsInfo(BaseModel):
    prometheus_endpoint: str
    available_metrics: list[str]


@router.get("/metrics/info", response_model=MetricsInfo)
async def metrics_info():
    """Return info about available Prometheus metrics."""
    return MetricsInfo(
        prometheus_endpoint="/metrics",
        available_metrics=[
            "agentops_predictions_total",
            "agentops_prediction_confidence",
            "agentops_prediction_latency_seconds",
            "agentops_model_f1_score",
            "agentops_model_accuracy",
            "agentops_model_version",
            "agentops_data_drift_score",
            "agentops_data_drift_detected",
            "agentops_drifted_features_count",
            "agentops_agent_runs_total",
            "agentops_retraining_total",
            "agentops_deployment_swaps_total",
            "agentops_agent_pipeline_duration_seconds",
        ],
    )
