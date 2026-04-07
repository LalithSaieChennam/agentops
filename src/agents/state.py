"""Shared state schema for the agent pipeline.

LangGraph agents communicate through this shared state object.
Each agent reads from it and writes its results back.
"""

from typing import Literal, TypedDict


class AgentState(TypedDict):
    """State shared across all agents in the pipeline.

    LangGraph passes this dict between nodes. Each agent
    reads what it needs and writes its output.
    """
    # Trigger info
    trigger_reason: Literal["scheduled", "manual", "alert"]
    triggered_at: str

    # Data Quality Agent output
    drift_detected: bool
    drift_score: float
    drifted_features: list[str]
    drift_report_summary: str

    # Model Evaluation Agent output
    performance_degraded: bool
    current_f1: float
    baseline_f1: float
    f1_drop: float
    eval_summary: str

    # Retraining Agent output
    retraining_triggered: bool
    new_model_f1: float
    new_model_path: str
    retraining_summary: str
    mlflow_run_id: str

    # Deployment Agent output
    deployment_action: Literal["swap", "rollback", "none"]
    deployed_model_version: str
    deployment_summary: str

    # Overall pipeline
    pipeline_decision: str  # LLM's reasoning about what to do
    final_summary: str
    errors: list[str]
