"""Agent pipeline control endpoints."""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from src.agents.orchestrator import run_pipeline

router = APIRouter()


class PipelineStatus(BaseModel):
    status: str
    last_run: Optional[str] = None
    drift_detected: Optional[bool] = None
    current_f1: Optional[float] = None
    last_action: Optional[str] = None
    summary: Optional[str] = None


class PipelineTriggerResponse(BaseModel):
    message: str
    trigger_reason: str


# Store last pipeline result
_last_result = {}


@router.get("/agents/status", response_model=PipelineStatus)
async def get_agent_status():
    """Get current status of the agent pipeline."""
    if not _last_result:
        return PipelineStatus(status="idle")

    return PipelineStatus(
        status="complete",
        last_run=_last_result.get("triggered_at"),
        drift_detected=_last_result.get("drift_detected"),
        current_f1=_last_result.get("current_f1"),
        last_action=_last_result.get("deployment_action"),
        summary=_last_result.get("final_summary"),
    )


@router.post("/agents/trigger", response_model=PipelineTriggerResponse)
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """Manually trigger the agent pipeline."""
    background_tasks.add_task(_run_and_store, "manual")
    return PipelineTriggerResponse(
        message="Pipeline triggered",
        trigger_reason="manual",
    )


async def _run_and_store(reason: str):
    global _last_result
    _last_result = await run_pipeline(trigger_reason=reason)
