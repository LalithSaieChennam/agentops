"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and model is loaded."""
    from src.api.app import model

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        version="1.0.0",
    )
