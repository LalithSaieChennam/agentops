"""Health check endpoint."""

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

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
