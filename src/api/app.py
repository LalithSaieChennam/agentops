"""FastAPI application — serves predictions and agent controls."""

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from contextlib import asynccontextmanager
import structlog

from src.api.routes import predict, health, agents
from src.ml.model import TicketClassifier
from src.ml.data_processor import TicketDataProcessor
from src.storage.database import init_db

logger = structlog.get_logger()

# Global model instance
model: TicketClassifier = None
processor: TicketDataProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, processor

    # Initialize database tables
    try:
        init_db()
    except Exception as e:
        logger.warning("database_init_failed", error=str(e))

    # Load model
    model = TicketClassifier()
    try:
        model.load("models/production")
        logger.info("production_model_loaded")
    except Exception:
        logger.warning("no_production_model_found, using pretrained")

    processor = TicketDataProcessor()

    yield
    logger.info("shutting_down")


app = FastAPI(
    title="AgentOps",
    description="Agentic MLOps Pipeline — Autonomous model monitoring, retraining, and deployment",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Register routes
app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(agents.router, prefix="/api/v1", tags=["Agents"])
