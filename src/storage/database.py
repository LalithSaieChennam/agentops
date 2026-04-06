"""PostgreSQL connection, tables, and queries for AgentOps."""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
import structlog

from src.config import settings

logger = structlog.get_logger()

Base = declarative_base()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Prediction(Base):
    """Stores every prediction for monitoring and drift detection."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    input_text = Column(Text, nullable=False)
    predicted_label = Column(String(50), nullable=False)
    predicted_label_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON)
    actual_label = Column(String(50), nullable=True)
    actual_label_id = Column(Integer, nullable=True)
    model_version = Column(String(100), nullable=True)


class PipelineRun(Base):
    """Stores results of each agent pipeline run."""
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    trigger_reason = Column(String(50), nullable=False)
    drift_detected = Column(Boolean, default=False)
    drift_score = Column(Float, default=0.0)
    performance_degraded = Column(Boolean, default=False)
    current_f1 = Column(Float, nullable=True)
    retraining_triggered = Column(Boolean, default=False)
    new_model_f1 = Column(Float, nullable=True)
    deployment_action = Column(String(50), default="none")
    summary = Column(Text, nullable=True)
    errors = Column(JSON, nullable=True)
    duration_seconds = Column(Float, nullable=True)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("database_initialized")


def log_prediction(
    input_text: str,
    predicted_label: str,
    predicted_label_id: int,
    confidence: float,
    probabilities: dict,
    model_version: str = None,
):
    """Log a prediction to the database."""
    with SessionLocal() as session:
        prediction = Prediction(
            input_text=input_text,
            predicted_label=predicted_label,
            predicted_label_id=predicted_label_id,
            confidence=confidence,
            probabilities=probabilities,
            model_version=model_version,
        )
        session.add(prediction)
        session.commit()


def get_recent_predictions(limit: int = 1000) -> pd.DataFrame:
    """Fetch recent predictions as a DataFrame for drift detection."""
    with SessionLocal() as session:
        predictions = (
            session.query(Prediction)
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
            .all()
        )

        if not predictions:
            return pd.DataFrame()

        data = [
            {
                "input_text": p.input_text,
                "predicted_label": p.predicted_label,
                "predicted_label_id": p.predicted_label_id,
                "confidence": p.confidence,
                "timestamp": p.timestamp,
            }
            for p in predictions
        ]
        return pd.DataFrame(data)


def log_pipeline_run(state: dict, duration: float = None):
    """Log a pipeline run result to the database."""
    with SessionLocal() as session:
        run = PipelineRun(
            trigger_reason=state.get("trigger_reason", "unknown"),
            drift_detected=state.get("drift_detected", False),
            drift_score=state.get("drift_score", 0.0),
            performance_degraded=state.get("performance_degraded", False),
            current_f1=state.get("current_f1"),
            retraining_triggered=state.get("retraining_triggered", False),
            new_model_f1=state.get("new_model_f1"),
            deployment_action=state.get("deployment_action", "none"),
            summary=state.get("final_summary"),
            errors=state.get("errors", []),
            duration_seconds=duration,
        )
        session.add(run)
        session.commit()


def get_pipeline_history(limit: int = 10) -> list:
    """Get recent pipeline run history."""
    with SessionLocal() as session:
        runs = (
            session.query(PipelineRun)
            .order_by(PipelineRun.timestamp.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "trigger_reason": r.trigger_reason,
                "drift_detected": r.drift_detected,
                "drift_score": r.drift_score,
                "performance_degraded": r.performance_degraded,
                "current_f1": r.current_f1,
                "retraining_triggered": r.retraining_triggered,
                "new_model_f1": r.new_model_f1,
                "deployment_action": r.deployment_action,
                "summary": r.summary,
                "errors": r.errors,
                "duration_seconds": r.duration_seconds,
            }
            for r in runs
        ]
