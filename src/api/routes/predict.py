"""Prediction endpoint with automatic monitoring."""

import time
from fastapi import APIRouter
from pydantic import BaseModel

from src.ml.data_processor import LABEL_NAMES
from src.monitoring.metrics_exporter import (
    PREDICTION_COUNT, PREDICTION_CONFIDENCE, PREDICTION_LATENCY,
)

router = APIRouter()


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: dict


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Classify a support ticket and log the prediction for monitoring."""
    from src.api.app import model, processor

    start = time.time()

    # Tokenize
    encoded = processor.tokenize_single(request.text)

    # Predict
    label_id, confidence, probs = model.predict(
        encoded["input_ids"], encoded["attention_mask"]
    )

    label_name = LABEL_NAMES[label_id]
    latency = time.time() - start

    # Update Prometheus metrics
    PREDICTION_COUNT.labels(predicted_class=label_name).inc()
    PREDICTION_CONFIDENCE.observe(confidence)
    PREDICTION_LATENCY.observe(latency)

    # Log prediction to database (async-safe)
    try:
        from src.storage.database import log_prediction
        log_prediction(
            input_text=request.text,
            predicted_label=label_name,
            predicted_label_id=label_id,
            confidence=confidence,
            probabilities={LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
        )
    except Exception:
        pass  # Don't fail predictions if DB logging fails

    return PredictResponse(
        label=label_name,
        label_id=label_id,
        confidence=round(confidence, 4),
        probabilities={LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    )
