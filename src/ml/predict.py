"""Inference pipeline for ticket classification."""

from typing import Any, Dict

import structlog

from src.ml.data_processor import LABEL_NAMES, TicketDataProcessor
from src.ml.model import TicketClassifier

logger = structlog.get_logger()


class InferencePipeline:
    """End-to-end inference: text in → prediction out.

    Combines the tokenizer and model for single-call inference.
    """

    def __init__(self, model: TicketClassifier, processor: TicketDataProcessor):
        self.model = model
        self.processor = processor

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify a single support ticket.

        Args:
            text: Raw ticket text

        Returns:
            Dict with label, confidence, and all probabilities
        """
        encoded = self.processor.tokenize_single(text)
        label_id, confidence, probs = self.model.predict(
            encoded["input_ids"], encoded["attention_mask"]
        )

        result = {
            "label": LABEL_NAMES[label_id],
            "label_id": label_id,
            "confidence": round(confidence, 4),
            "probabilities": {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
        }

        logger.debug(
            "prediction_made",
            label=result["label"],
            confidence=result["confidence"],
        )

        return result

    def predict_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        """Classify multiple tickets.

        Args:
            texts: List of raw ticket texts

        Returns:
            List of prediction dicts
        """
        return [self.predict(text) for text in texts]
