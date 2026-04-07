"""DistilBERT model wrapper for ticket classification."""

from pathlib import Path
from typing import Tuple

import structlog
import torch
import torch.nn.functional as fn
from transformers import DistilBertForSequenceClassification

from src.config import settings

logger = structlog.get_logger()


class TicketClassifier:
    """Wraps DistilBERT for support ticket classification.

    This class handles model loading, inference, and confidence scoring.
    """

    def __init__(
        self,
        model_name: str = None,
        num_labels: int = None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name or settings.model_name, num_labels=num_labels or settings.num_labels
        ).to(self.device)
        self.num_labels = num_labels or settings.num_labels

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[int, float, list]:
        """Run inference on tokenized input.

        Returns:
            Tuple of (predicted_label, confidence, all_probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = fn.softmax(outputs.logits, dim=-1)
            confidence, predicted = torch.max(probabilities, dim=-1)

        return (
            predicted.item(),
            confidence.item(),
            probabilities.squeeze().cpu().tolist(),
        )

    def save(self, path: str):
        """Save model to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        logger.info("model_saved", path=path)

    def load(self, path: str):
        """Load model from disk."""
        self.model = DistilBertForSequenceClassification.from_pretrained(path).to(self.device)
        logger.info("model_loaded", path=path)
