"""Track model performance metrics over time."""

from collections import deque
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import structlog
from typing import Dict, Optional
from dataclasses import dataclass

from src.config import settings

logger = structlog.get_logger()


@dataclass
class PerformanceSnapshot:
    """Point-in-time model performance metrics."""
    timestamp: datetime
    accuracy: float
    f1_weighted: float
    precision_weighted: float
    recall_weighted: float
    sample_count: int
    prediction_confidence_mean: float


class PerformanceTracker:
    """Sliding window tracker for model performance in production.

    Collects predictions and ground truth labels, computes metrics
    over a rolling window, and detects performance degradation.
    """

    _instance = None

    def __init__(
        self,
        window_size: int = None,
        degradation_threshold: float = None,
        baseline_f1: float = None,
    ):
        self.window_size = window_size or settings.performance_window_size
        self.degradation_threshold = degradation_threshold or settings.degradation_threshold
        self.baseline_f1 = baseline_f1 or settings.baseline_f1

        # Rolling window of (prediction, ground_truth, confidence)
        self.predictions = deque(maxlen=self.window_size)
        self.history: list[PerformanceSnapshot] = []

    @classmethod
    def get_instance(cls, **kwargs) -> "PerformanceTracker":
        """Get or create the singleton PerformanceTracker instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def log_prediction(self, predicted: int, actual: int, confidence: float):
        """Log a single prediction with its ground truth."""
        self.predictions.append({
            "predicted": predicted,
            "actual": actual,
            "confidence": confidence,
            "timestamp": datetime.utcnow(),
        })

    def compute_metrics(self) -> Optional[PerformanceSnapshot]:
        """Compute current performance over the sliding window."""
        if len(self.predictions) < settings.min_samples_for_metrics:
            return None

        preds = [p["predicted"] for p in self.predictions]
        actuals = [p["actual"] for p in self.predictions]
        confidences = [p["confidence"] for p in self.predictions]

        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            accuracy=accuracy_score(actuals, preds),
            f1_weighted=f1_score(actuals, preds, average="weighted", zero_division=0),
            precision_weighted=precision_score(actuals, preds, average="weighted", zero_division=0),
            recall_weighted=recall_score(actuals, preds, average="weighted", zero_division=0),
            sample_count=len(preds),
            prediction_confidence_mean=sum(confidences) / len(confidences),
        )

        self.history.append(snapshot)
        return snapshot

    def is_degraded(self) -> Dict:
        """Check if model performance has degraded below threshold.

        Returns dict with degradation status and details.
        """
        snapshot = self.compute_metrics()
        if snapshot is None:
            return {"degraded": False, "reason": "insufficient_data"}

        f1_drop = self.baseline_f1 - snapshot.f1_weighted
        is_degraded = f1_drop > self.degradation_threshold

        result = {
            "degraded": is_degraded,
            "current_f1": snapshot.f1_weighted,
            "baseline_f1": self.baseline_f1,
            "f1_drop": f1_drop,
            "confidence_mean": snapshot.prediction_confidence_mean,
            "sample_count": snapshot.sample_count,
        }

        if is_degraded:
            logger.warning("model_degradation_detected", **result)

        return result

    def update_baseline(self, new_baseline_f1: float):
        """Update baseline after successful retraining."""
        self.baseline_f1 = new_baseline_f1
        self.predictions.clear()
        logger.info("baseline_updated", new_f1=new_baseline_f1)
