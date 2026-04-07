"""Evaluation metrics utilities for model assessment."""

from typing import Any, Dict

import structlog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.ml.data_processor import LABEL_NAMES

logger = structlog.get_logger()


def compute_metrics(predictions: list, labels: list) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Args:
        predictions: List of predicted label IDs
        labels: List of ground truth label IDs

    Returns:
        Dict with accuracy, F1, precision, recall, confusion matrix, and full report
    """
    target_names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]

    report = classification_report(
        labels, predictions,
        target_names=target_names,
        output_dict=True,
    )

    cm = confusion_matrix(labels, predictions)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1_per_class": {
            LABEL_NAMES[i]: report[LABEL_NAMES[i]]["f1-score"]
            for i in range(len(LABEL_NAMES))
        },
        "confusion_matrix": cm.tolist(),
        "full_report": report,
    }

    logger.info(
        "evaluation_complete",
        accuracy=round(metrics["accuracy"], 4),
        f1_weighted=round(metrics["f1_weighted"], 4),
    )

    return metrics


def compare_models(current_metrics: Dict, new_metrics: Dict) -> Dict[str, Any]:
    """Compare two models' metrics to decide on deployment.

    Args:
        current_metrics: Metrics from the currently deployed model
        new_metrics: Metrics from the newly trained model

    Returns:
        Dict with comparison results and recommendation
    """
    f1_improvement = new_metrics["f1_weighted"] - current_metrics["f1_weighted"]
    accuracy_improvement = new_metrics["accuracy"] - current_metrics["accuracy"]

    recommendation = "deploy" if f1_improvement > 0 else "skip"

    comparison = {
        "current_f1": current_metrics["f1_weighted"],
        "new_f1": new_metrics["f1_weighted"],
        "f1_improvement": f1_improvement,
        "current_accuracy": current_metrics["accuracy"],
        "new_accuracy": new_metrics["accuracy"],
        "accuracy_improvement": accuracy_improvement,
        "recommendation": recommendation,
    }

    logger.info("model_comparison", **comparison)
    return comparison
