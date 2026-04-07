"""Export metrics to Prometheus for Grafana dashboards."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Prediction metrics
PREDICTION_COUNT = Counter(
    "agentops_predictions_total",
    "Total predictions made",
    ["predicted_class"],
)

PREDICTION_CONFIDENCE = Histogram(
    "agentops_prediction_confidence",
    "Prediction confidence distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

PREDICTION_LATENCY = Histogram(
    "agentops_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Model performance metrics
MODEL_F1_SCORE = Gauge("agentops_model_f1_score", "Current model F1 score")
MODEL_ACCURACY = Gauge("agentops_model_accuracy", "Current model accuracy")
MODEL_VERSION = Info("agentops_model_version", "Current deployed model version")

# Drift metrics
DATA_DRIFT_SCORE = Gauge("agentops_data_drift_score", "Current data drift score")
DATA_DRIFT_DETECTED = Gauge("agentops_data_drift_detected", "Whether drift is detected (0/1)")
DRIFTED_FEATURES_COUNT = Gauge("agentops_drifted_features_count", "Number of drifted features")

# Agent metrics
AGENT_RUNS_TOTAL = Counter(
    "agentops_agent_runs_total",
    "Total agent pipeline runs",
    ["trigger_reason"],
)
RETRAINING_RUNS = Counter("agentops_retraining_total", "Total retraining runs")
DEPLOYMENT_SWAPS = Counter("agentops_deployment_swaps_total", "Total model swaps")
AGENT_PIPELINE_DURATION = Histogram(
    "agentops_agent_pipeline_duration_seconds",
    "Duration of full agent pipeline run",
)
