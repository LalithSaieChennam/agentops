# AgentOps — Agentic MLOps Pipeline with MCP Integration

## Complete End-to-End Project Guide

---

## 1. What You're Building

AgentOps is a multi-agent system that automates the ML model lifecycle. It monitors a deployed ML model in production, detects when the data or model performance starts drifting, automatically retrains the model when quality degrades, deploys the new model with zero downtime, and exposes the entire system as an MCP server so any AI tool (Claude, Cursor, etc.) can query and control it.

You're building 4 autonomous agents orchestrated through LangGraph:

- **Data Quality Agent** — monitors incoming data for statistical drift
- **Model Evaluation Agent** — tracks prediction accuracy, precision, recall over time
- **Retraining Agent** — fine-tunes a DistilBERT model when performance drops
- **Deployment Agent** — handles model swaps, A/B testing, and rollback

The demo use case: **Customer support ticket classification**. You train a model to classify incoming support tickets into categories (billing, technical, account, feature request). When the distribution of tickets changes over time (drift), or the model starts misclassifying, the agents kick in and fix it autonomously.

---

## 2. Why This Use Case

Customer support classification is perfect because:

- Easy to get training data (public datasets + synthetic generation)
- Drift is natural and easy to simulate (new product launches change ticket patterns)
- Fine-tuning DistilBERT on text classification is well-documented
- The business value is obvious to any interviewer
- It maps to real production systems at companies like Capital One, Optum, etc.

---

## 3. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Agent Orchestration | LangGraph | Already on your resume, graph-based multi-agent workflows |
| LLM Integration | OpenAI GPT-4o-mini | Agent reasoning, decision-making |
| ML Model | DistilBERT (PyTorch + HuggingFace) | Lightweight, fast fine-tuning, fills your PyTorch gap |
| Drift Detection | Evidently AI | Industry-standard, open-source, Python-native |
| Experiment Tracking | MLflow | Logs experiments, model versions, metrics |
| API Layer | FastAPI | Already on your resume, serves predictions + MCP |
| MCP Server | FastMCP / fastapi-mcp | Exposes system to Claude, Cursor, any MCP client |
| Monitoring | Prometheus + Grafana | Real-time dashboards, alerting |
| Database | PostgreSQL | Metadata store, prediction logs |
| Cache | Redis | Model caching, prediction caching |
| Containerization | Docker + Docker Compose | Everything runs in containers |
| CI/CD | GitHub Actions | Automated testing, linting, deployment |
| Cloud (optional) | AWS (S3, Lambda) | Model artifact storage, triggers |

---

## 4. Repository Structure

```
agentops/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── Makefile
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint, test, type-check on every PR
│       └── deploy.yml                # Build & push Docker images
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # All configuration (env vars, thresholds)
│   │
│   ├── agents/                       # The 4 LangGraph agents
│   │   ├── __init__.py
│   │   ├── orchestrator.py           # LangGraph graph definition
│   │   ├── state.py                  # Shared state schema (TypedDict)
│   │   ├── data_quality_agent.py     # Agent 1: Drift detection
│   │   ├── model_eval_agent.py       # Agent 2: Performance monitoring
│   │   ├── retraining_agent.py       # Agent 3: Fine-tuning trigger
│   │   └── deployment_agent.py       # Agent 4: Model swap / rollback
│   │
│   ├── ml/                           # ML model code
│   │   ├── __init__.py
│   │   ├── model.py                  # DistilBERT wrapper class
│   │   ├── train.py                  # Training / fine-tuning logic
│   │   ├── evaluate.py               # Evaluation metrics
│   │   ├── predict.py                # Inference pipeline
│   │   └── data_processor.py         # Data loading, preprocessing
│   │
│   ├── monitoring/                   # Drift & performance monitoring
│   │   ├── __init__.py
│   │   ├── drift_detector.py         # Evidently AI integration
│   │   ├── performance_tracker.py    # Accuracy/F1 over time
│   │   └── metrics_exporter.py       # Prometheus metrics export
│   │
│   ├── api/                          # FastAPI application
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI app initialization
│   │   ├── routes/
│   │   │   ├── predict.py            # POST /predict endpoint
│   │   │   ├── health.py             # GET /health endpoint
│   │   │   ├── metrics.py            # GET /metrics (Prometheus)
│   │   │   └── agents.py             # GET/POST agent status & triggers
│   │   └── middleware/
│   │       ├── logging.py            # Structured request logging
│   │       └── prediction_logger.py  # Log predictions to PostgreSQL
│   │
│   ├── mcp/                          # MCP Server
│   │   ├── __init__.py
│   │   └── server.py                 # MCP tools exposed to AI clients
│   │
│   ├── storage/                      # Data & model persistence
│   │   ├── __init__.py
│   │   ├── database.py               # PostgreSQL connection & queries
│   │   ├── model_registry.py         # MLflow model versioning
│   │   └── s3_client.py              # AWS S3 for model artifacts (optional)
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py                 # Structured logging setup
│
├── tests/
│   ├── __init__.py
│   ├── test_agents/
│   │   ├── test_orchestrator.py
│   │   ├── test_data_quality.py
│   │   ├── test_model_eval.py
│   │   ├── test_retraining.py
│   │   └── test_deployment.py
│   ├── test_ml/
│   │   ├── test_model.py
│   │   ├── test_train.py
│   │   └── test_predict.py
│   ├── test_api/
│   │   ├── test_predict_endpoint.py
│   │   └── test_health.py
│   └── test_monitoring/
│       ├── test_drift_detector.py
│       └── test_performance_tracker.py
│
├── scripts/
│   ├── seed_data.py                  # Generate initial training data
│   ├── simulate_drift.py             # Simulate data drift for demo
│   ├── initial_train.py              # Train the first model version
│   └── demo.py                       # Run the full demo end-to-end
│
├── monitoring/
│   ├── prometheus.yml                # Prometheus config
│   └── grafana/
│       ├── provisioning/
│       │   ├── dashboards.yml
│       │   └── datasources.yml
│       └── dashboards/
│           ├── model_performance.json # Pre-built Grafana dashboard
│           └── data_drift.json        # Drift monitoring dashboard
│
├── data/
│   ├── raw/                          # Raw training data
│   ├── processed/                    # Preprocessed data
│   └── reference/                    # Reference data for drift comparison
│
└── models/                           # Local model storage
    └── .gitkeep
```

---

## 5. Phase-by-Phase Implementation

### Phase 1: ML Model (Week 1)

This is the foundation. You build the DistilBERT classifier and the training pipeline.

**Step 1: Set up the project**

```bash
mkdir agentops && cd agentops
git init
python -m venv venv && source venv/bin/activate

# Create pyproject.toml
```

**pyproject.toml dependencies:**

```toml
[project]
name = "agentops"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "datasets>=2.18.0",
    "scikit-learn>=1.4.0",
    "fastapi>=0.111.0",
    "uvicorn>=0.29.0",
    "langgraph>=0.2.0",
    "langchain-openai>=0.2.0",
    "langchain-core>=0.3.0",
    "evidently>=0.5.0",
    "mlflow>=2.14.0",
    "prometheus-client>=0.20.0",
    "psycopg2-binary>=2.9.0",
    "redis>=5.0.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.2.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "fastmcp>=1.0.0",
    "structlog>=24.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]
```

**Step 2: Data preparation — `src/ml/data_processor.py`**

```python
"""Data loading and preprocessing for ticket classification."""

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from typing import Tuple
import torch


LABEL_MAP = {
    "billing": 0,
    "technical": 1,
    "account": 2,
    "feature_request": 3,
    "general": 4,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


class TicketDataProcessor:
    """Handles all data loading, preprocessing, and tokenization."""

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def load_and_prepare(self, data_path: str = None) -> Tuple[Dataset, Dataset, Dataset]:
        """Load data, split into train/val/test, tokenize.

        If no data_path provided, uses a public dataset and maps it
        to our label schema.
        """
        if data_path:
            df = pd.read_csv(data_path)
        else:
            # Use a public dataset as starting point
            # We'll remap categories to our support ticket labels
            dataset = load_dataset("ag_news", split="train[:10000]")
            df = pd.DataFrame(dataset)
            # Map ag_news labels (0-3) to our labels
            ag_to_ticket = {0: "general", 1: "technical", 2: "billing", 3: "feature_request"}
            df["label_name"] = df["label"].map(ag_to_ticket)
            df["label"] = df["label_name"].map(LABEL_MAP)
            df = df.rename(columns={"text": "ticket_text"})

        # Split: 70/15/15
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

        # Tokenize
        train_dataset = self._tokenize_df(train_df)
        val_dataset = self._tokenize_df(val_df)
        test_dataset = self._tokenize_df(test_df)

        return train_dataset, val_dataset, test_dataset

    def _tokenize_df(self, df: pd.DataFrame) -> Dataset:
        """Tokenize a DataFrame into a HuggingFace Dataset."""
        dataset = Dataset.from_pandas(df[["ticket_text", "label"]].reset_index(drop=True))
        dataset = dataset.map(
            lambda batch: self.tokenizer(
                batch["ticket_text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            ),
            batched=True,
        )
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset

    def tokenize_single(self, text: str) -> dict:
        """Tokenize a single input for inference."""
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded
```

**Step 3: Model wrapper — `src/ml/model.py`**

```python
"""DistilBERT model wrapper for ticket classification."""

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification
from pathlib import Path
from typing import Tuple
import structlog

logger = structlog.get_logger()


class TicketClassifier:
    """Wraps DistilBERT for support ticket classification.

    This class handles model loading, inference, and confidence scoring.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 5,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.num_labels = num_labels

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
            probabilities = F.softmax(outputs.logits, dim=-1)
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
```

**Step 4: Training logic — `src/ml/train.py`**

```python
"""Fine-tuning pipeline for DistilBERT ticket classifier."""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
import mlflow
import structlog
from typing import Dict, Any

from src.ml.model import TicketClassifier
from src.ml.data_processor import LABEL_NAMES

logger = structlog.get_logger()


class Trainer:
    """Handles the full training and evaluation loop.

    Integrates with MLflow for experiment tracking.
    """

    def __init__(
        self,
        model: TicketClassifier,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3,
        warmup_steps: int = 100,
    ):
        self.model = model
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps

    def train(self, train_dataset, val_dataset, experiment_name: str = "ticket_classifier") -> Dict[str, Any]:
        """Full training loop with MLflow logging.

        Returns dict with final metrics and model path.
        """
        mlflow.set_experiment(experiment_name)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = AdamW(self.model.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps
        )

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
                "model_name": "distilbert-base-uncased",
            })

            best_f1 = 0.0
            for epoch in range(self.num_epochs):
                # Training
                self.model.model.train()
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model.model(
                        input_ids=batch["input_ids"].to(self.model.device),
                        attention_mask=batch["attention_mask"].to(self.model.device),
                        labels=batch["label"].to(self.model.device),
                    )
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                # Validation
                val_metrics = self._evaluate(val_loader)
                mlflow.log_metric("val_f1", val_metrics["f1_weighted"], step=epoch)
                mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)

                logger.info(
                    "epoch_complete",
                    epoch=epoch + 1,
                    train_loss=round(avg_loss, 4),
                    val_f1=round(val_metrics["f1_weighted"], 4),
                )

                # Save best model
                if val_metrics["f1_weighted"] > best_f1:
                    best_f1 = val_metrics["f1_weighted"]
                    self.model.save("models/best")
                    mlflow.log_artifact("models/best")

            mlflow.log_metric("best_f1", best_f1)

        return {"best_f1": best_f1, "model_path": "models/best"}

    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataloader. Returns metrics dict."""
        self.model.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model.model(
                    input_ids=batch["input_ids"].to(self.model.device),
                    attention_mask=batch["attention_mask"].to(self.model.device),
                )
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["label"].tolist())

        report = classification_report(
            all_labels, all_preds,
            target_names=[LABEL_NAMES[i] for i in range(len(LABEL_NAMES))],
            output_dict=True,
        )

        return {
            "accuracy": report["accuracy"],
            "f1_weighted": report["weighted avg"]["f1-score"],
            "f1_per_class": {
                LABEL_NAMES[i]: report[LABEL_NAMES[i]]["f1-score"]
                for i in range(len(LABEL_NAMES))
            },
            "full_report": report,
        }
```

---

### Phase 2: Monitoring Layer (Week 2)

Now you build the drift detection and performance monitoring.

**Step 5: Drift detection — `src/monitoring/drift_detector.py`**

```python
"""Data drift detection using Evidently AI."""

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)
import pandas as pd
import structlog
from typing import Dict, Any
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class DriftReport:
    """Result of a drift detection check."""
    is_drifted: bool
    drift_score: float           # Overall drift score (0-1)
    drifted_columns: list        # Which features drifted
    column_scores: Dict[str, float]  # Per-column drift scores
    details: Dict[str, Any]      # Full Evidently report data


class DriftDetector:
    """Monitors incoming data for statistical drift against a reference dataset.

    Uses Evidently AI under the hood. Supports multiple drift detection methods:
    - PSI (Population Stability Index)
    - KS (Kolmogorov-Smirnov) test
    - Jensen-Shannon divergence
    """

    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.3):
        """
        Args:
            reference_data: The "known good" data distribution to compare against.
                           Typically your training data or a validated production window.
            drift_threshold: Fraction of columns that need to drift to trigger alert.
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold

    def check_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Compare current production data against reference.

        Args:
            current_data: Recent production data window (e.g., last 1000 predictions)

        Returns:
            DriftReport with drift status, scores, and details
        """
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )

        result = report.as_dict()

        # Extract drift info from Evidently result
        dataset_drift = result["metrics"][0]["result"]
        drift_table = result["metrics"][1]["result"]

        # Get per-column drift scores
        column_scores = {}
        drifted_columns = []
        for col_name, col_data in drift_table.get("drift_by_columns", {}).items():
            score = col_data.get("drift_score", 0)
            column_scores[col_name] = score
            if col_data.get("drift_detected", False):
                drifted_columns.append(col_name)

        drift_report = DriftReport(
            is_drifted=dataset_drift.get("dataset_drift", False),
            drift_score=dataset_drift.get("share_of_drifted_columns", 0),
            drifted_columns=drifted_columns,
            column_scores=column_scores,
            details=result,
        )

        logger.info(
            "drift_check_complete",
            is_drifted=drift_report.is_drifted,
            drift_score=drift_report.drift_score,
            drifted_columns=drifted_columns,
        )

        return drift_report

    def update_reference(self, new_reference: pd.DataFrame):
        """Update reference data after a successful retraining cycle."""
        self.reference_data = new_reference
        logger.info("reference_data_updated", rows=len(new_reference))
```

**Step 6: Performance tracking — `src/monitoring/performance_tracker.py`**

```python
"""Track model performance metrics over time."""

from collections import deque
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import structlog
from typing import Dict, Optional
from dataclasses import dataclass, field

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

    def __init__(
        self,
        window_size: int = 500,
        degradation_threshold: float = 0.05,  # 5% drop triggers alert
        baseline_f1: float = 0.85,
    ):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.baseline_f1 = baseline_f1

        # Rolling window of (prediction, ground_truth, confidence)
        self.predictions = deque(maxlen=window_size)
        self.history: list[PerformanceSnapshot] = []

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
        if len(self.predictions) < 50:  # Need minimum samples
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
```

**Step 7: Prometheus metrics — `src/monitoring/metrics_exporter.py`**

```python
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
```

---

### Phase 3: The 4 Agents (Week 3)

This is the core of the project — LangGraph orchestrating 4 autonomous agents.

**Step 8: Shared state — `src/agents/state.py`**

```python
"""Shared state schema for the agent pipeline.

LangGraph agents communicate through this shared state object.
Each agent reads from it and writes its results back.
"""

from typing import TypedDict, Optional, Literal
from dataclasses import dataclass


class AgentState(TypedDict):
    """State shared across all agents in the pipeline.

    LangGraph passes this dict between nodes. Each agent
    reads what it needs and writes its output.
    """
    # Trigger info
    trigger_reason: Literal["scheduled", "manual", "alert"]
    triggered_at: str

    # Data Quality Agent output
    drift_detected: bool
    drift_score: float
    drifted_features: list[str]
    drift_report_summary: str

    # Model Evaluation Agent output
    performance_degraded: bool
    current_f1: float
    baseline_f1: float
    f1_drop: float
    eval_summary: str

    # Retraining Agent output
    retraining_triggered: bool
    new_model_f1: float
    new_model_path: str
    retraining_summary: str
    mlflow_run_id: str

    # Deployment Agent output
    deployment_action: Literal["swap", "rollback", "none"]
    deployed_model_version: str
    deployment_summary: str

    # Overall pipeline
    pipeline_decision: str  # LLM's reasoning about what to do
    final_summary: str
    errors: list[str]
```

**Step 9: Agent 1 — Data Quality Agent — `src/agents/data_quality_agent.py`**

```python
"""Agent 1: Monitors incoming data for drift.

This agent fetches recent production data, compares it against
the reference distribution, and reports whether drift occurred.
"""

from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import structlog

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.metrics_exporter import DATA_DRIFT_SCORE, DATA_DRIFT_DETECTED, DRIFTED_FEATURES_COUNT
from src.agents.state import AgentState
from src.storage.database import get_recent_predictions

logger = structlog.get_logger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def data_quality_agent(state: AgentState) -> AgentState:
    """Check production data for drift against reference distribution.

    1. Fetches last N predictions from PostgreSQL
    2. Runs Evidently drift detection against training reference
    3. Uses LLM to interpret results and generate summary
    4. Updates Prometheus metrics
    """
    logger.info("data_quality_agent_started")

    try:
        # Fetch recent production data
        recent_data = get_recent_predictions(limit=1000)

        if recent_data.empty:
            state["drift_detected"] = False
            state["drift_score"] = 0.0
            state["drifted_features"] = []
            state["drift_report_summary"] = "No recent data available for drift analysis."
            return state

        # Run drift detection
        detector = DriftDetector.get_instance()  # Singleton with reference data
        drift_report = detector.check_drift(recent_data)

        # Update Prometheus metrics
        DATA_DRIFT_SCORE.set(drift_report.drift_score)
        DATA_DRIFT_DETECTED.set(1 if drift_report.is_drifted else 0)
        DRIFTED_FEATURES_COUNT.set(len(drift_report.drifted_columns))

        # Use LLM to interpret the drift results
        interpretation = llm.invoke([HumanMessage(content=f"""
You are an MLOps monitoring agent. Analyze this data drift report and provide
a concise summary (2-3 sentences) of what's happening:

- Dataset drift detected: {drift_report.is_drifted}
- Overall drift score: {drift_report.drift_score:.3f}
- Drifted columns: {drift_report.drifted_columns}
- Column-level scores: {drift_report.column_scores}

Focus on: Is this concerning? What kind of drift is this?
What might be causing it in a customer support ticket context?
""")])

        state["drift_detected"] = drift_report.is_drifted
        state["drift_score"] = drift_report.drift_score
        state["drifted_features"] = drift_report.drifted_columns
        state["drift_report_summary"] = interpretation.content

        logger.info(
            "data_quality_agent_complete",
            drift_detected=drift_report.is_drifted,
            drift_score=drift_report.drift_score,
        )

    except Exception as e:
        logger.error("data_quality_agent_failed", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Data quality agent: {str(e)}"]
        state["drift_detected"] = False

    return state
```

**Step 10: Agent 2 — Model Evaluation Agent — `src/agents/model_eval_agent.py`**

```python
"""Agent 2: Tracks model performance and detects degradation.

Computes accuracy, F1, precision, recall over a sliding window
and determines if the model needs retraining.
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import structlog

from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.metrics_exporter import MODEL_F1_SCORE, MODEL_ACCURACY
from src.agents.state import AgentState

logger = structlog.get_logger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def model_eval_agent(state: AgentState) -> AgentState:
    """Evaluate current model performance against baseline.

    1. Computes metrics over the sliding prediction window
    2. Checks for degradation against baseline thresholds
    3. Cross-references with drift data from Agent 1
    4. Uses LLM to decide if retraining is warranted
    """
    logger.info("model_eval_agent_started")

    try:
        tracker = PerformanceTracker.get_instance()
        degradation = tracker.is_degraded()

        # Update Prometheus
        if degradation.get("current_f1"):
            MODEL_F1_SCORE.set(degradation["current_f1"])
        if degradation.get("current_accuracy"):
            MODEL_ACCURACY.set(degradation["current_accuracy"])

        # LLM decides: is retraining needed?
        decision = llm.invoke([HumanMessage(content=f"""
You are an MLOps evaluation agent. Based on these metrics, decide if
the model needs retraining. Reply with a JSON object.

Model Performance:
- Current F1: {degradation.get('current_f1', 'N/A')}
- Baseline F1: {degradation.get('baseline_f1', 'N/A')}
- F1 Drop: {degradation.get('f1_drop', 'N/A')}
- Confidence Mean: {degradation.get('confidence_mean', 'N/A')}

Data Drift Status (from previous agent):
- Drift Detected: {state.get('drift_detected', False)}
- Drift Score: {state.get('drift_score', 0)}
- Drifted Features: {state.get('drifted_features', [])}

Decision criteria:
- If F1 dropped > 5% AND drift detected → RETRAIN (urgent)
- If F1 dropped > 5% but no drift → RETRAIN (model degradation)
- If drift detected but F1 still good → MONITOR (drift hasn't impacted yet)
- If neither → NO ACTION

Provide your reasoning in 2-3 sentences and your decision.
""")])

        state["performance_degraded"] = degradation.get("degraded", False)
        state["current_f1"] = degradation.get("current_f1", 0)
        state["baseline_f1"] = degradation.get("baseline_f1", 0)
        state["f1_drop"] = degradation.get("f1_drop", 0)
        state["eval_summary"] = decision.content

        logger.info(
            "model_eval_agent_complete",
            degraded=degradation.get("degraded"),
            current_f1=degradation.get("current_f1"),
        )

    except Exception as e:
        logger.error("model_eval_agent_failed", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Model eval agent: {str(e)}"]
        state["performance_degraded"] = False

    return state
```

**Step 11: Agent 3 — Retraining Agent — `src/agents/retraining_agent.py`**

```python
"""Agent 3: Automatically fine-tunes the model when needed.

Triggered by the evaluation agent's decision. Pulls new data,
fine-tunes DistilBERT, evaluates the new model, and logs to MLflow.
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import structlog

from src.ml.model import TicketClassifier
from src.ml.train import Trainer
from src.ml.data_processor import TicketDataProcessor
from src.monitoring.metrics_exporter import RETRAINING_RUNS
from src.agents.state import AgentState

logger = structlog.get_logger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def retraining_agent(state: AgentState) -> AgentState:
    """Fine-tune the model on updated data.

    1. Checks if retraining is warranted (from Agent 2's decision)
    2. Loads updated training data (including recent production data)
    3. Fine-tunes DistilBERT
    4. Evaluates new model against test set
    5. Logs everything to MLflow
    6. Decides if new model is good enough to deploy
    """
    logger.info("retraining_agent_started")

    # Check if retraining is needed
    should_retrain = state.get("performance_degraded", False) or state.get("drift_detected", False)

    if not should_retrain:
        state["retraining_triggered"] = False
        state["retraining_summary"] = "No retraining needed. Model performance is within acceptable bounds."
        return state

    try:
        RETRAINING_RUNS.inc()

        # Load fresh data (includes recent production samples if labeled)
        processor = TicketDataProcessor()
        train_dataset, val_dataset, test_dataset = processor.load_and_prepare()

        # Initialize fresh model for fine-tuning
        model = TicketClassifier()

        # Load current best weights as starting point
        try:
            model.load("models/best")
            logger.info("loaded_existing_model_for_finetuning")
        except Exception:
            logger.info("starting_from_pretrained_distilbert")

        # Train
        trainer = Trainer(
            model=model,
            learning_rate=1e-5,    # Lower LR for fine-tuning existing model
            batch_size=16,
            num_epochs=2,          # Fewer epochs for fine-tuning
        )
        results = trainer.train(train_dataset, val_dataset, experiment_name="agentops_retraining")

        # LLM evaluates the training results
        evaluation = llm.invoke([HumanMessage(content=f"""
You are an MLOps retraining agent. Evaluate these training results:

- New model F1: {results['best_f1']:.4f}
- Previous baseline F1: {state.get('baseline_f1', 'N/A')}
- F1 improvement: {results['best_f1'] - state.get('baseline_f1', 0):.4f}

Should we deploy this new model? Consider:
- Is the new F1 better than the degraded model?
- Is it close to or better than the original baseline?
- Are there any red flags?

Respond with DEPLOY or SKIP and your reasoning.
""")])

        state["retraining_triggered"] = True
        state["new_model_f1"] = results["best_f1"]
        state["new_model_path"] = results["model_path"]
        state["retraining_summary"] = evaluation.content
        state["mlflow_run_id"] = results.get("mlflow_run_id", "")

        logger.info("retraining_agent_complete", new_f1=results["best_f1"])

    except Exception as e:
        logger.error("retraining_agent_failed", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Retraining agent: {str(e)}"]
        state["retraining_triggered"] = False

    return state
```

**Step 12: Agent 4 — Deployment Agent — `src/agents/deployment_agent.py`**

```python
"""Agent 4: Handles model deployment, A/B testing, and rollback.

Takes the retrained model and safely deploys it to production
with zero-downtime swap and automatic rollback capability.
"""

import shutil
from pathlib import Path
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import structlog

from src.monitoring.metrics_exporter import DEPLOYMENT_SWAPS, MODEL_VERSION
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.drift_detector import DriftDetector
from src.agents.state import AgentState

logger = structlog.get_logger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def deployment_agent(state: AgentState) -> AgentState:
    """Deploy or rollback model based on pipeline results.

    1. If retraining produced a better model → swap to new model
    2. Update model version metadata
    3. Reset monitoring baselines
    4. Generate final pipeline summary
    """
    logger.info("deployment_agent_started")

    if not state.get("retraining_triggered", False):
        state["deployment_action"] = "none"
        state["deployment_summary"] = "No deployment needed."
        _generate_final_summary(state)
        return state

    try:
        new_f1 = state.get("new_model_f1", 0)
        baseline_f1 = state.get("baseline_f1", 0)
        new_model_path = state.get("new_model_path", "")

        # Decision: deploy only if new model is better than degraded performance
        should_deploy = new_f1 > state.get("current_f1", 0) and "DEPLOY" in state.get("retraining_summary", "").upper()

        if should_deploy and new_model_path:
            # Atomic model swap
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            production_path = Path("models/production")
            backup_path = Path(f"models/backup_{version}")

            # Backup current production model
            if production_path.exists():
                shutil.copytree(production_path, backup_path)

            # Deploy new model
            if production_path.exists():
                shutil.rmtree(production_path)
            shutil.copytree(new_model_path, production_path)

            # Update metrics
            DEPLOYMENT_SWAPS.inc()
            MODEL_VERSION.info({"version": version, "f1": str(new_f1)})

            # Reset monitoring baselines
            tracker = PerformanceTracker.get_instance()
            tracker.update_baseline(new_f1)

            state["deployment_action"] = "swap"
            state["deployed_model_version"] = version
            state["deployment_summary"] = f"Deployed model v{version} with F1={new_f1:.4f}"

            logger.info("model_deployed", version=version, f1=new_f1)

        else:
            state["deployment_action"] = "none"
            state["deployment_summary"] = f"New model (F1={new_f1:.4f}) did not meet deployment criteria."

    except Exception as e:
        logger.error("deployment_agent_failed", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Deployment agent: {str(e)}"]
        state["deployment_action"] = "none"

    _generate_final_summary(state)
    return state


def _generate_final_summary(state: AgentState):
    """Use LLM to generate a human-readable pipeline summary."""
    summary = llm.invoke([HumanMessage(content=f"""
Summarize this MLOps pipeline run in 3-4 sentences for a status dashboard:

- Data Drift: {state.get('drift_report_summary', 'N/A')}
- Model Performance: F1={state.get('current_f1', 'N/A')}, Degraded={state.get('performance_degraded', False)}
- Retraining: {state.get('retraining_summary', 'Not triggered')}
- Deployment: {state.get('deployment_summary', 'No action')}
- Errors: {state.get('errors', [])}

Be concise and factual.
""")])
    state["final_summary"] = summary.content
```

**Step 13: The Orchestrator — `src/agents/orchestrator.py`**

```python
"""LangGraph orchestrator — wires the 4 agents into a pipeline.

This is the brain of AgentOps. It defines the graph of agents,
the conditional edges (which agent runs next based on state),
and the overall execution flow.
"""

from datetime import datetime
from langgraph.graph import StateGraph, END
import structlog

from src.agents.state import AgentState
from src.agents.data_quality_agent import data_quality_agent
from src.agents.model_eval_agent import model_eval_agent
from src.agents.retraining_agent import retraining_agent
from src.agents.deployment_agent import deployment_agent
from src.monitoring.metrics_exporter import AGENT_RUNS_TOTAL, AGENT_PIPELINE_DURATION

logger = structlog.get_logger()


def should_evaluate(state: AgentState) -> str:
    """After drift check: always proceed to evaluation."""
    return "evaluate"


def should_retrain(state: AgentState) -> str:
    """After evaluation: retrain if degraded or drifted, else skip to deploy."""
    if state.get("performance_degraded") or state.get("drift_detected"):
        return "retrain"
    return "deploy"


def build_pipeline() -> StateGraph:
    """Construct the LangGraph agent pipeline.

    Flow:
        data_quality → model_eval → (retrain if needed) → deploy → END

    Returns a compiled LangGraph that can be invoked with initial state.
    """
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("data_quality", data_quality_agent)
    workflow.add_node("model_eval", model_eval_agent)
    workflow.add_node("retrain", retraining_agent)
    workflow.add_node("deploy", deployment_agent)

    # Define the flow
    workflow.set_entry_point("data_quality")
    workflow.add_edge("data_quality", "model_eval")
    workflow.add_conditional_edges("model_eval", should_retrain, {
        "retrain": "retrain",
        "deploy": "deploy",
    })
    workflow.add_edge("retrain", "deploy")
    workflow.add_edge("deploy", END)

    return workflow.compile()


# Build once, reuse
pipeline = build_pipeline()


async def run_pipeline(trigger_reason: str = "scheduled") -> AgentState:
    """Execute the full agent pipeline.

    Args:
        trigger_reason: What triggered this run (scheduled/manual/alert)

    Returns:
        Final AgentState with all agent outputs
    """
    import time
    start = time.time()

    AGENT_RUNS_TOTAL.labels(trigger_reason=trigger_reason).inc()

    initial_state: AgentState = {
        "trigger_reason": trigger_reason,
        "triggered_at": datetime.utcnow().isoformat(),
        "drift_detected": False,
        "drift_score": 0.0,
        "drifted_features": [],
        "drift_report_summary": "",
        "performance_degraded": False,
        "current_f1": 0.0,
        "baseline_f1": 0.0,
        "f1_drop": 0.0,
        "eval_summary": "",
        "retraining_triggered": False,
        "new_model_f1": 0.0,
        "new_model_path": "",
        "retraining_summary": "",
        "mlflow_run_id": "",
        "deployment_action": "none",
        "deployed_model_version": "",
        "deployment_summary": "",
        "pipeline_decision": "",
        "final_summary": "",
        "errors": [],
    }

    logger.info("pipeline_started", trigger=trigger_reason)
    result = await pipeline.ainvoke(initial_state)

    duration = time.time() - start
    AGENT_PIPELINE_DURATION.observe(duration)

    logger.info(
        "pipeline_complete",
        duration=round(duration, 2),
        drift_detected=result["drift_detected"],
        retrained=result["retraining_triggered"],
        deployed=result["deployment_action"],
    )

    return result
```

---

### Phase 4: API + MCP Server (Week 4)

**Step 14: FastAPI application — `src/api/app.py`**

```python
"""FastAPI application — serves predictions and agent controls."""

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from contextlib import asynccontextmanager
import structlog

from src.api.routes import predict, health, agents
from src.ml.model import TicketClassifier
from src.ml.data_processor import TicketDataProcessor

logger = structlog.get_logger()

# Global model instance
model: TicketClassifier = None
processor: TicketDataProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, processor
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
```

**Step 15: Prediction endpoint — `src/api/routes/predict.py`**

```python
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

    return PredictResponse(
        label=label_name,
        label_id=label_id,
        confidence=round(confidence, 4),
        probabilities={LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    )
```

**Step 16: Agent control endpoint — `src/api/routes/agents.py`**

```python
"""Agent pipeline control endpoints."""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from src.agents.orchestrator import run_pipeline

router = APIRouter()


class PipelineStatus(BaseModel):
    status: str
    last_run: Optional[str] = None
    drift_detected: Optional[bool] = None
    current_f1: Optional[float] = None
    last_action: Optional[str] = None
    summary: Optional[str] = None


class PipelineTriggerResponse(BaseModel):
    message: str
    trigger_reason: str


# Store last pipeline result
_last_result = {}


@router.get("/agents/status", response_model=PipelineStatus)
async def get_agent_status():
    """Get current status of the agent pipeline."""
    if not _last_result:
        return PipelineStatus(status="idle")

    return PipelineStatus(
        status="complete",
        last_run=_last_result.get("triggered_at"),
        drift_detected=_last_result.get("drift_detected"),
        current_f1=_last_result.get("current_f1"),
        last_action=_last_result.get("deployment_action"),
        summary=_last_result.get("final_summary"),
    )


@router.post("/agents/trigger", response_model=PipelineTriggerResponse)
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """Manually trigger the agent pipeline."""
    background_tasks.add_task(_run_and_store, "manual")
    return PipelineTriggerResponse(
        message="Pipeline triggered",
        trigger_reason="manual",
    )


async def _run_and_store(reason: str):
    global _last_result
    _last_result = await run_pipeline(trigger_reason=reason)
```

**Step 17: MCP Server — `src/mcp/server.py`**

This is the key differentiator. This exposes your entire system as an MCP server.

```python
"""MCP Server — exposes AgentOps to Claude, Cursor, and any MCP client.

Tools exposed:
- check_model_status: Get current model health
- check_drift: Run drift detection
- trigger_retraining: Manually trigger the pipeline
- predict_ticket: Classify a support ticket
- get_pipeline_history: View past pipeline runs
"""

from fastmcp import FastMCP
from src.agents.orchestrator import run_pipeline
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance_tracker import PerformanceTracker
from src.storage.database import get_recent_predictions, get_pipeline_history

mcp = FastMCP("AgentOps", description="Agentic MLOps Pipeline — monitor, retrain, and deploy ML models")


@mcp.tool()
async def check_model_status() -> dict:
    """Check the current health and performance of the deployed model.

    Returns F1 score, accuracy, drift status, and whether
    the model is degraded.
    """
    tracker = PerformanceTracker.get_instance()
    degradation = tracker.is_degraded()

    detector = DriftDetector.get_instance()
    recent_data = get_recent_predictions(limit=500)
    drift = detector.check_drift(recent_data) if not recent_data.empty else None

    return {
        "model_health": "degraded" if degradation.get("degraded") else "healthy",
        "current_f1": degradation.get("current_f1"),
        "baseline_f1": degradation.get("baseline_f1"),
        "drift_detected": drift.is_drifted if drift else False,
        "drift_score": drift.drift_score if drift else 0,
        "prediction_count": degradation.get("sample_count", 0),
    }


@mcp.tool()
async def trigger_retraining(reason: str = "manual") -> dict:
    """Trigger the full agent pipeline: drift check → evaluation → retrain → deploy.

    Args:
        reason: Why you're triggering this (e.g., "manual", "performance_drop")

    Returns the pipeline result including what actions were taken.
    """
    result = await run_pipeline(trigger_reason=reason)
    return {
        "drift_detected": result["drift_detected"],
        "performance_degraded": result["performance_degraded"],
        "retrained": result["retraining_triggered"],
        "deployed": result["deployment_action"],
        "new_f1": result.get("new_model_f1"),
        "summary": result["final_summary"],
    }


@mcp.tool()
async def predict_ticket(text: str) -> dict:
    """Classify a customer support ticket.

    Args:
        text: The ticket text to classify

    Returns predicted category and confidence.
    """
    from src.api.app import model, processor
    from src.ml.data_processor import LABEL_NAMES

    encoded = processor.tokenize_single(text)
    label_id, confidence, probs = model.predict(encoded["input_ids"], encoded["attention_mask"])

    return {
        "category": LABEL_NAMES[label_id],
        "confidence": round(confidence, 4),
        "all_probabilities": {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    }


@mcp.tool()
async def get_pipeline_history(limit: int = 10) -> list:
    """Get history of recent pipeline runs.

    Args:
        limit: Number of recent runs to return

    Returns list of pipeline run summaries.
    """
    return get_pipeline_history(limit=limit)


if __name__ == "__main__":
    mcp.run()
```

---

### Phase 5: Docker + Monitoring Dashboards (Week 5)

**Step 18: docker-compose.yml**

```yaml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://agentops:agentops@postgres:5432/agentops
      - REDIS_URL=redis://redis:6379
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - postgres
      - redis
      - mlflow
    volumes:
      - ./models:/app/models
      - ./data:/app/data

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: agentops
      POSTGRES_PASSWORD: agentops
      POSTGRES_DB: agentops
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.0
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - mlflow_data:/mlflow

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards

volumes:
  pgdata:
  mlflow_data:
```

**Step 19: Prometheus config — `monitoring/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "agentops"
    static_configs:
      - targets: ["app:8000"]
    metrics_path: "/metrics"
```

**Step 20: Drift simulation script — `scripts/simulate_drift.py`**

This is how you demo the project. You simulate data drift and watch the agents react.

```python
"""Simulate data drift to trigger the agent pipeline.

This script gradually shifts the distribution of incoming tickets
to simulate a real-world scenario (e.g., a product launch causes
a surge in 'technical' tickets, or a pricing change causes
'billing' tickets to spike).
"""

import httpx
import random
import time
import asyncio

API_URL = "http://localhost:8000/api/v1/predict"

# Normal distribution of tickets
NORMAL_DISTRIBUTION = {
    "billing": 0.25,
    "technical": 0.25,
    "account": 0.20,
    "feature_request": 0.15,
    "general": 0.15,
}

# Drifted distribution (simulate product launch — technical tickets spike)
DRIFTED_DISTRIBUTION = {
    "billing": 0.10,
    "technical": 0.55,  # Spike!
    "account": 0.10,
    "feature_request": 0.20,
    "general": 0.05,
}

# Sample ticket templates per category
TICKET_TEMPLATES = {
    "billing": [
        "I was charged twice for my subscription this month",
        "Can I get a refund for the premium plan?",
        "My invoice shows the wrong amount",
        "How do I update my payment method?",
    ],
    "technical": [
        "The app crashes when I try to upload files",
        "API returns 500 error on the /users endpoint",
        "Integration with Slack stopped working after the update",
        "WebSocket connection drops every 30 seconds",
        "Getting CORS errors when calling from our frontend",
        "Memory usage spikes to 95% under load",
    ],
    "account": [
        "I can't reset my password",
        "How do I change my email address?",
        "I need to merge two accounts",
    ],
    "feature_request": [
        "Can you add dark mode to the dashboard?",
        "We need SSO support for our enterprise team",
        "Please add CSV export for reports",
    ],
    "general": [
        "What are your business hours?",
        "Is there a mobile app available?",
        "How do I contact sales?",
    ],
}


async def send_ticket(client: httpx.AsyncClient, category: str):
    """Send a simulated ticket to the prediction endpoint."""
    text = random.choice(TICKET_TEMPLATES[category])
    # Add some noise to make it more realistic
    text += f" [ref:{random.randint(1000, 9999)}]"

    response = await client.post(API_URL, json={"text": text})
    return response.json()


async def simulate(phase: str = "normal", count: int = 100, delay: float = 0.1):
    """Run a simulation phase.

    Args:
        phase: 'normal' or 'drift'
        count: Number of tickets to send
        delay: Seconds between tickets
    """
    distribution = NORMAL_DISTRIBUTION if phase == "normal" else DRIFTED_DISTRIBUTION
    categories = list(distribution.keys())
    weights = list(distribution.values())

    async with httpx.AsyncClient() as client:
        for i in range(count):
            category = random.choices(categories, weights=weights, k=1)[0]
            result = await send_ticket(client, category)

            if i % 25 == 0:
                print(f"[{phase}] Sent {i}/{count} tickets. "
                      f"Last prediction: {result['label']} ({result['confidence']:.2f})")

            await asyncio.sleep(delay)

    print(f"\n[{phase}] Phase complete. Sent {count} tickets.\n")


async def main():
    """Full demo: normal traffic → drift → watch agents react."""
    print("=== Phase 1: Normal Traffic (building baseline) ===")
    await simulate("normal", count=500, delay=0.05)

    print("Waiting 10 seconds for metrics to stabilize...\n")
    time.sleep(10)

    print("=== Phase 2: Drifted Traffic (simulating product launch) ===")
    await simulate("drift", count=500, delay=0.05)

    print("=== Phase 3: Triggering agent pipeline ===")
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/api/v1/agents/trigger")
        print(f"Pipeline triggered: {response.json()}")

    print("\nMonitor the pipeline at:")
    print("  Grafana:  http://localhost:3000")
    print("  MLflow:   http://localhost:5000")
    print("  API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    asyncio.run(main())
```

---

### Phase 6: CI/CD + Tests (Week 5-6)

**Step 21: GitHub Actions — `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: agentops_test
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check src/ tests/

      - name: Type check
        run: mypy src/ --ignore-missing-imports

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/agentops_test
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## 6. Grafana Dashboard Layout

Create two dashboards:

**Dashboard 1: Model Performance**
- Panel 1: F1 Score over time (line chart) — `agentops_model_f1_score`
- Panel 2: Prediction confidence distribution (histogram) — `agentops_prediction_confidence`
- Panel 3: Predictions per class (stacked bar) — `agentops_predictions_total`
- Panel 4: Prediction latency p50/p95/p99 (line chart) — `agentops_prediction_latency_seconds`

**Dashboard 2: Agent Pipeline & Drift**
- Panel 1: Data drift score over time (line chart with threshold line) — `agentops_data_drift_score`
- Panel 2: Drift detected alert (stat panel, red/green) — `agentops_data_drift_detected`
- Panel 3: Pipeline runs (counter) — `agentops_agent_runs_total`
- Panel 4: Retraining count (counter) — `agentops_retraining_total`
- Panel 5: Model deployments (counter) — `agentops_deployment_swaps_total`
- Panel 6: Pipeline duration (line chart) — `agentops_agent_pipeline_duration_seconds`

---

## 7. Resume Bullets You Can Write After Completing This

**Primary bullet (the killer one):**

"Built AgentOps, an agentic MLOps platform using LangGraph to orchestrate 4 autonomous agents (data monitoring, evaluation, retraining, deployment) for a DistilBERT ticket classifier — featuring automated drift detection via Evidently AI, PyTorch fine-tuning triggered by performance degradation, zero-downtime model deployment, and an MCP server enabling AI tool integration, with Prometheus/Grafana observability tracking F1 score, drift metrics, and pipeline health in real-time."

**Shorter alternative (for space-constrained resumes):**

"Developed an agentic MLOps pipeline with LangGraph orchestrating autonomous model monitoring, drift detection (Evidently AI), automated PyTorch fine-tuning, and zero-downtime deployment — exposed via MCP server with Prometheus/Grafana observability."

---

## 8. How to Demo This in an Interview

1. Open your laptop, run `docker-compose up`
2. Show Grafana dashboard — model is healthy, F1 is 0.88, no drift
3. Run `python scripts/simulate_drift.py` — watch drift score climb in Grafana
4. Point to the moment the agent pipeline auto-triggers
5. Show MLflow — new training run appears, metrics logged
6. Show Grafana again — model swapped, F1 recovered
7. Open Claude or Cursor, show MCP integration: "How's my model doing?" → AgentOps responds with live stats
8. Total demo time: 3-4 minutes

---

## 9. Timeline

| Week | What You Build | Deliverable |
|------|---------------|-------------|
| 1 | ML model + training pipeline | DistilBERT classifier working, MLflow logging |
| 2 | Monitoring layer | Drift detection + performance tracking + Prometheus metrics |
| 3 | The 4 LangGraph agents | Full pipeline: detect → evaluate → retrain → deploy |
| 4 | FastAPI + MCP server | API serving predictions, MCP tools working |
| 5 | Docker + Grafana dashboards | Full stack running in containers, dashboards live |
| 6 | CI/CD + tests + drift simulation | GitHub Actions, demo script, README, polish |

---

## 10. Key Libraries and Docs to Reference

- LangGraph docs: https://langchain-ai.github.io/langgraph/
- FastMCP: https://github.com/jlowin/fastmcp
- Evidently AI: https://docs.evidentlyai.com/
- MLflow: https://mlflow.org/docs/latest/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- Prometheus Python Client: https://prometheus.github.io/client_python/
- FastAPI: https://fastapi.tiangolo.com/

---

## 11. GitHub README Structure

Your README should have:
1. One-line description + badges (CI, coverage, Python version)
2. Architecture diagram (use Mermaid or a PNG)
3. Quick start (docker-compose up and you're running)
4. Demo GIF showing the drift → retrain → deploy cycle
5. MCP integration section (how to connect from Claude/Cursor)
6. API reference (link to /docs)
7. Tech stack table
8. Contributing guide

This is what makes the difference between "another GitHub project" and "I'd hire this person."