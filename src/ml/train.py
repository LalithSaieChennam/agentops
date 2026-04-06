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
from src.config import settings

logger = structlog.get_logger()


class Trainer:
    """Handles the full training and evaluation loop.

    Integrates with MLflow for experiment tracking.
    """

    def __init__(
        self,
        model: TicketClassifier,
        learning_rate: float = None,
        batch_size: int = None,
        num_epochs: int = None,
        warmup_steps: int = None,
    ):
        self.model = model
        self.lr = learning_rate or settings.learning_rate
        self.batch_size = batch_size or settings.batch_size
        self.num_epochs = num_epochs or settings.num_epochs
        self.warmup_steps = warmup_steps or settings.warmup_steps

    def train(self, train_dataset, val_dataset, experiment_name: str = "ticket_classifier") -> Dict[str, Any]:
        """Full training loop with MLflow logging.

        Returns dict with final metrics and model path.
        """
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = AdamW(self.model.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps
        )

        with mlflow.start_run() as run:
            # Log hyperparameters
            mlflow.log_params({
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
                "model_name": settings.model_name,
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

        return {
            "best_f1": best_f1,
            "model_path": "models/best",
            "mlflow_run_id": run.info.run_id,
        }

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
