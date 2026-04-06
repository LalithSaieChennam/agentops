"""Tests for the Trainer class."""

import pytest
from unittest.mock import patch, MagicMock
import torch
from src.ml.train import Trainer


class TestTrainer:
    """Test the training pipeline."""

    def _make_mock_model(self):
        """Create a properly configured mock model."""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.model = MagicMock()
        mock_model.model.parameters.return_value = [torch.randn(2, 2, requires_grad=True)]
        mock_model.model.train = MagicMock()
        mock_model.model.eval = MagicMock()
        return mock_model

    def _make_fake_batch(self, num_classes=5, batch_size=10):
        """Create a fake batch with all classes represented."""
        labels = torch.tensor([i % num_classes for i in range(batch_size)])
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, 16)),
            "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
            "label": labels,
        }

    def test_trainer_init(self):
        """Test Trainer initializes with correct parameters."""
        mock_model = MagicMock()
        trainer = Trainer(
            model=mock_model,
            learning_rate=1e-5,
            batch_size=8,
            num_epochs=2,
            warmup_steps=50,
        )

        assert trainer.lr == 1e-5
        assert trainer.batch_size == 8
        assert trainer.num_epochs == 2
        assert trainer.warmup_steps == 50

    def test_trainer_default_params(self):
        """Test Trainer uses config defaults when no params given."""
        mock_model = MagicMock()
        trainer = Trainer(model=mock_model)

        assert trainer.lr is not None
        assert trainer.batch_size is not None
        assert trainer.num_epochs is not None
        assert trainer.warmup_steps is not None

    @patch("src.ml.train.mlflow")
    def test_train_calls_mlflow(self, mock_mlflow):
        """Test that training integrates with MLflow for experiment tracking."""
        mock_model = self._make_mock_model()

        mock_outputs = MagicMock()
        mock_outputs.loss = MagicMock()
        mock_outputs.loss.item.return_value = 0.5
        mock_outputs.loss.backward = MagicMock()
        # Produce logits for all 5 classes with batch_size 10
        mock_outputs.logits = torch.randn(10, 5)
        mock_model.model.return_value = mock_outputs

        fake_batch = self._make_fake_batch()

        trainer = Trainer(model=mock_model, num_epochs=1, batch_size=10, warmup_steps=0)

        with patch("src.ml.train.DataLoader") as mock_dl:
            mock_dl.return_value = [fake_batch]

            mock_run = MagicMock()
            mock_run.info.run_id = "test-run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            result = trainer.train(
                MagicMock(__len__=MagicMock(return_value=10)),
                MagicMock(__len__=MagicMock(return_value=10)),
                experiment_name="test_exp",
            )

        mock_mlflow.set_experiment.assert_called_once_with("test_exp")
        mock_mlflow.log_params.assert_called_once()
        assert mock_mlflow.log_metric.call_count >= 1

    @patch("src.ml.train.mlflow")
    def test_train_returns_metrics(self, mock_mlflow):
        """Test that train() returns a dict with required keys."""
        mock_model = self._make_mock_model()

        mock_outputs = MagicMock()
        mock_outputs.loss = MagicMock()
        mock_outputs.loss.item.return_value = 0.3
        mock_outputs.loss.backward = MagicMock()
        mock_outputs.logits = torch.randn(10, 5)
        mock_model.model.return_value = mock_outputs

        fake_batch = self._make_fake_batch()

        trainer = Trainer(model=mock_model, num_epochs=1, batch_size=10, warmup_steps=0)

        with patch("src.ml.train.DataLoader") as mock_dl:
            mock_dl.return_value = [fake_batch]
            mock_run = MagicMock()
            mock_run.info.run_id = "run-456"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            result = trainer.train(
                MagicMock(__len__=MagicMock(return_value=10)),
                MagicMock(__len__=MagicMock(return_value=10)),
            )

        assert "best_f1" in result
        assert "model_path" in result
        assert "mlflow_run_id" in result
        assert isinstance(result["best_f1"], float)
        assert result["model_path"] == "models/best"

    def test_evaluate_returns_metrics(self):
        """Test that _evaluate computes correct metric structure."""
        mock_model = self._make_mock_model()

        # Create logits for 10 samples covering all 5 classes — predict correct class
        logits = torch.zeros(10, 5)
        for i in range(10):
            logits[i, i % 5] = 10.0  # High score for correct class

        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.model.return_value = mock_outputs

        trainer = Trainer(model=mock_model)

        fake_batch = self._make_fake_batch()  # labels = [0,1,2,3,4,0,1,2,3,4]

        metrics = trainer._evaluate([fake_batch])

        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert "f1_per_class" in metrics
        assert "full_report" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["accuracy"] == 1.0  # All predictions correct

    def test_evaluate_with_wrong_predictions(self):
        """Test _evaluate computes lower metrics when predictions are wrong."""
        mock_model = self._make_mock_model()

        # All predict class 0 — but labels cycle through all 5 classes
        logits = torch.zeros(10, 5)
        logits[:, 0] = 10.0  # Always predict class 0

        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.model.return_value = mock_outputs

        trainer = Trainer(model=mock_model)

        fake_batch = self._make_fake_batch()  # labels = [0,1,2,3,4,0,1,2,3,4]

        metrics = trainer._evaluate([fake_batch])
        # Only 2 out of 10 are class 0, so accuracy = 0.2
        assert metrics["accuracy"] == pytest.approx(0.2, abs=0.01)
