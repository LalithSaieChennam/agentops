"""Tests for TicketClassifier model wrapper."""

from unittest.mock import MagicMock, patch

import torch

from src.ml.model import TicketClassifier


class TestTicketClassifier:
    """Test the TicketClassifier model wrapper."""

    def test_init_default(self):
        """Test model initializes with default parameters."""
        with patch("src.ml.model.DistilBertForSequenceClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value.to.return_value = mock_model.from_pretrained.return_value

            classifier = TicketClassifier(device="cpu")
            assert classifier.num_labels == 5
            assert classifier.device == "cpu"

    def test_predict_returns_tuple(self):
        """Test that predict returns (label_id, confidence, probabilities)."""
        with patch("src.ml.model.DistilBertForSequenceClassification") as mock_model_cls:
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model

            # Mock model output
            mock_logits = torch.tensor([[0.1, 0.8, 0.05, 0.03, 0.02]])
            mock_output = MagicMock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output
            mock_model.eval = MagicMock()

            classifier = TicketClassifier(device="cpu")
            input_ids = torch.tensor([[1, 2, 3]])
            attention_mask = torch.tensor([[1, 1, 1]])

            label_id, confidence, probs = classifier.predict(input_ids, attention_mask)

            assert isinstance(label_id, int)
            assert isinstance(confidence, float)
            assert isinstance(probs, list)
            assert len(probs) == 5
            assert 0 <= confidence <= 1

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates the output directory."""
        with patch("src.ml.model.DistilBertForSequenceClassification") as mock_model_cls:
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model

            classifier = TicketClassifier(device="cpu")
            save_path = str(tmp_path / "test_model")
            classifier.save(save_path)

            mock_model.save_pretrained.assert_called_once_with(save_path)

    def test_load_from_path(self):
        """Test loading a model from a path."""
        with patch("src.ml.model.DistilBertForSequenceClassification") as mock_model_cls:
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model

            classifier = TicketClassifier(device="cpu")
            classifier.load("models/test")

            # Should call from_pretrained with the path
            assert mock_model_cls.from_pretrained.call_count == 2  # init + load
