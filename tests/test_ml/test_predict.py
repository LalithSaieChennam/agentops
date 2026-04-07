"""Tests for the inference pipeline."""

from unittest.mock import MagicMock

from src.ml.predict import InferencePipeline


class TestInferencePipeline:
    """Test the InferencePipeline class."""

    def test_predict_returns_dict(self):
        """Test that predict returns a properly structured dict."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (1, 0.95, [0.01, 0.95, 0.02, 0.01, 0.01])

        mock_processor = MagicMock()
        mock_processor.tokenize_single.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        pipeline = InferencePipeline(mock_model, mock_processor)
        result = pipeline.predict("Test ticket text")

        assert "label" in result
        assert "label_id" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["label"] == "technical"
        assert result["label_id"] == 1
        assert result["confidence"] == 0.95

    def test_predict_batch(self):
        """Test batch prediction."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, 0.9, [0.9, 0.05, 0.02, 0.02, 0.01])

        mock_processor = MagicMock()
        mock_processor.tokenize_single.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        pipeline = InferencePipeline(mock_model, mock_processor)
        results = pipeline.predict_batch(["ticket 1", "ticket 2", "ticket 3"])

        assert len(results) == 3
        assert all("label" in r for r in results)
