"""Tests for the prediction endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestPredictEndpoint:
    """Test the prediction API endpoint."""

    def test_predict_returns_valid_response(self):
        """Test prediction endpoint returns correct structure."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (1, 0.95, [0.01, 0.95, 0.02, 0.01, 0.01])

        mock_processor = MagicMock()
        mock_processor.tokenize_single.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        with patch("src.api.app.model", mock_model), \
             patch("src.api.app.processor", mock_processor), \
             patch("src.storage.database.init_db"), \
             patch("src.storage.database.log_prediction"):
            from src.api.app import app
            client = TestClient(app)

            response = client.post(
                "/api/v1/predict",
                json={"text": "I was charged twice this month"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "label" in data
            assert "label_id" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert data["label"] == "technical"  # label_id 1 = technical
            assert data["confidence"] == 0.95

    def test_predict_requires_text(self):
        """Test prediction endpoint validates input."""
        with patch("src.api.app.model", MagicMock()), \
             patch("src.api.app.processor", MagicMock()), \
             patch("src.storage.database.init_db"):
            from src.api.app import app
            client = TestClient(app)

            response = client.post("/api/v1/predict", json={})
            assert response.status_code == 422  # Validation error
