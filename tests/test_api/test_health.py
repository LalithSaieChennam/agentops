"""Tests for the health endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self):
        """Test health endpoint returns 200."""
        with patch("src.api.app.model", MagicMock()), \
             patch("src.api.app.processor", MagicMock()), \
             patch("src.storage.database.init_db"):
            from src.api.app import app
            client = TestClient(app)
            response = client.get("/api/v1/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert data["model_loaded"] is True
            assert data["version"] == "1.0.0"
