"""All configuration for AgentOps — loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API
    app_name: str = "AgentOps"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Database
    database_url: str = "postgresql://agentops:agentops@localhost:5432/agentops"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # OpenAI
    openai_api_key: str = ""

    # Model
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 5
    max_token_length: int = 128
    model_path: str = "models/production"
    model_backup_dir: str = "models"

    # Training
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    fine_tune_learning_rate: float = 1e-5
    fine_tune_epochs: int = 2

    # Monitoring thresholds
    drift_threshold: float = 0.3
    degradation_threshold: float = 0.05
    baseline_f1: float = 0.85
    performance_window_size: int = 500
    min_samples_for_metrics: int = 50

    # Drift detection
    drift_check_batch_size: int = 1000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
