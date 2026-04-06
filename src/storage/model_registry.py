"""MLflow model versioning and registry helper."""

import mlflow
from mlflow.tracking import MlflowClient
import structlog
from typing import Dict, Any, Optional

from src.config import settings

logger = structlog.get_logger()


class ModelRegistry:
    """Wraps MLflow for model version management.

    Handles experiment tracking, model registration, and
    version promotion (staging → production).
    """

    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient()
        self.model_name = "ticket-classifier"

    def register_model(self, run_id: str, model_path: str = "model") -> str:
        """Register a model from an MLflow run.

        Args:
            run_id: MLflow run ID containing the model artifact
            model_path: Path within the run artifacts

        Returns:
            Model version string
        """
        model_uri = f"runs:/{run_id}/{model_path}"

        try:
            result = mlflow.register_model(model_uri, self.model_name)
            version = result.version
            logger.info("model_registered", name=self.model_name, version=version)
            return version
        except Exception as e:
            logger.error("model_registration_failed", error=str(e))
            raise

    def promote_to_production(self, version: str):
        """Promote a model version to production stage.

        Args:
            version: Model version to promote
        """
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("model_promoted", version=version, stage="Production")

    def get_production_model_uri(self) -> Optional[str]:
        """Get the URI of the current production model."""
        try:
            latest = self.client.get_latest_versions(self.model_name, stages=["Production"])
            if latest:
                return f"models:/{self.model_name}/Production"
        except Exception:
            pass
        return None

    def get_model_info(self, version: str = None) -> Dict[str, Any]:
        """Get info about a specific model version or the latest."""
        try:
            if version:
                mv = self.client.get_model_version(self.model_name, version)
            else:
                versions = self.client.get_latest_versions(self.model_name)
                if not versions:
                    return {"status": "no_models_registered"}
                mv = versions[0]

            return {
                "name": mv.name,
                "version": mv.version,
                "stage": mv.current_stage,
                "status": mv.status,
                "run_id": mv.run_id,
                "creation_timestamp": mv.creation_timestamp,
            }
        except Exception as e:
            logger.error("model_info_failed", error=str(e))
            return {"status": "error", "error": str(e)}

    def list_versions(self, limit: int = 10) -> list:
        """List recent model versions."""
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                }
                for v in sorted(versions, key=lambda x: int(x.version), reverse=True)[:limit]
            ]
        except Exception:
            return []
