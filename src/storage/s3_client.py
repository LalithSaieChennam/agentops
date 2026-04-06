"""AWS S3 client for model artifact storage (optional).

This module is used when deploying to AWS for storing model
artifacts in S3 instead of local disk.
"""

import structlog
from pathlib import Path
from typing import Optional

logger = structlog.get_logger()

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


class S3Client:
    """Manages model artifacts in AWS S3.

    Falls back gracefully if boto3 is not installed or
    AWS credentials are not configured.
    """

    def __init__(self, bucket_name: str = "agentops-models", region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self._client = None

        if S3_AVAILABLE:
            try:
                self._client = boto3.client("s3", region_name=region)
                logger.info("s3_client_initialized", bucket=bucket_name)
            except NoCredentialsError:
                logger.warning("s3_no_credentials_found")
        else:
            logger.info("s3_not_available_boto3_not_installed")

    @property
    def is_available(self) -> bool:
        """Check if S3 is configured and accessible."""
        return self._client is not None

    def upload_model(self, local_path: str, s3_key: str) -> Optional[str]:
        """Upload a model directory to S3.

        Args:
            local_path: Local path to model directory
            s3_key: S3 key prefix for the model

        Returns:
            S3 URI if successful, None otherwise
        """
        if not self.is_available:
            logger.warning("s3_upload_skipped_not_available")
            return None

        local = Path(local_path)
        if not local.exists():
            logger.error("s3_upload_local_path_not_found", path=local_path)
            return None

        try:
            for file_path in local.rglob("*"):
                if file_path.is_file():
                    key = f"{s3_key}/{file_path.relative_to(local)}"
                    self._client.upload_file(str(file_path), self.bucket_name, key)

            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info("model_uploaded_to_s3", uri=s3_uri)
            return s3_uri

        except ClientError as e:
            logger.error("s3_upload_failed", error=str(e))
            return None

    def download_model(self, s3_key: str, local_path: str) -> bool:
        """Download a model from S3 to local disk.

        Args:
            s3_key: S3 key prefix for the model
            local_path: Local path to download to

        Returns:
            True if successful
        """
        if not self.is_available:
            logger.warning("s3_download_skipped_not_available")
            return False

        try:
            local = Path(local_path)
            local.mkdir(parents=True, exist_ok=True)

            paginator = self._client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_key)

            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative = key[len(s3_key):].lstrip("/")
                    if relative:
                        target = local / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        self._client.download_file(self.bucket_name, key, str(target))

            logger.info("model_downloaded_from_s3", s3_key=s3_key, local_path=local_path)
            return True

        except ClientError as e:
            logger.error("s3_download_failed", error=str(e))
            return False

    def list_models(self, prefix: str = "models/") -> list:
        """List model artifacts in S3."""
        if not self.is_available:
            return []

        try:
            response = self._client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix, Delimiter="/"
            )
            return [p["Prefix"] for p in response.get("CommonPrefixes", [])]
        except ClientError:
            return []
