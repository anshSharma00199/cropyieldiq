"""Model-artifact storage. 'local' = files on disk (default). 's3' = S3-compatible object storage
(AWS S3, Cloudflare R2, MinIO). Lets every API replica pull the same trained model at start-up."""

import logging
from pathlib import Path

from app.config import settings

log = logging.getLogger("storage")
FILES = ("crop_model.joblib", "model_card.json")


def _client():
    import boto3  # imported lazily: only needed when STORAGE_BACKEND=s3 (pip install boto3)

    return boto3.client("s3")


def upload_artifacts(directory: Path):
    c = _client()
    for name in FILES:
        c.upload_file(str(Path(directory) / name), settings.s3_bucket, settings.s3_prefix + name)
        log.info("uploaded %s", name)


def fetch_artifacts():
    """Download artifacts to the configured local paths when STORAGE_BACKEND=s3."""
    if settings.storage_backend != "s3":
        return
    c = _client()
    for name, dest in zip(FILES, (settings.model_path, settings.model_card_path)):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        c.download_file(settings.s3_bucket, settings.s3_prefix + name, dest)
        log.info("downloaded %s", name)
