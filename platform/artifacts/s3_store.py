from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .adapter import ArtifactStoreAdapter
from .store import ArtifactStore


@dataclass
class S3ArtifactStore(ArtifactStoreAdapter):
    base_dir: Path
    bucket: str
    prefix: str = "korith"

    def __post_init__(self) -> None:
        try:
            import boto3  # type: ignore
        except Exception as exc:
            raise RuntimeError("boto3 not installed") from exc
        self._s3 = boto3.client("s3")
        self._local = ArtifactStore(self.base_dir)

    def _key(self, job_id: str, artifact_type: str, org_id: str = "default") -> str:
        return f"{self.prefix}/{org_id}/{job_id}/{artifact_type}"

    def init_job(self, job_id: str, org_id: str = "default") -> Dict[str, Path]:
        return self._local.init_job(job_id, org_id=org_id)

    def store_artifact(self, job_id: str, artifact_type: str, blob: bytes, org_id: str = "default") -> str:
        key = self._key(job_id, artifact_type, org_id=org_id)
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=blob)
        return key

    def get_artifact(self, job_id: str, artifact_type: str, org_id: str = "default") -> Optional[bytes]:
        key = self._key(job_id, artifact_type, org_id=org_id)
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
        except Exception:
            return None
        return resp["Body"].read()
