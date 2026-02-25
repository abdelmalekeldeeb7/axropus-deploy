from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .adapter import ArtifactStoreAdapter
from .store import ArtifactStore


@dataclass
class MinioArtifactStore(ArtifactStoreAdapter):
    base_dir: Path
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    prefix: str = "korith"
    secure: bool = True

    def __post_init__(self) -> None:
        try:
            from minio import Minio  # type: ignore
        except Exception as exc:
            raise RuntimeError("minio library not installed") from exc
        self._client = Minio(self.endpoint, access_key=self.access_key, secret_key=self.secret_key, secure=self.secure)
        self._local = ArtifactStore(self.base_dir)

    def _key(self, job_id: str, artifact_type: str, org_id: str = "default") -> str:
        return f"{self.prefix}/{org_id}/{job_id}/{artifact_type}"

    def init_job(self, job_id: str, org_id: str = "default") -> Dict[str, Path]:
        return self._local.init_job(job_id, org_id=org_id)

    def store_artifact(self, job_id: str, artifact_type: str, blob: bytes, org_id: str = "default") -> str:
        key = self._key(job_id, artifact_type, org_id=org_id)
        self._client.put_object(self.bucket, key, data=io.BytesIO(blob), length=len(blob))
        return key

    def get_artifact(self, job_id: str, artifact_type: str, org_id: str = "default") -> Optional[bytes]:
        key = self._key(job_id, artifact_type, org_id=org_id)
        try:
            obj = self._client.get_object(self.bucket, key)
        except Exception:
            return None
        return obj.read()
