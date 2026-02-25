from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .store import ArtifactStore


class ArtifactStoreAdapter:
    def init_job(self, job_id: str, org_id: str = "default") -> Dict[str, Path]:
        raise NotImplementedError

    def store_artifact(self, job_id: str, artifact_type: str, blob: bytes, org_id: str = "default") -> str:
        raise NotImplementedError

    def get_artifact(self, job_id: str, artifact_type: str, org_id: str = "default") -> Optional[bytes]:
        raise NotImplementedError


@dataclass
class LocalArtifactStoreAdapter(ArtifactStoreAdapter):
    base_dir: Path

    def __post_init__(self) -> None:
        self._store = ArtifactStore(self.base_dir)

    def init_job(self, job_id: str, org_id: str = "default") -> Dict[str, Path]:
        return self._store.init_job(job_id, org_id=org_id)

    def store_artifact(self, job_id: str, artifact_type: str, blob: bytes, org_id: str = "default") -> str:
        paths = self._store.init_job(job_id, org_id=org_id)
        if artifact_type not in paths:
            raise ValueError(f"unknown artifact type {artifact_type}")
        path = paths[artifact_type]
        Path(path).write_bytes(blob)
        return str(path)

    def get_artifact(self, job_id: str, artifact_type: str, org_id: str = "default") -> Optional[bytes]:
        paths = self._store.init_job(job_id, org_id=org_id)
        if artifact_type not in paths:
            return None
        path = paths[artifact_type]
        if not Path(path).exists():
            return None
        return Path(path).read_bytes()
