from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List

from ..artifacts.store import ArtifactStore
from ..ledger import db as ledger
from .router import Router
from .worker import Worker
from ..adapters.registry import AdapterRegistry


class RestoreStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_fingerprint: Dict[str, str] = {}

    def set(self, fingerprint_hash: str, snapshot_path: str) -> None:
        with self._lock:
            self._by_fingerprint[fingerprint_hash] = snapshot_path

    def get(self, fingerprint_hash: str):
        with self._lock:
            return self._by_fingerprint.get(fingerprint_hash)


def build_runtime(
    db_path: Path,
    artifacts_dir: Path,
    gpu_ids: List[int],
) -> Router:
    ledger.init_db(db_path)
    artifacts = ArtifactStore(artifacts_dir)
    registry = AdapterRegistry()
    restore_store = RestoreStore()

    workers: List[Worker] = []
    for idx, gpu_id in enumerate(gpu_ids):
        worker = Worker(
            worker_id=f"worker-{idx}",
            gpu_id=gpu_id,
            db_path=db_path,
            artifacts=artifacts,
            adapter_factory=registry.get_adapter,
            restore_store=restore_store,
        )
        worker.start()
        workers.append(worker)

    router = Router(
        db_path=db_path,
        artifacts=artifacts,
        workers=workers,
        adapter_registry=registry,
        restore_store=restore_store,
    )
    router.restore_store = restore_store
    return router


def apply_restore(router: Router, job_id: str) -> Dict[str, Any]:
    snap = ledger.get_snapshot_for_job(router.db_path, job_id)
    if not snap:
        return {"restored": False, "reason": "snapshot_not_found"}
    router.restore_store.set(snap["fingerprint_hash"], snap["snapshot_path"])
    return {"restored": True, "snapshot_path": snap["snapshot_path"]}
