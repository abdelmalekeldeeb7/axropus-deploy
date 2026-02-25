from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from . import db as sqlite_db


class LedgerStore(Protocol):
    def init(self) -> None: ...
    def insert_job(self, job_id: str, created_at: str, jobspec: Dict[str, Any], fingerprint: Dict[str, Any],
                   prompt_hash: str, fingerprint_hash: str, idempotency_key: Optional[str], status: str,
                   org_id: str = "default") -> None: ...
    def update_status(self, job_id: str, new_status: str) -> None: ...
    def insert_run(self, run_id: str, job_id: str, started_at: str, finished_at: str, exit_code: int,
                   metrics_path: str, events_path: str, output_path: str, log_path: str,
                   org_id: str = "default") -> None: ...
    def insert_snapshot(self, snapshot_id: str, job_id: str, fingerprint_hash: str, snapshot_path: str,
                        created_at: str, org_id: str = "default") -> None: ...
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]: ...
    def get_latest_run(self, job_id: str) -> Optional[Dict[str, Any]]: ...
    def list_jobs(self, limit: int, org_id: Optional[str] = None) -> list[Dict[str, Any]]: ...
    def find_job_by_fingerprint(self, fingerprint_hash: str, org_id: Optional[str] = None) -> Optional[str]: ...
    def find_job_by_idempotency(self, idempotency_key: str, org_id: Optional[str] = None) -> Optional[str]: ...
    def get_snapshot_for_job(self, job_id: str) -> Optional[Dict[str, Any]]: ...
    def create_api_key(self, key_id: str, key_hash: str, org_id: str, created_at: str,
                       rate_limit_tpm: int, rate_limit_rpm: int, permissions_json: str) -> None: ...
    def revoke_api_key(self, key_id: str, revoked_at: str) -> bool: ...
    def list_api_keys(self, org_id: Optional[str] = None) -> list[Dict[str, Any]]: ...
    def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]: ...
    def upsert_replay_governance(self, fingerprint_hash: str, replay_disabled: int,
                                 disabled_reason: Optional[str], disabled_at: Optional[str],
                                 cooldown_until: float, negative_roi_streak: int,
                                 corruption_detected: int, restore_guard_disabled: int,
                                 updated_at: str) -> None: ...
    def get_replay_governance(self, fingerprint_hash: str) -> Optional[Dict[str, Any]]: ...
    def list_replay_governance(self, limit: int = 100) -> list[Dict[str, Any]]: ...
    def upsert_spec_governance(self, fingerprint_hash: str, org_id: str, spec_disabled: int,
                               reason: Optional[str], cooldown_until: float, bad_accept_streak: int,
                               updated_at: str) -> None: ...
    def get_spec_governance(self, fingerprint_hash: str) -> Optional[Dict[str, Any]]: ...
    def list_spec_governance(self, limit: int = 100, org_id: Optional[str] = None) -> list[Dict[str, Any]]: ...
    def upsert_snapshot_location(self, fingerprint_hash: str, snapshot_id: str, org_id: str, node_id: str,
                                 worker_id: Optional[str], snapshot_path: str, size_bytes: int,
                                 created_at: str, last_used_at: str) -> None: ...
    def list_snapshot_locations(self, fingerprint_hash: str, org_id: str) -> list[Dict[str, Any]]: ...
    def mark_snapshot_location_used(self, fingerprint_hash: str, org_id: str, node_id: str,
                                    snapshot_id: str, last_used_at: str) -> None: ...
    def prune_snapshot_locations(self, max_entries_per_fingerprint: int = 8) -> int: ...


@dataclass
class SQLiteLedgerStore(LedgerStore):
    db_path: Path

    def init(self) -> None:
        sqlite_db.init_db(self.db_path)

    def insert_job(self, job_id: str, created_at: str, jobspec: Dict[str, Any], fingerprint: Dict[str, Any],
                   prompt_hash: str, fingerprint_hash: str, idempotency_key: Optional[str], status: str,
                   org_id: str = "default") -> None:
        sqlite_db.insert_job(
            self.db_path,
            job_id=job_id,
            created_at=created_at,
            jobspec=jobspec,
            fingerprint=fingerprint,
            prompt_hash=prompt_hash,
            fingerprint_hash=fingerprint_hash,
            idempotency_key=idempotency_key,
            status=status,
            org_id=org_id,
        )

    def update_status(self, job_id: str, new_status: str) -> None:
        sqlite_db.update_status(self.db_path, job_id, new_status)

    def insert_run(self, run_id: str, job_id: str, started_at: str, finished_at: str, exit_code: int,
                   metrics_path: str, events_path: str, output_path: str, log_path: str,
                   org_id: str = "default") -> None:
        sqlite_db.insert_run(self.db_path, run_id, job_id, started_at, finished_at, exit_code,
                             metrics_path, events_path, output_path, log_path, org_id=org_id)

    def insert_snapshot(self, snapshot_id: str, job_id: str, fingerprint_hash: str,
                        snapshot_path: str, created_at: str, org_id: str = "default") -> None:
        sqlite_db.insert_snapshot(
            self.db_path,
            snapshot_id,
            job_id,
            fingerprint_hash,
            snapshot_path,
            created_at,
            org_id=org_id,
        )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return sqlite_db.get_job(self.db_path, job_id)

    def get_latest_run(self, job_id: str) -> Optional[Dict[str, Any]]:
        return sqlite_db.get_latest_run(self.db_path, job_id)

    def list_jobs(self, limit: int, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
        return sqlite_db.list_jobs(self.db_path, limit=limit, org_id=org_id)

    def find_job_by_fingerprint(self, fingerprint_hash: str, org_id: Optional[str] = None) -> Optional[str]:
        return sqlite_db.find_job_by_fingerprint(self.db_path, fingerprint_hash, org_id=org_id)

    def find_job_by_idempotency(self, idempotency_key: str, org_id: Optional[str] = None) -> Optional[str]:
        return sqlite_db.find_job_by_idempotency(self.db_path, idempotency_key, org_id=org_id)

    def get_snapshot_for_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return sqlite_db.get_snapshot_for_job(self.db_path, job_id)

    def create_api_key(self, key_id: str, key_hash: str, org_id: str, created_at: str,
                       rate_limit_tpm: int, rate_limit_rpm: int, permissions_json: str) -> None:
        sqlite_db.create_api_key(
            self.db_path,
            key_id=key_id,
            key_hash=key_hash,
            org_id=org_id,
            created_at=created_at,
            rate_limit_tpm=rate_limit_tpm,
            rate_limit_rpm=rate_limit_rpm,
            permissions_json=permissions_json,
        )

    def revoke_api_key(self, key_id: str, revoked_at: str) -> bool:
        return sqlite_db.revoke_api_key(self.db_path, key_id=key_id, revoked_at=revoked_at)

    def list_api_keys(self, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
        return sqlite_db.list_api_keys(self.db_path, org_id=org_id)

    def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        return sqlite_db.get_api_key_by_hash(self.db_path, key_hash=key_hash)

    def upsert_replay_governance(self, fingerprint_hash: str, replay_disabled: int,
                                 disabled_reason: Optional[str], disabled_at: Optional[str],
                                 cooldown_until: float, negative_roi_streak: int,
                                 corruption_detected: int, restore_guard_disabled: int,
                                 updated_at: str) -> None:
        sqlite_db.upsert_replay_governance(
            self.db_path,
            fingerprint_hash=fingerprint_hash,
            replay_disabled=replay_disabled,
            disabled_reason=disabled_reason,
            disabled_at=disabled_at,
            cooldown_until=cooldown_until,
            negative_roi_streak=negative_roi_streak,
            corruption_detected=corruption_detected,
            restore_guard_disabled=restore_guard_disabled,
            updated_at=updated_at,
        )

    def get_replay_governance(self, fingerprint_hash: str) -> Optional[Dict[str, Any]]:
        return sqlite_db.get_replay_governance(self.db_path, fingerprint_hash=fingerprint_hash)

    def list_replay_governance(self, limit: int = 100) -> list[Dict[str, Any]]:
        return sqlite_db.list_replay_governance(self.db_path, limit=limit)

    def upsert_spec_governance(self, fingerprint_hash: str, org_id: str, spec_disabled: int,
                               reason: Optional[str], cooldown_until: float, bad_accept_streak: int,
                               updated_at: str) -> None:
        sqlite_db.upsert_spec_governance(
            self.db_path,
            fingerprint_hash=fingerprint_hash,
            org_id=org_id,
            spec_disabled=spec_disabled,
            reason=reason,
            cooldown_until=cooldown_until,
            bad_accept_streak=bad_accept_streak,
            updated_at=updated_at,
        )

    def get_spec_governance(self, fingerprint_hash: str) -> Optional[Dict[str, Any]]:
        return sqlite_db.get_spec_governance(self.db_path, fingerprint_hash=fingerprint_hash)

    def list_spec_governance(self, limit: int = 100, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
        return sqlite_db.list_spec_governance(self.db_path, limit=limit, org_id=org_id)

    def upsert_snapshot_location(self, fingerprint_hash: str, snapshot_id: str, org_id: str, node_id: str,
                                 worker_id: Optional[str], snapshot_path: str, size_bytes: int,
                                 created_at: str, last_used_at: str) -> None:
        sqlite_db.upsert_snapshot_location(
            self.db_path,
            fingerprint_hash=fingerprint_hash,
            snapshot_id=snapshot_id,
            org_id=org_id,
            node_id=node_id,
            worker_id=worker_id,
            snapshot_path=snapshot_path,
            size_bytes=size_bytes,
            created_at=created_at,
            last_used_at=last_used_at,
        )

    def list_snapshot_locations(self, fingerprint_hash: str, org_id: str) -> list[Dict[str, Any]]:
        return sqlite_db.list_snapshot_locations(self.db_path, fingerprint_hash=fingerprint_hash, org_id=org_id)

    def mark_snapshot_location_used(self, fingerprint_hash: str, org_id: str, node_id: str,
                                    snapshot_id: str, last_used_at: str) -> None:
        sqlite_db.mark_snapshot_location_used(
            self.db_path,
            fingerprint_hash=fingerprint_hash,
            org_id=org_id,
            node_id=node_id,
            snapshot_id=snapshot_id,
            last_used_at=last_used_at,
        )

    def prune_snapshot_locations(self, max_entries_per_fingerprint: int = 8) -> int:
        return sqlite_db.prune_snapshot_locations(
            self.db_path,
            max_entries_per_fingerprint=max_entries_per_fingerprint,
        )
