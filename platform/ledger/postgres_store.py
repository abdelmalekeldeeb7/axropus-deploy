from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PostgresLedgerStore:
    dsn: str

    def _conn(self):
        try:
            import psycopg2  # type: ignore
        except Exception as exc:
            raise RuntimeError("psycopg2 not installed") from exc
        return psycopg2.connect(self.dsn)

    def init(self) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  job_id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  org_id TEXT NOT NULL DEFAULT 'default',
                  jobspec_json TEXT NOT NULL,
                  fingerprint_json TEXT NOT NULL,
                  prompt_hash TEXT NOT NULL,
                  fingerprint_hash TEXT NOT NULL,
                  idempotency_key TEXT,
                  status TEXT NOT NULL,
                  immutable_hash TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_org_created ON jobs(org_id, created_at DESC)")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  org_id TEXT NOT NULL DEFAULT 'default',
                  started_at TEXT NOT NULL,
                  finished_at TEXT NOT NULL,
                  exit_code INTEGER NOT NULL,
                  metrics_path TEXT NOT NULL,
                  events_path TEXT NOT NULL,
                  output_path TEXT NOT NULL,
                  log_path TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                  snapshot_id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  org_id TEXT NOT NULL DEFAULT 'default',
                  fingerprint_hash TEXT NOT NULL,
                  snapshot_path TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                  key_id TEXT PRIMARY KEY,
                  key_hash TEXT NOT NULL UNIQUE,
                  org_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  revoked_at TEXT,
                  rate_limit_tpm INTEGER NOT NULL DEFAULT 120000,
                  rate_limit_rpm INTEGER NOT NULL DEFAULT 600,
                  permissions_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS replay_governance (
                  fingerprint_hash TEXT PRIMARY KEY,
                  replay_disabled INTEGER NOT NULL DEFAULT 0,
                  disabled_reason TEXT,
                  disabled_at TEXT,
                  cooldown_until DOUBLE PRECISION NOT NULL DEFAULT 0,
                  negative_roi_streak INTEGER NOT NULL DEFAULT 0,
                  corruption_detected INTEGER NOT NULL DEFAULT 0,
                  restore_guard_disabled INTEGER NOT NULL DEFAULT 0,
                  updated_at TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS spec_governance (
                  fingerprint_hash TEXT PRIMARY KEY,
                  org_id TEXT NOT NULL DEFAULT 'default',
                  spec_disabled INTEGER NOT NULL DEFAULT 0,
                  reason TEXT,
                  cooldown_until DOUBLE PRECISION NOT NULL DEFAULT 0,
                  bad_accept_streak INTEGER NOT NULL DEFAULT 0,
                  updated_at TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_spec_governance_org_updated ON spec_governance(org_id, updated_at DESC)")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot_locations (
                  fingerprint_hash TEXT NOT NULL,
                  snapshot_id TEXT NOT NULL,
                  org_id TEXT NOT NULL,
                  node_id TEXT NOT NULL,
                  worker_id TEXT,
                  snapshot_path TEXT NOT NULL,
                  size_bytes BIGINT NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  last_used_at TEXT NOT NULL,
                  PRIMARY KEY (fingerprint_hash, org_id, node_id, snapshot_id)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshot_locations_lookup "
                "ON snapshot_locations(fingerprint_hash, org_id, last_used_at DESC)"
            )

            cur.execute(
                """
                CREATE OR REPLACE FUNCTION prevent_jobspec_update()
                RETURNS trigger AS $$
                BEGIN
                  RAISE EXCEPTION 'jobspec is immutable';
                END;
                $$ LANGUAGE plpgsql;
                """
            )
            cur.execute(
                """
                CREATE OR REPLACE FUNCTION prevent_append_only()
                RETURNS trigger AS $$
                BEGIN
                  RAISE EXCEPTION 'append-only table';
                END;
                $$ LANGUAGE plpgsql;
                """
            )
            cur.execute(
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'jobspec_immutable') THEN
                    CREATE TRIGGER jobspec_immutable
                    BEFORE UPDATE OF jobspec_json ON jobs
                    FOR EACH ROW EXECUTE FUNCTION prevent_jobspec_update();
                  END IF;
                  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'runs_immutable') THEN
                    CREATE TRIGGER runs_immutable
                    BEFORE UPDATE OR DELETE ON runs
                    FOR EACH ROW EXECUTE FUNCTION prevent_append_only();
                  END IF;
                  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'snapshots_immutable') THEN
                    CREATE TRIGGER snapshots_immutable
                    BEFORE UPDATE OR DELETE ON snapshots
                    FOR EACH ROW EXECUTE FUNCTION prevent_append_only();
                  END IF;
                END
                $$;
                """
            )
            conn.commit()

    def insert_job(self, job_id: str, created_at: str, jobspec: Dict[str, Any], fingerprint: Dict[str, Any],
                   prompt_hash: str, fingerprint_hash: str, idempotency_key: Optional[str], status: str,
                   org_id: str = "default") -> None:
        immutable_hash = hashlib.sha256(json.dumps(jobspec, sort_keys=True).encode("utf-8")).hexdigest()
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO jobs(job_id, created_at, org_id, jobspec_json, fingerprint_json, prompt_hash,
                                 fingerprint_hash, idempotency_key, status, immutable_hash)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    job_id,
                    created_at,
                    org_id,
                    json.dumps(jobspec),
                    json.dumps(fingerprint),
                    prompt_hash,
                    fingerprint_hash,
                    idempotency_key,
                    status,
                    immutable_hash,
                ),
            )
            conn.commit()

    def update_status(self, job_id: str, new_status: str) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE jobs SET status=%s WHERE job_id=%s", (new_status, job_id))
            conn.commit()

    def insert_run(self, run_id: str, job_id: str, started_at: str, finished_at: str, exit_code: int,
                   metrics_path: str, events_path: str, output_path: str, log_path: str,
                   org_id: str = "default") -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO runs(run_id, job_id, org_id, started_at, finished_at, exit_code,
                                 metrics_path, events_path, output_path, log_path)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (run_id, job_id, org_id, started_at, finished_at, exit_code,
                 metrics_path, events_path, output_path, log_path),
            )
            conn.commit()

    def insert_snapshot(self, snapshot_id: str, job_id: str, fingerprint_hash: str,
                        snapshot_path: str, created_at: str, org_id: str = "default") -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO snapshots(snapshot_id, job_id, org_id, fingerprint_hash, snapshot_path, created_at)
                VALUES(%s,%s,%s,%s,%s,%s)
                """,
                (snapshot_id, job_id, org_id, fingerprint_hash, snapshot_path, created_at),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT job_id, created_at, org_id, jobspec_json, fingerprint_json, prompt_hash, fingerprint_hash, idempotency_key, status FROM jobs WHERE job_id=%s",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "job_id": row[0],
                "created_at": row[1],
                "org_id": row[2],
                "jobspec": json.loads(row[3]),
                "fingerprint": json.loads(row[4]),
                "prompt_hash": row[5],
                "fingerprint_hash": row[6],
                "idempotency_key": row[7],
                "status": row[8],
            }

    def get_latest_run(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT run_id, started_at, finished_at, exit_code, metrics_path, events_path, output_path, log_path, org_id
                FROM runs WHERE job_id=%s ORDER BY started_at DESC LIMIT 1
                """,
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "run_id": row[0],
                "started_at": row[1],
                "finished_at": row[2],
                "exit_code": row[3],
                "metrics_path": row[4],
                "events_path": row[5],
                "output_path": row[6],
                "log_path": row[7],
                "org_id": row[8],
            }

    def list_jobs(self, limit: int, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            if org_id:
                cur.execute(
                    "SELECT job_id, created_at, status, org_id FROM jobs WHERE org_id=%s ORDER BY created_at DESC LIMIT %s",
                    (org_id, limit),
                )
            else:
                cur.execute(
                    "SELECT job_id, created_at, status, org_id FROM jobs ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
            return [{"job_id": row[0], "created_at": row[1], "status": row[2], "org_id": row[3]} for row in cur.fetchall()]

    def find_job_by_fingerprint(self, fingerprint_hash: str, org_id: Optional[str] = None) -> Optional[str]:
        with self._conn() as conn:
            cur = conn.cursor()
            if org_id:
                cur.execute(
                    "SELECT job_id FROM jobs WHERE fingerprint_hash=%s AND org_id=%s ORDER BY created_at DESC LIMIT 1",
                    (fingerprint_hash, org_id),
                )
            else:
                cur.execute(
                    "SELECT job_id FROM jobs WHERE fingerprint_hash=%s ORDER BY created_at DESC LIMIT 1",
                    (fingerprint_hash,),
                )
            row = cur.fetchone()
            return row[0] if row else None

    def find_job_by_idempotency(self, idempotency_key: str, org_id: Optional[str] = None) -> Optional[str]:
        if not idempotency_key:
            return None
        with self._conn() as conn:
            cur = conn.cursor()
            if org_id:
                cur.execute(
                    "SELECT job_id FROM jobs WHERE idempotency_key=%s AND org_id=%s ORDER BY created_at DESC LIMIT 1",
                    (idempotency_key, org_id),
                )
            else:
                cur.execute(
                    "SELECT job_id FROM jobs WHERE idempotency_key=%s ORDER BY created_at DESC LIMIT 1",
                    (idempotency_key,),
                )
            row = cur.fetchone()
            return row[0] if row else None

    def get_snapshot_for_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT snapshot_id, fingerprint_hash, snapshot_path, created_at, org_id FROM snapshots WHERE job_id=%s ORDER BY created_at DESC LIMIT 1",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "snapshot_id": row[0],
                "fingerprint_hash": row[1],
                "snapshot_path": row[2],
                "created_at": row[3],
                "org_id": row[4],
            }

    def create_api_key(self, key_id: str, key_hash: str, org_id: str, created_at: str,
                       rate_limit_tpm: int, rate_limit_rpm: int, permissions_json: str) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO api_keys(key_id, key_hash, org_id, created_at, revoked_at, rate_limit_tpm, rate_limit_rpm, permissions_json)
                VALUES(%s,%s,%s,%s,NULL,%s,%s,%s)
                """,
                (key_id, key_hash, org_id, created_at, rate_limit_tpm, rate_limit_rpm, permissions_json),
            )
            conn.commit()

    def revoke_api_key(self, key_id: str, revoked_at: str) -> bool:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE api_keys SET revoked_at=%s WHERE key_id=%s AND revoked_at IS NULL", (revoked_at, key_id))
            conn.commit()
            return cur.rowcount > 0

    def list_api_keys(self, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            if org_id:
                cur.execute(
                    "SELECT key_id, org_id, created_at, revoked_at, rate_limit_tpm, rate_limit_rpm, permissions_json FROM api_keys WHERE org_id=%s ORDER BY created_at DESC",
                    (org_id,),
                )
            else:
                cur.execute(
                    "SELECT key_id, org_id, created_at, revoked_at, rate_limit_tpm, rate_limit_rpm, permissions_json FROM api_keys ORDER BY created_at DESC"
                )
            rows = cur.fetchall()
        return [
            {
                "key_id": row[0],
                "org_id": row[1],
                "created_at": row[2],
                "revoked_at": row[3],
                "rate_limit_tpm": row[4],
                "rate_limit_rpm": row[5],
                "permissions_json": row[6],
            }
            for row in rows
        ]

    def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT key_id, key_hash, org_id, created_at, revoked_at, rate_limit_tpm, rate_limit_rpm, permissions_json FROM api_keys WHERE key_hash=%s",
                (key_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "key_id": row[0],
                "key_hash": row[1],
                "org_id": row[2],
                "created_at": row[3],
                "revoked_at": row[4],
                "rate_limit_tpm": row[5],
                "rate_limit_rpm": row[6],
                "permissions_json": row[7],
            }

    def upsert_replay_governance(self, fingerprint_hash: str, replay_disabled: int,
                                 disabled_reason: Optional[str], disabled_at: Optional[str],
                                 cooldown_until: float, negative_roi_streak: int,
                                 corruption_detected: int, restore_guard_disabled: int,
                                 updated_at: str) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO replay_governance(
                    fingerprint_hash, replay_disabled, disabled_reason, disabled_at,
                    cooldown_until, negative_roi_streak, corruption_detected,
                    restore_guard_disabled, updated_at
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (fingerprint_hash) DO UPDATE SET
                    replay_disabled=EXCLUDED.replay_disabled,
                    disabled_reason=EXCLUDED.disabled_reason,
                    disabled_at=EXCLUDED.disabled_at,
                    cooldown_until=EXCLUDED.cooldown_until,
                    negative_roi_streak=EXCLUDED.negative_roi_streak,
                    corruption_detected=EXCLUDED.corruption_detected,
                    restore_guard_disabled=EXCLUDED.restore_guard_disabled,
                    updated_at=EXCLUDED.updated_at
                """,
                (
                    fingerprint_hash,
                    replay_disabled,
                    disabled_reason,
                    disabled_at,
                    cooldown_until,
                    negative_roi_streak,
                    corruption_detected,
                    restore_guard_disabled,
                    updated_at,
                ),
            )
            conn.commit()

    def get_replay_governance(self, fingerprint_hash: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT replay_disabled, disabled_reason, disabled_at, cooldown_until,
                       negative_roi_streak, corruption_detected, restore_guard_disabled, updated_at
                FROM replay_governance WHERE fingerprint_hash=%s
                """,
                (fingerprint_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "fingerprint_hash": fingerprint_hash,
                "replay_disabled": int(row[0]),
                "disabled_reason": row[1],
                "disabled_at": row[2],
                "cooldown_until": float(row[3]),
                "negative_roi_streak": int(row[4]),
                "corruption_detected": int(row[5]),
                "restore_guard_disabled": int(row[6]),
                "updated_at": row[7],
            }

    def list_replay_governance(self, limit: int = 100) -> list[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT fingerprint_hash, replay_disabled, disabled_reason, disabled_at,
                       cooldown_until, negative_roi_streak, corruption_detected,
                       restore_guard_disabled, updated_at
                FROM replay_governance ORDER BY updated_at DESC LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "fingerprint_hash": row[0],
                "replay_disabled": int(row[1]),
                "disabled_reason": row[2],
                "disabled_at": row[3],
                "cooldown_until": float(row[4]),
                "negative_roi_streak": int(row[5]),
                "corruption_detected": int(row[6]),
                "restore_guard_disabled": int(row[7]),
                "updated_at": row[8],
            }
            for row in rows
        ]

    def upsert_spec_governance(self, fingerprint_hash: str, org_id: str, spec_disabled: int,
                               reason: Optional[str], cooldown_until: float, bad_accept_streak: int,
                               updated_at: str) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO spec_governance(
                    fingerprint_hash, org_id, spec_disabled, reason, cooldown_until, bad_accept_streak, updated_at
                ) VALUES(%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (fingerprint_hash) DO UPDATE SET
                    org_id=EXCLUDED.org_id,
                    spec_disabled=EXCLUDED.spec_disabled,
                    reason=EXCLUDED.reason,
                    cooldown_until=EXCLUDED.cooldown_until,
                    bad_accept_streak=EXCLUDED.bad_accept_streak,
                    updated_at=EXCLUDED.updated_at
                """,
                (
                    fingerprint_hash,
                    org_id,
                    int(spec_disabled),
                    reason,
                    float(cooldown_until),
                    int(bad_accept_streak),
                    updated_at,
                ),
            )
            conn.commit()

    def get_spec_governance(self, fingerprint_hash: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT org_id, spec_disabled, reason, cooldown_until, bad_accept_streak, updated_at
                FROM spec_governance
                WHERE fingerprint_hash=%s
                """,
                (fingerprint_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "fingerprint_hash": fingerprint_hash,
                "org_id": row[0],
                "spec_disabled": int(row[1] or 0),
                "reason": row[2],
                "cooldown_until": float(row[3] or 0.0),
                "bad_accept_streak": int(row[4] or 0),
                "updated_at": row[5],
            }

    def list_spec_governance(self, limit: int = 100, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            if org_id:
                cur.execute(
                    """
                    SELECT fingerprint_hash, org_id, spec_disabled, reason, cooldown_until, bad_accept_streak, updated_at
                    FROM spec_governance
                    WHERE org_id=%s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (org_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT fingerprint_hash, org_id, spec_disabled, reason, cooldown_until, bad_accept_streak, updated_at
                    FROM spec_governance
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
        return [
            {
                "fingerprint_hash": row[0],
                "org_id": row[1],
                "spec_disabled": int(row[2] or 0),
                "reason": row[3],
                "cooldown_until": float(row[4] or 0.0),
                "bad_accept_streak": int(row[5] or 0),
                "updated_at": row[6],
            }
            for row in rows
        ]

    def upsert_snapshot_location(self, fingerprint_hash: str, snapshot_id: str, org_id: str, node_id: str,
                                 worker_id: Optional[str], snapshot_path: str, size_bytes: int,
                                 created_at: str, last_used_at: str) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO snapshot_locations(
                    fingerprint_hash, snapshot_id, org_id, node_id, worker_id, snapshot_path,
                    size_bytes, created_at, last_used_at
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (fingerprint_hash, org_id, node_id, snapshot_id) DO UPDATE SET
                    worker_id=EXCLUDED.worker_id,
                    snapshot_path=EXCLUDED.snapshot_path,
                    size_bytes=EXCLUDED.size_bytes,
                    created_at=EXCLUDED.created_at,
                    last_used_at=EXCLUDED.last_used_at
                """,
                (
                    fingerprint_hash,
                    snapshot_id,
                    org_id,
                    node_id,
                    worker_id,
                    snapshot_path,
                    int(size_bytes),
                    created_at,
                    last_used_at,
                ),
            )
            conn.commit()

    def list_snapshot_locations(self, fingerprint_hash: str, org_id: str) -> list[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT fingerprint_hash, snapshot_id, org_id, node_id, worker_id, snapshot_path,
                       size_bytes, created_at, last_used_at
                FROM snapshot_locations
                WHERE fingerprint_hash=%s AND org_id=%s
                ORDER BY last_used_at DESC
                """,
                (fingerprint_hash, org_id),
            )
            rows = cur.fetchall()
        return [
            {
                "fingerprint_hash": row[0],
                "snapshot_id": row[1],
                "org_id": row[2],
                "node_id": row[3],
                "worker_id": row[4],
                "snapshot_path": row[5],
                "size_bytes": int(row[6] or 0),
                "created_at": row[7],
                "last_used_at": row[8],
            }
            for row in rows
        ]

    def mark_snapshot_location_used(self, fingerprint_hash: str, org_id: str, node_id: str,
                                    snapshot_id: str, last_used_at: str) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE snapshot_locations
                SET last_used_at=%s
                WHERE fingerprint_hash=%s AND org_id=%s AND node_id=%s AND snapshot_id=%s
                """,
                (last_used_at, fingerprint_hash, org_id, node_id, snapshot_id),
            )
            conn.commit()

    def prune_snapshot_locations(self, max_entries_per_fingerprint: int = 8) -> int:
        if max_entries_per_fingerprint < 1:
            return 0
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                WITH ranked AS (
                    SELECT fingerprint_hash, org_id, node_id, snapshot_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY fingerprint_hash, org_id
                               ORDER BY last_used_at DESC
                           ) AS rn
                    FROM snapshot_locations
                )
                DELETE FROM snapshot_locations sl
                USING ranked r
                WHERE sl.fingerprint_hash=r.fingerprint_hash
                  AND sl.org_id=r.org_id
                  AND sl.node_id=r.node_id
                  AND sl.snapshot_id=r.snapshot_id
                  AND r.rn > %s
                """,
                (max_entries_per_fingerprint,),
            )
            deleted = cur.rowcount
            conn.commit()
            return int(deleted or 0)
