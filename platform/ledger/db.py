from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


STATUS_ORDER = ["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]


def _hash_json(obj: Dict[str, Any]) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cur.fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
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
        _ensure_column(conn, "jobs", "org_id", "TEXT NOT NULL DEFAULT 'default'")
        _ensure_column(conn, "jobs", "idempotency_key", "TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_org_created ON jobs(org_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_fingerprint_org ON jobs(fingerprint_hash, org_id)")

        conn.execute(
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
              log_path TEXT NOT NULL,
              FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            )
            """
        )
        _ensure_column(conn, "runs", "org_id", "TEXT NOT NULL DEFAULT 'default'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_job_id ON runs(job_id)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
              snapshot_id TEXT PRIMARY KEY,
              job_id TEXT NOT NULL,
              org_id TEXT NOT NULL DEFAULT 'default',
              fingerprint_hash TEXT NOT NULL,
              snapshot_path TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            )
            """
        )
        _ensure_column(conn, "snapshots", "org_id", "TEXT NOT NULL DEFAULT 'default'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_job_id ON snapshots(job_id)")

        conn.execute(
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_org ON api_keys(org_id)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS replay_governance (
              fingerprint_hash TEXT PRIMARY KEY,
              replay_disabled INTEGER NOT NULL DEFAULT 0,
              disabled_reason TEXT,
              disabled_at TEXT,
              cooldown_until REAL NOT NULL DEFAULT 0,
              negative_roi_streak INTEGER NOT NULL DEFAULT 0,
              corruption_detected INTEGER NOT NULL DEFAULT 0,
              restore_guard_disabled INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spec_governance (
              fingerprint_hash TEXT PRIMARY KEY,
              org_id TEXT NOT NULL DEFAULT 'default',
              spec_disabled INTEGER NOT NULL DEFAULT 0,
              reason TEXT,
              cooldown_until REAL NOT NULL DEFAULT 0,
              bad_accept_streak INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spec_governance_org_updated ON spec_governance(org_id, updated_at DESC)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_locations (
              fingerprint_hash TEXT NOT NULL,
              snapshot_id TEXT NOT NULL,
              org_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              worker_id TEXT,
              snapshot_path TEXT NOT NULL,
              size_bytes INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              last_used_at TEXT NOT NULL,
              PRIMARY KEY (fingerprint_hash, org_id, node_id, snapshot_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshot_locations_lookup "
            "ON snapshot_locations(fingerprint_hash, org_id, last_used_at DESC)"
        )

        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS jobspec_immutable
            BEFORE UPDATE OF jobspec_json ON jobs
            BEGIN
              SELECT RAISE(ABORT, 'jobspec is immutable');
            END;
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS immutable_hash_immutable
            BEFORE UPDATE OF immutable_hash ON jobs
            BEGIN
              SELECT RAISE(ABORT, 'immutable_hash is immutable');
            END;
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS runs_immutable
            BEFORE UPDATE ON runs
            BEGIN
              SELECT RAISE(ABORT, 'runs are append-only');
            END;
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS runs_no_delete
            BEFORE DELETE ON runs
            BEGIN
              SELECT RAISE(ABORT, 'runs are append-only');
            END;
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS snapshots_immutable
            BEFORE UPDATE ON snapshots
            BEGIN
              SELECT RAISE(ABORT, 'snapshots are append-only');
            END;
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS snapshots_no_delete
            BEFORE DELETE ON snapshots
            BEGIN
              SELECT RAISE(ABORT, 'snapshots are append-only');
            END;
            """
        )


def insert_job(
    db_path: Path,
    job_id: str,
    created_at: str,
    jobspec: Dict[str, Any],
    fingerprint: Dict[str, Any],
    prompt_hash: str,
    fingerprint_hash: str,
    idempotency_key: Optional[str] = None,
    status: str = "QUEUED",
    org_id: str = "default",
) -> None:
    immutable_hash = _hash_json(jobspec)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs(job_id, created_at, org_id, jobspec_json, fingerprint_json, prompt_hash,
                             fingerprint_hash, idempotency_key, status, immutable_hash)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job_id,
                created_at,
                org_id,
                json.dumps(jobspec, ensure_ascii=False),
                json.dumps(fingerprint, ensure_ascii=False),
                prompt_hash,
                fingerprint_hash,
                idempotency_key,
                status,
                immutable_hash,
            ),
        )


def get_job(db_path: Path, job_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT job_id, created_at, org_id, jobspec_json, fingerprint_json, prompt_hash, fingerprint_hash, idempotency_key, status FROM jobs WHERE job_id=?",
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


def update_status(db_path: Path, job_id: str, new_status: str) -> None:
    if new_status not in STATUS_ORDER:
        raise ValueError("invalid status")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("job not found")
        current = row[0]
        if current == new_status:
            return
        if current not in STATUS_ORDER:
            raise ValueError("invalid current status")
        if STATUS_ORDER.index(new_status) < STATUS_ORDER.index(current):
            raise ValueError("status regression not allowed")
        conn.execute("UPDATE jobs SET status=? WHERE job_id=?", (new_status, job_id))


def insert_run(
    db_path: Path,
    run_id: str,
    job_id: str,
    started_at: str,
    finished_at: str,
    exit_code: int,
    metrics_path: str,
    events_path: str,
    output_path: str,
    log_path: str,
    org_id: str = "default",
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs(run_id, job_id, org_id, started_at, finished_at, exit_code,
                             metrics_path, events_path, output_path, log_path)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                job_id,
                org_id,
                started_at,
                finished_at,
                exit_code,
                metrics_path,
                events_path,
                output_path,
                log_path,
            ),
        )


def insert_snapshot(
    db_path: Path,
    snapshot_id: str,
    job_id: str,
    fingerprint_hash: str,
    snapshot_path: str,
    created_at: str,
    org_id: str = "default",
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO snapshots(snapshot_id, job_id, org_id, fingerprint_hash, snapshot_path, created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (snapshot_id, job_id, org_id, fingerprint_hash, snapshot_path, created_at),
        )


def find_job_by_fingerprint(db_path: Path, fingerprint_hash: str, org_id: Optional[str] = None) -> Optional[str]:
    with sqlite3.connect(db_path) as conn:
        if org_id:
            cur = conn.execute(
                "SELECT job_id FROM jobs WHERE fingerprint_hash=? AND org_id=? ORDER BY created_at DESC LIMIT 1",
                (fingerprint_hash, org_id),
            )
        else:
            cur = conn.execute(
                "SELECT job_id FROM jobs WHERE fingerprint_hash=? ORDER BY created_at DESC LIMIT 1",
                (fingerprint_hash,),
            )
        row = cur.fetchone()
        if not row:
            return None
        return row[0]


def find_job_by_idempotency(db_path: Path, idempotency_key: str, org_id: Optional[str] = None) -> Optional[str]:
    if not idempotency_key:
        return None
    with sqlite3.connect(db_path) as conn:
        if org_id:
            cur = conn.execute(
                "SELECT job_id FROM jobs WHERE idempotency_key=? AND org_id=? ORDER BY created_at DESC LIMIT 1",
                (idempotency_key, org_id),
            )
        else:
            cur = conn.execute(
                "SELECT job_id FROM jobs WHERE idempotency_key=? ORDER BY created_at DESC LIMIT 1",
                (idempotency_key,),
            )
        row = cur.fetchone()
        if not row:
            return None
        return row[0]


def get_snapshot_for_job(db_path: Path, job_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT snapshot_id, fingerprint_hash, snapshot_path, created_at, org_id FROM snapshots WHERE job_id=? ORDER BY created_at DESC LIMIT 1",
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


def list_jobs(db_path: Path, limit: int = 20, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        if org_id:
            cur = conn.execute(
                "SELECT job_id, created_at, status, org_id FROM jobs WHERE org_id=? ORDER BY created_at DESC LIMIT ?",
                (org_id, limit),
            )
        else:
            cur = conn.execute(
                "SELECT job_id, created_at, status, org_id FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [{"job_id": row[0], "created_at": row[1], "status": row[2], "org_id": row[3]} for row in cur.fetchall()]


def get_latest_run(db_path: Path, job_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT run_id, started_at, finished_at, exit_code,
                   metrics_path, events_path, output_path, log_path, org_id
            FROM runs WHERE job_id=? ORDER BY started_at DESC LIMIT 1
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


def create_api_key(
    db_path: Path,
    key_id: str,
    key_hash: str,
    org_id: str,
    created_at: str,
    rate_limit_tpm: int,
    rate_limit_rpm: int,
    permissions_json: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO api_keys(key_id, key_hash, org_id, created_at, revoked_at, rate_limit_tpm, rate_limit_rpm, permissions_json)
            VALUES(?,?,?,?,NULL,?,?,?)
            """,
            (key_id, key_hash, org_id, created_at, rate_limit_tpm, rate_limit_rpm, permissions_json),
        )


def revoke_api_key(db_path: Path, key_id: str, revoked_at: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("UPDATE api_keys SET revoked_at=? WHERE key_id=? AND revoked_at IS NULL", (revoked_at, key_id))
        return cur.rowcount > 0


def list_api_keys(db_path: Path, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        if org_id:
            cur = conn.execute(
                "SELECT key_id, org_id, created_at, revoked_at, rate_limit_tpm, rate_limit_rpm, permissions_json FROM api_keys WHERE org_id=? ORDER BY created_at DESC",
                (org_id,),
            )
        else:
            cur = conn.execute(
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


def get_api_key_by_hash(db_path: Path, key_hash: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT key_id, key_hash, org_id, created_at, revoked_at, rate_limit_tpm, rate_limit_rpm, permissions_json FROM api_keys WHERE key_hash=?",
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


def upsert_replay_governance(
    db_path: Path,
    fingerprint_hash: str,
    replay_disabled: int,
    disabled_reason: Optional[str],
    disabled_at: Optional[str],
    cooldown_until: float,
    negative_roi_streak: int,
    corruption_detected: int,
    restore_guard_disabled: int,
    updated_at: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO replay_governance(
                fingerprint_hash, replay_disabled, disabled_reason, disabled_at,
                cooldown_until, negative_roi_streak, corruption_detected,
                restore_guard_disabled, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(fingerprint_hash) DO UPDATE SET
                replay_disabled=excluded.replay_disabled,
                disabled_reason=excluded.disabled_reason,
                disabled_at=excluded.disabled_at,
                cooldown_until=excluded.cooldown_until,
                negative_roi_streak=excluded.negative_roi_streak,
                corruption_detected=excluded.corruption_detected,
                restore_guard_disabled=excluded.restore_guard_disabled,
                updated_at=excluded.updated_at
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


def get_replay_governance(db_path: Path, fingerprint_hash: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT replay_disabled, disabled_reason, disabled_at, cooldown_until,
                   negative_roi_streak, corruption_detected, restore_guard_disabled, updated_at
            FROM replay_governance WHERE fingerprint_hash=?
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


def list_replay_governance(db_path: Path, limit: int = 100) -> list[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT fingerprint_hash, replay_disabled, disabled_reason, disabled_at,
                   cooldown_until, negative_roi_streak, corruption_detected,
                   restore_guard_disabled, updated_at
            FROM replay_governance ORDER BY updated_at DESC LIMIT ?
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


def upsert_spec_governance(
    db_path: Path,
    fingerprint_hash: str,
    org_id: str,
    spec_disabled: int,
    reason: Optional[str],
    cooldown_until: float,
    bad_accept_streak: int,
    updated_at: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO spec_governance(
                fingerprint_hash, org_id, spec_disabled, reason, cooldown_until,
                bad_accept_streak, updated_at
            ) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(fingerprint_hash) DO UPDATE SET
                org_id=excluded.org_id,
                spec_disabled=excluded.spec_disabled,
                reason=excluded.reason,
                cooldown_until=excluded.cooldown_until,
                bad_accept_streak=excluded.bad_accept_streak,
                updated_at=excluded.updated_at
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


def get_spec_governance(db_path: Path, fingerprint_hash: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT org_id, spec_disabled, reason, cooldown_until, bad_accept_streak, updated_at
            FROM spec_governance
            WHERE fingerprint_hash=?
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


def list_spec_governance(db_path: Path, limit: int = 100, org_id: Optional[str] = None) -> list[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        if org_id:
            cur = conn.execute(
                """
                SELECT fingerprint_hash, org_id, spec_disabled, reason, cooldown_until, bad_accept_streak, updated_at
                FROM spec_governance
                WHERE org_id=?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (org_id, int(limit)),
            )
        else:
            cur = conn.execute(
                """
                SELECT fingerprint_hash, org_id, spec_disabled, reason, cooldown_until, bad_accept_streak, updated_at
                FROM spec_governance
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
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


def upsert_snapshot_location(
    db_path: Path,
    fingerprint_hash: str,
    snapshot_id: str,
    org_id: str,
    node_id: str,
    worker_id: Optional[str],
    snapshot_path: str,
    size_bytes: int,
    created_at: str,
    last_used_at: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO snapshot_locations(
              fingerprint_hash, snapshot_id, org_id, node_id, worker_id, snapshot_path,
              size_bytes, created_at, last_used_at
            ) VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(fingerprint_hash, org_id, node_id, snapshot_id) DO UPDATE SET
              worker_id=excluded.worker_id,
              snapshot_path=excluded.snapshot_path,
              size_bytes=excluded.size_bytes,
              created_at=excluded.created_at,
              last_used_at=excluded.last_used_at
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


def list_snapshot_locations(
    db_path: Path,
    fingerprint_hash: str,
    org_id: str,
) -> list[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT fingerprint_hash, snapshot_id, org_id, node_id, worker_id, snapshot_path,
                   size_bytes, created_at, last_used_at
            FROM snapshot_locations
            WHERE fingerprint_hash=? AND org_id=?
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


def mark_snapshot_location_used(
    db_path: Path,
    fingerprint_hash: str,
    org_id: str,
    node_id: str,
    snapshot_id: str,
    last_used_at: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE snapshot_locations
            SET last_used_at=?
            WHERE fingerprint_hash=? AND org_id=? AND node_id=? AND snapshot_id=?
            """,
            (last_used_at, fingerprint_hash, org_id, node_id, snapshot_id),
        )


def prune_snapshot_locations(
    db_path: Path,
    max_entries_per_fingerprint: int = 8,
) -> int:
    if max_entries_per_fingerprint < 1:
        return 0
    deleted = 0
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT fingerprint_hash, org_id, snapshot_id
            FROM (
              SELECT fingerprint_hash, org_id, snapshot_id,
                     ROW_NUMBER() OVER (
                       PARTITION BY fingerprint_hash, org_id
                       ORDER BY last_used_at DESC
                     ) AS rn
              FROM snapshot_locations
            )
            WHERE rn > ?
            """,
            (max_entries_per_fingerprint,),
        )
        rows = cur.fetchall()
        for fp, org_id, snapshot_id in rows:
            conn.execute(
                "DELETE FROM snapshot_locations WHERE fingerprint_hash=? AND org_id=? AND snapshot_id=?",
                (fp, org_id, snapshot_id),
            )
            deleted += 1
    return deleted
