from __future__ import annotations

import json
import sqlite3
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .base import QueueBase, QueueItem


class SQLiteQueue(QueueBase):
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._reclaim_stale_inflight()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queue (
                  job_id TEXT PRIMARY KEY,
                  payload_json TEXT NOT NULL,
                  lane TEXT NOT NULL,
                  status TEXT NOT NULL,
                  target_worker_id TEXT,
                  worker_id TEXT,
                  enqueued_at REAL NOT NULL,
                  available_at REAL NOT NULL,
                  attempts INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status, available_at)")
            # Add missing column for older DBs.
            cur = conn.execute("PRAGMA table_info(queue)")
            cols = {row[1] for row in cur.fetchall()}
            if "target_worker_id" not in cols:
                conn.execute("ALTER TABLE queue ADD COLUMN target_worker_id TEXT")
            if "lane" not in cols:
                conn.execute("ALTER TABLE queue ADD COLUMN lane TEXT NOT NULL DEFAULT 'MISS'")

    def _reclaim_stale_inflight(self) -> None:
        reclaim = os.environ.get("KORITH_QUEUE_RECLAIM_INFLIGHT", "1").strip().lower() in ("1", "true", "yes", "on")
        if not reclaim:
            return
        stale_s = max(1.0, float(os.environ.get("KORITH_QUEUE_INFLIGHT_STALE_S", "900")))
        cutoff = time.time() - stale_s
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE queue
                SET status='QUEUED', worker_id=NULL, available_at=?
                WHERE status='INFLIGHT' AND available_at <= ?
                """,
                (time.time(), cutoff),
            )

    def enqueue(self, job_id: str, payload: Dict[str, Any]) -> None:
        now = time.time()
        target = payload.get("target_worker_id")
        lane = str(payload.get("lane", "MISS")).upper()
        if lane not in ("HIT", "MISS", "SPEC_HIT", "SPEC_MISS"):
            lane = "MISS"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO queue(job_id, payload_json, lane, status, target_worker_id, worker_id, enqueued_at, available_at, attempts)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (job_id, json.dumps(payload, ensure_ascii=False), lane, "QUEUED", target, None, now, now, 0),
            )

    def dequeue(self, worker_id: str, timeout_s: float = 1.0) -> Optional[QueueItem]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            # Continuously reclaim stale inflight rows so interrupted clients/workers
            # cannot leave jobs permanently stuck.
            self._reclaim_stale_inflight()
            now = time.time()
            with sqlite3.connect(self.db_path) as conn:
                conn.isolation_level = None
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    """
                    SELECT job_id, payload_json, enqueued_at, attempts
                    FROM queue
                    WHERE status='QUEUED' AND available_at <= ?
                      AND (target_worker_id IS NULL OR target_worker_id = ?)
                    ORDER BY enqueued_at ASC
                    LIMIT 1
                    """,
                    (now, worker_id),
                )
                row = cur.fetchone()
                if not row:
                    conn.execute("COMMIT")
                    time.sleep(0.05)
                    continue
                job_id, payload_json, enqueued_at, attempts = row
                conn.execute(
                    "UPDATE queue SET status='INFLIGHT', worker_id=? WHERE job_id=?",
                    (worker_id, job_id),
                )
                conn.execute("COMMIT")
                payload = json.loads(payload_json)
                return QueueItem(job_id=job_id, payload=payload, enqueued_at=enqueued_at, attempts=attempts)
        return None

    def ack(self, job_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM queue WHERE job_id=?", (job_id,))

    def retry(self, job_id: str, delay_s: float) -> None:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE queue
                SET status='QUEUED', worker_id=NULL, available_at=?, attempts=attempts+1
                WHERE job_id=?
                """,
                (now + delay_s, job_id),
            )

    def stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT status, COUNT(*) FROM queue GROUP BY status")
            counts = {row[0]: row[1] for row in cur.fetchall()}
            cur = conn.execute(
                "SELECT lane, COUNT(*) FROM queue WHERE status='QUEUED' GROUP BY lane"
            )
            lane_counts = {row[0]: row[1] for row in cur.fetchall()}
        counts["queue_depth_hit"] = int(lane_counts.get("HIT", 0)) + int(lane_counts.get("SPEC_HIT", 0))
        counts["queue_depth_miss"] = int(lane_counts.get("MISS", 0)) + int(lane_counts.get("SPEC_MISS", 0))
        return counts
