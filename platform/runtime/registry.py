from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List


class WorkerRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workers (
                  worker_id TEXT PRIMARY KEY,
                  node_id TEXT NOT NULL DEFAULT '',
                  host TEXT NOT NULL,
                  gpu_id INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  inflight INTEGER NOT NULL,
                  last_heartbeat REAL NOT NULL,
                  capabilities_json TEXT NOT NULL
                )
                """
            )
            cur = conn.execute("PRAGMA table_info(workers)")
            cols = {row[1] for row in cur.fetchall()}
            if "node_id" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN node_id TEXT NOT NULL DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_node ON workers(node_id, status)")

    def register(self, worker_id: str, host: str, gpu_id: int, capabilities: Dict[str, Any], node_id: str = "") -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workers(worker_id, node_id, host, gpu_id, status, inflight, last_heartbeat, capabilities_json)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (worker_id, node_id, host, gpu_id, "READY", 0, time.time(), json.dumps(capabilities)),
            )

    def _stale_cutoff(self) -> float:
        stale_s = max(1.0, float(os.environ.get("KORITH_WORKER_STALE_S", "15") or 15.0))
        return time.time() - stale_s

    def heartbeat(self, worker_id: str, inflight: int, capabilities: Dict[str, Any] | None = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            if capabilities is None:
                conn.execute(
                    "UPDATE workers SET inflight=?, last_heartbeat=?, status=? WHERE worker_id=?",
                    (inflight, time.time(), "READY", worker_id),
                )
            else:
                conn.execute(
                    "UPDATE workers SET inflight=?, last_heartbeat=?, status=?, capabilities_json=? WHERE worker_id=?",
                    (inflight, time.time(), "READY", json.dumps(capabilities, ensure_ascii=False), worker_id),
                )

    def list_workers(self) -> List[Dict[str, Any]]:
        cutoff = self._stale_cutoff()
        with sqlite3.connect(self.db_path) as conn:
            # Keep registry self-healing so routers avoid assigning dead workers.
            conn.execute("DELETE FROM workers WHERE last_heartbeat < ?", (cutoff,))
            cur = conn.execute(
                """
                SELECT worker_id, node_id, host, gpu_id, status, inflight, last_heartbeat, capabilities_json
                FROM workers
                WHERE status='READY' AND last_heartbeat >= ?
                """,
                (cutoff,),
            )
            return [
                {
                    "worker_id": row[0],
                    "node_id": row[1],
                    "host": row[2],
                    "gpu_id": row[3],
                    "status": row[4],
                    "inflight": row[5],
                    "last_heartbeat": row[6],
                    "capabilities": json.loads(row[7]),
                }
                for row in cur.fetchall()
            ]

    def get_worker(self, worker_id: str) -> Dict[str, Any] | None:
        cutoff = self._stale_cutoff()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT worker_id, node_id, host, gpu_id, status, inflight, last_heartbeat, capabilities_json FROM workers WHERE worker_id=?",
                (worker_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            if row[4] != "READY" or float(row[6]) < cutoff:
                return None
            return {
                "worker_id": row[0],
                "node_id": row[1],
                "host": row[2],
                "gpu_id": row[3],
                "status": row[4],
                "inflight": row[5],
                "last_heartbeat": row[6],
                "capabilities": json.loads(row[7]),
            }
