from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class NodeRegistry:
    def __init__(
        self,
        sqlite_path: Path,
        redis_url: str = "",
        key_prefix: str = "korith:nodes",
    ) -> None:
        self.sqlite_path = Path(sqlite_path).resolve()
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._redis = None
        self._key_prefix = key_prefix
        if redis_url:
            try:
                import redis  # type: ignore

                self._redis = redis.Redis.from_url(redis_url)
                self._redis.ping()
            except Exception:
                self._redis = None
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                  node_id TEXT PRIMARY KEY,
                  host TEXT NOT NULL,
                  router_port INTEGER NOT NULL,
                  gpu_count INTEGER NOT NULL,
                  inflight INTEGER NOT NULL,
                  queue_depth_hit INTEGER NOT NULL,
                  queue_depth_miss INTEGER NOT NULL,
                  capabilities_json TEXT NOT NULL,
                  last_heartbeat REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_last_hb ON nodes(last_heartbeat DESC)")

    def register_or_heartbeat(
        self,
        node_id: str,
        host: str,
        router_port: int,
        gpu_count: int,
        inflight: int,
        queue_depth_hit: int,
        queue_depth_miss: int,
        capabilities: Dict[str, Any],
    ) -> None:
        now = time.time()
        payload = {
            "node_id": node_id,
            "host": host,
            "router_port": int(router_port),
            "gpu_count": int(gpu_count),
            "inflight": int(inflight),
            "queue_depth_hit": int(queue_depth_hit),
            "queue_depth_miss": int(queue_depth_miss),
            "capabilities": capabilities,
            "last_heartbeat": now,
        }
        if self._redis is not None:
            key = f"{self._key_prefix}:{node_id}"
            self._redis.set(key, json.dumps(payload, ensure_ascii=False), ex=30)
            self._redis.sadd(f"{self._key_prefix}:ids", node_id)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT INTO nodes(node_id, host, router_port, gpu_count, inflight, queue_depth_hit,
                                  queue_depth_miss, capabilities_json, last_heartbeat)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(node_id) DO UPDATE SET
                  host=excluded.host,
                  router_port=excluded.router_port,
                  gpu_count=excluded.gpu_count,
                  inflight=excluded.inflight,
                  queue_depth_hit=excluded.queue_depth_hit,
                  queue_depth_miss=excluded.queue_depth_miss,
                  capabilities_json=excluded.capabilities_json,
                  last_heartbeat=excluded.last_heartbeat
                """,
                (
                    node_id,
                    host,
                    int(router_port),
                    int(gpu_count),
                    int(inflight),
                    int(queue_depth_hit),
                    int(queue_depth_miss),
                    json.dumps(capabilities, ensure_ascii=False),
                    now,
                ),
            )

    def list_nodes(self, healthy_only: bool = True, max_stale_s: float = 30.0) -> List[Dict[str, Any]]:
        if self._redis is not None:
            now = time.time()
            rows: List[Dict[str, Any]] = []
            try:
                ids = self._redis.smembers(f"{self._key_prefix}:ids")
                for raw in ids:
                    node_id = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    val = self._redis.get(f"{self._key_prefix}:{node_id}")
                    if not val:
                        continue
                    payload = json.loads(val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val))
                    if healthy_only and (now - float(payload.get("last_heartbeat", 0.0) or 0.0)) > max_stale_s:
                        continue
                    rows.append(payload)
                rows.sort(key=lambda x: (int(x.get("inflight", 0)), int(x.get("queue_depth_miss", 0))))
                return rows
            except Exception:
                pass

        now = time.time()
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute(
                """
                SELECT node_id, host, router_port, gpu_count, inflight, queue_depth_hit,
                       queue_depth_miss, capabilities_json, last_heartbeat
                FROM nodes
                ORDER BY inflight ASC, queue_depth_miss ASC, last_heartbeat DESC
                """
            )
            rows = []
            for row in cur.fetchall():
                payload = {
                    "node_id": row[0],
                    "host": row[1],
                    "router_port": int(row[2]),
                    "gpu_count": int(row[3]),
                    "inflight": int(row[4]),
                    "queue_depth_hit": int(row[5]),
                    "queue_depth_miss": int(row[6]),
                    "capabilities": json.loads(row[7] or "{}"),
                    "last_heartbeat": float(row[8] or 0.0),
                }
                if healthy_only and (now - payload["last_heartbeat"]) > max_stale_s:
                    continue
                rows.append(payload)
            return rows

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for row in self.list_nodes(healthy_only=False, max_stale_s=1e9):
            if row.get("node_id") == node_id:
                return row
        return None
