from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Set
from urllib.parse import parse_qs, urlparse

from ..observability.metrics_exporter import render_coordinator_metrics

_SCHEMA = """
CREATE TABLE IF NOT EXISTS coordinator_entries (
    composite_key TEXT NOT NULL,
    node_id       TEXT NOT NULL,
    worker_id     TEXT NOT NULL DEFAULT '',
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    prefix_hash   TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at    REAL NOT NULL,
    PRIMARY KEY (composite_key, node_id)
);
CREATE INDEX IF NOT EXISTS idx_entries_key ON coordinator_entries (composite_key);
CREATE INDEX IF NOT EXISTS idx_entries_node ON coordinator_entries (node_id);
"""


def _safe_tenant_id(raw: str) -> str:
    tenant = str(raw or "default").strip() or "default"
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in tenant)
    return safe[:128] or "default"


def _safe_hash(raw: str) -> str:
    text = str(raw or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())[:128]


class _CoordinatorIndex:
    def __init__(self, db_path: str = "/tmp/amf_coordinator.db") -> None:
        self._mu = threading.Lock()
        self._entries_by_key: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._keys_by_node: Dict[str, Set[str]] = {}
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._load_from_db()

    def _load_from_db(self) -> None:
        cur = self._conn.execute(
            "SELECT composite_key, node_id, worker_id, tenant_id, prefix_hash, metadata_json, updated_at "
            "FROM coordinator_entries"
        )
        for row in cur.fetchall():
            key, node, worker_id, tenant_id, prefix_hash, meta_json, updated_at = row
            try:
                metadata = json.loads(meta_json) if meta_json else {}
            except Exception:
                metadata = {}
            record = {
                "node_id": node,
                "worker_id": worker_id,
                "tenant_id": tenant_id,
                "hash": prefix_hash,
                "metadata": metadata,
                "updated_at": updated_at,
            }
            self._entries_by_key.setdefault(key, {})[node] = record
            self._keys_by_node.setdefault(node, set()).add(key)

    def _persist_record(self, key: str, node: str, record: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO coordinator_entries "
            "(composite_key, node_id, worker_id, tenant_id, prefix_hash, metadata_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                node,
                record.get("worker_id", ""),
                record.get("tenant_id", "default"),
                record.get("hash", ""),
                json.dumps(record.get("metadata", {}), ensure_ascii=False),
                record.get("updated_at", time.time()),
            ),
        )
        self._conn.commit()

    def _delete_record(self, key: str, node: str) -> None:
        self._conn.execute(
            "DELETE FROM coordinator_entries WHERE composite_key=? AND node_id=?",
            (key, node),
        )
        self._conn.commit()

    def _composite(self, *, tenant_id: str, prefix_hash: str) -> str:
        return f"{_safe_tenant_id(tenant_id)}:{_safe_hash(prefix_hash)}"

    def register(
        self,
        *,
        tenant_id: str,
        prefix_hash: str,
        node_id: str,
        worker_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        key = self._composite(tenant_id=tenant_id, prefix_hash=prefix_hash)
        node = str(node_id or "").strip()
        if not key or not node:
            return
        record = {
            "node_id": node,
            "worker_id": str(worker_id or "").strip(),
            "tenant_id": _safe_tenant_id(tenant_id),
            "hash": _safe_hash(prefix_hash),
            "metadata": metadata or {},
            "updated_at": time.time(),
        }
        with self._mu:
            self._entries_by_key.setdefault(key, {})
            self._entries_by_key[key][node] = record
            self._keys_by_node.setdefault(node, set()).add(key)
            self._persist_record(key, node, record)

    def heartbeat(self, *, node_id: str, entries: List[Dict[str, Any]]) -> None:
        node = str(node_id or "").strip()
        if not node:
            return
        new_keys: Set[str] = set()
        with self._mu:
            for item in entries:
                if not isinstance(item, dict):
                    continue
                tenant_id = _safe_tenant_id(str(item.get("tenant_id", "default")))
                prefix_hash = _safe_hash(str(item.get("hash", "")))
                if not prefix_hash:
                    continue
                worker_id = str(item.get("worker_id", "") or "")
                metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                key = self._composite(tenant_id=tenant_id, prefix_hash=prefix_hash)
                new_keys.add(key)
                record = {
                    "node_id": node,
                    "worker_id": worker_id,
                    "tenant_id": tenant_id,
                    "hash": prefix_hash,
                    "metadata": metadata,
                    "updated_at": time.time(),
                }
                self._entries_by_key.setdefault(key, {})
                self._entries_by_key[key][node] = record
                self._persist_record(key, node, record)

            old_keys = self._keys_by_node.get(node, set())
            stale = old_keys - new_keys
            for key in stale:
                by_node = self._entries_by_key.get(key, {})
                by_node.pop(node, None)
                if not by_node:
                    self._entries_by_key.pop(key, None)
                self._delete_record(key, node)
            self._keys_by_node[node] = new_keys

    def lookup(self, *, tenant_id: str, prefix_hash: str) -> List[Dict[str, Any]]:
        key = self._composite(tenant_id=tenant_id, prefix_hash=prefix_hash)
        with self._mu:
            rows = list(self._entries_by_key.get(key, {}).values())
        rows.sort(
            key=lambda row: (
                -float((row.get("metadata", {}) or {}).get("hit_rate", 0.0) or 0.0),
                -int((row.get("metadata", {}) or {}).get("cache_entries", 0) or 0),
                -float(row.get("updated_at", 0.0) or 0.0),
            )
        )
        return rows

    def evict(self, *, tenant_id: str, prefix_hash: str) -> int:
        key = self._composite(tenant_id=tenant_id, prefix_hash=prefix_hash)
        with self._mu:
            rows = self._entries_by_key.pop(key, {})
            count = len(rows)
            if count > 0:
                for node in list(rows.keys()):
                    keys = self._keys_by_node.get(node, set())
                    keys.discard(key)
                    self._keys_by_node[node] = keys
                    self._delete_record(key, node)
            return count

    def stats(self) -> Dict[str, Any]:
        with self._mu:
            key_count = len(self._entries_by_key)
            node_count = len([k for k, v in self._keys_by_node.items() if v])
            row_count = sum(len(v) for v in self._entries_by_key.values())
        return {
            "keys": int(key_count),
            "nodes": int(node_count),
            "rows": int(row_count),
        }


class CoordinatorHandler(BaseHTTPRequestHandler):
    index = _CoordinatorIndex()
    _timing_mu = threading.Lock()
    _lookup_ms_total = 0.0
    _lookup_count = 0

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            obj = json.loads(body)
        except Exception:
            return {}
        return obj if isinstance(obj, dict) else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._send(200, {"status": "ok", **CoordinatorHandler.index.stats()})
            return
        if parsed.path == "/metrics":
            with CoordinatorHandler._timing_mu:
                avg_lookup_ms = (
                    CoordinatorHandler._lookup_ms_total / float(CoordinatorHandler._lookup_count)
                    if CoordinatorHandler._lookup_count > 0
                    else 0.0
                )
            text = render_coordinator_metrics(CoordinatorHandler.index.stats(), lookup_ms=avg_lookup_ms).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            self.wfile.write(text)
            return
        if parsed.path == "/lookup":
            prefix_hash = _safe_hash((query.get("hash") or [""])[0])
            tenant_id = _safe_tenant_id((query.get("tenant_id") or ["default"])[0])
            if not prefix_hash:
                self._send(400, {"error": "hash is required"})
                return
            t0 = time.time()
            rows = CoordinatorHandler.index.lookup(tenant_id=tenant_id, prefix_hash=prefix_hash)
            elapsed_ms = max(0.0, (time.time() - t0) * 1000.0)
            with CoordinatorHandler._timing_mu:
                CoordinatorHandler._lookup_ms_total += elapsed_ms
                CoordinatorHandler._lookup_count += 1
            self._send(200, {"nodes": rows})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json()
        if parsed.path == "/register":
            prefix_hash = _safe_hash(str(payload.get("hash", "")))
            tenant_id = _safe_tenant_id(str(payload.get("tenant_id", "default")))
            node_id = str(payload.get("node_id", "") or "")
            worker_id = str(payload.get("worker_id", "") or "")
            metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
            if not prefix_hash or not node_id:
                self._send(400, {"ok": False, "error": "hash and node_id required"})
                return
            CoordinatorHandler.index.register(
                tenant_id=tenant_id,
                prefix_hash=prefix_hash,
                node_id=node_id,
                worker_id=worker_id,
                metadata=metadata,
            )
            self._send(200, {"ok": True})
            return
        if parsed.path == "/heartbeat":
            node_id = str(payload.get("node_id", "") or "")
            entries = payload.get("entries", []) if isinstance(payload.get("entries", []), list) else []
            if not node_id:
                self._send(400, {"ok": False, "error": "node_id required"})
                return
            CoordinatorHandler.index.heartbeat(node_id=node_id, entries=entries)
            self._send(200, {"ok": True})
            return
        self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/evict":
            prefix_hash = _safe_hash((query.get("hash") or [""])[0])
            tenant_id = _safe_tenant_id((query.get("tenant_id") or ["default"])[0])
            if not prefix_hash:
                self._send(400, {"ok": False, "error": "hash is required"})
                return
            deleted = CoordinatorHandler.index.evict(tenant_id=tenant_id, prefix_hash=prefix_hash)
            self._send(200, {"ok": True, "deleted": int(deleted)})
            return
        self._send(404, {"error": "not found"})


def run_coordinator(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), CoordinatorHandler)
    print(f"[KORITH_AMF_COORDINATOR] serving on {host}:{port}")
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8500)
    args = parser.parse_args()
    run_coordinator(args.host, args.port)


if __name__ == "__main__":
    main()
