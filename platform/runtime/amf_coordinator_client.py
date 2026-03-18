from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class AmfCoordinatorClient:
    def __init__(self, base_url: str, timeout_s: float = 0.25) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout_s = max(0.05, float(timeout_s))

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        qs = ""
        if query:
            qs = "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        url = f"{self.base_url}{path}{qs}"
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            if not body:
                return {}
            decoded = json.loads(body)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return None

    def lookup(self, prefix_hash: str, tenant_id: str) -> List[Dict[str, Any]]:
        out = self._request(
            "GET",
            "/lookup",
            query={"hash": str(prefix_hash), "tenant_id": str(tenant_id)},
        )
        if not isinstance(out, dict):
            return []
        rows = out.get("nodes", [])
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]

    def register(
        self,
        *,
        prefix_hash: str,
        tenant_id: str,
        node_id: str,
        worker_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        out = self._request(
            "POST",
            "/register",
            payload={
                "hash": str(prefix_hash),
                "tenant_id": str(tenant_id),
                "node_id": str(node_id),
                "worker_id": str(worker_id),
                "metadata": metadata or {},
            },
        )
        return bool(isinstance(out, dict) and out.get("ok", False))

    def heartbeat(self, *, node_id: str, entries: List[Dict[str, Any]]) -> bool:
        out = self._request(
            "POST",
            "/heartbeat",
            payload={
                "node_id": str(node_id),
                "entries": entries,
            },
        )
        return bool(isinstance(out, dict) and out.get("ok", False))

    def evict(self, *, prefix_hash: str, tenant_id: str) -> bool:
        out = self._request(
            "DELETE",
            "/evict",
            query={"hash": str(prefix_hash), "tenant_id": str(tenant_id)},
        )
        return bool(isinstance(out, dict) and out.get("ok", False))
