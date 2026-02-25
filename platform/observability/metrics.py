from __future__ import annotations

import hashlib
import os
import threading
from typing import Dict, Optional


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._org_alias: Dict[str, str] = {}
        self._org_label_mode = os.environ.get("KORITH_METRICS_ORG_LABEL_MODE", "bounded").strip().lower()
        max_labels_raw = os.environ.get("KORITH_METRICS_MAX_ORG_LABELS", "200").strip()
        self._max_org_labels = max(1, int(max_labels_raw)) if max_labels_raw.isdigit() else 200
        allowlist_raw = os.environ.get("KORITH_METRICS_ORG_ALLOWLIST", "")
        self._org_allowlist = {x.strip() for x in allowlist_raw.split(",") if x.strip()}

    def _sanitize_org_label(self, org_id: str) -> str:
        if self._org_label_mode == "raw":
            return org_id
        if self._org_label_mode == "hash":
            return f"org_{hashlib.sha1(org_id.encode('utf-8')).hexdigest()[:12]}"

        # Bounded mode caps distinct org labels to avoid high-cardinality explosions.
        if self._org_allowlist and org_id not in self._org_allowlist:
            return "other"
        if org_id in self._org_alias:
            return self._org_alias[org_id]
        if len(self._org_alias) >= self._max_org_labels:
            return "other"
        alias = f"org_{len(self._org_alias) + 1:04d}"
        self._org_alias[org_id] = alias
        return alias

    def _sanitize_labels(self, labels: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if not labels:
            return labels
        sanitized: Dict[str, str] = {}
        for key, value in labels.items():
            val = str(value)
            if key == "org_id":
                val = self._sanitize_org_label(val)
            sanitized[str(key)] = val
        return sanitized

    def _format(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return name
        parts = [f'{k}=\"{v}\"' for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(parts)}}}"

    def inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            key = self._format(name, self._sanitize_labels(labels))
            self._counters[key] = self._counters.get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            key = self._format(name, self._sanitize_labels(labels))
            self._gauges[key] = value

    def get_counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        with self._lock:
            return self._gauges.get(name, 0.0)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }

    def render_prometheus(self) -> str:
        lines = []
        with self._lock:
            for k, v in sorted(self._counters.items()):
                lines.append(f"# TYPE {k} counter")
                lines.append(f"{k} {v}")
            for k, v in sorted(self._gauges.items()):
                lines.append(f"# TYPE {k} gauge")
                lines.append(f"{k} {v}")
        return "\n".join(lines) + "\n"


GLOBAL_METRICS = MetricsRegistry()
