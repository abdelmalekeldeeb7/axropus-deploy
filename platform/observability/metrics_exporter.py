from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _fmt(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> str:
    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}} {float(value)}"
    return f"{name} {float(value)}"


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def render_worker_amf_metrics(snapshot: Dict[str, Any]) -> str:
    if not isinstance(snapshot, dict):
        return ""
    node = str(snapshot.get("node_id", "") or "")
    lines: List[str] = []
    hit_rate = _safe_float(snapshot.get("hit_rate", 0.0))
    entries = _safe_float(snapshot.get("entries", 0.0))
    storage = _safe_float(snapshot.get("storage_bytes", 0.0))
    savings = _safe_float(snapshot.get("savings_ms_total", 0.0))
    requests = _safe_float(snapshot.get("requests_total", 0.0))
    warmth = _safe_float(snapshot.get("warmth", 0.0))

    lines.append(_fmt("korith_amf_hit_rate_total", hit_rate))
    lines.append(_fmt("korith_amf_entries_total", entries))
    lines.append(_fmt("korith_amf_storage_bytes_total", storage))
    lines.append(_fmt("korith_amf_savings_ms_total", savings))
    lines.append(_fmt("korith_amf_requests_total", requests))
    lines.append(_fmt("korith_amf_node_hit_rate", hit_rate, {"node": node}))
    lines.append(_fmt("korith_amf_node_entries", entries, {"node": node}))
    lines.append(_fmt("korith_amf_node_storage_bytes", storage, {"node": node}))
    lines.append(_fmt("korith_amf_node_warmth", warmth, {"node": node}))

    tenants = snapshot.get("tenants", {}) if isinstance(snapshot.get("tenants", {}), dict) else {}
    for tenant, row in tenants.items():
        if not isinstance(row, dict):
            continue
        lines.append(_fmt("korith_amf_tenant_hit_rate", _safe_float(row.get("hit_rate", 0.0)), {"tenant": str(tenant)}))
        lines.append(_fmt("korith_amf_tenant_savings_ms", _safe_float(row.get("savings_ms", 0.0)), {"tenant": str(tenant)}))
        lines.append(_fmt("korith_amf_tenant_requests", _safe_float(row.get("requests", 0.0)), {"tenant": str(tenant)}))
        lines.append(_fmt("korith_amf_tenant_cache_bytes", storage, {"tenant": str(tenant)}))

    return "\n".join(lines) + ("\n" if lines else "")


def render_router_amf_metrics(workers: Iterable[Dict[str, Any]]) -> str:
    rows = [w for w in workers if isinstance(w, dict)]
    if not rows:
        return ""
    lines: List[str] = []
    total_entries = 0.0
    total_storage = 0.0
    total_hit = 0.0
    for row in rows:
        caps = row.get("capabilities", {}) if isinstance(row.get("capabilities", {}), dict) else {}
        node = str(row.get("node_id", "") or caps.get("node_id", "") or "")
        hit = _safe_float(caps.get("amf_hit_rate", 0.0))
        entries = _safe_float(caps.get("amf_cache_entries", 0.0))
        storage = _safe_float(caps.get("amf_cache_bytes", 0.0))
        warmth = _safe_float(caps.get("amf_warm_ratio", 0.0))
        total_hit += hit
        total_entries += entries
        total_storage += storage
        lines.append(_fmt("korith_amf_node_hit_rate", hit, {"node": node}))
        lines.append(_fmt("korith_amf_node_entries", entries, {"node": node}))
        lines.append(_fmt("korith_amf_node_storage_bytes", storage, {"node": node}))
        lines.append(_fmt("korith_amf_node_warmth", warmth, {"node": node}))

    avg_hit = total_hit / float(max(1, len(rows)))
    lines.append(_fmt("korith_amf_hit_rate_total", avg_hit))
    lines.append(_fmt("korith_amf_entries_total", total_entries))
    lines.append(_fmt("korith_amf_storage_bytes_total", total_storage))
    return "\n".join(lines) + ("\n" if lines else "")


def render_coordinator_metrics(stats: Dict[str, Any], lookup_ms: float = 0.0) -> str:
    if not isinstance(stats, dict):
        return ""
    lines = [
        _fmt("korith_amf_entries_total", _safe_float(stats.get("keys", 0))),
        _fmt("korith_amf_requests_total", _safe_float(stats.get("rows", 0))),
        _fmt("korith_amf_coordinator_lookup_ms", max(0.0, _safe_float(lookup_ms))),
    ]
    return "\n".join(lines) + "\n"
