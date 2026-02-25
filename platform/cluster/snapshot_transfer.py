from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib import parse, request


def estimate_transfer_ms(size_bytes: int, bandwidth_mbps: float, rtt_ms: float) -> float:
    bw = max(1.0, float(bandwidth_mbps))
    transfer_ms = (float(max(0, size_bytes)) / (bw * 1024.0 * 1024.0)) * 1000.0
    return transfer_ms + max(0.0, float(rtt_ms))


def should_transfer_snapshot(
    predicted_lane: str,
    snapshot_remote: bool,
    estimated_transfer_ms: float,
    baseline_prefix_ms: float,
    threshold: float,
) -> bool:
    if str(predicted_lane).upper() != "HIT":
        return False
    if not snapshot_remote:
        return False
    baseline = max(0.0, float(baseline_prefix_ms))
    if baseline <= 0.0:
        return False
    return float(estimated_transfer_ms) < (baseline * float(threshold))


def fetch_snapshot_bytes(
    node_host: str,
    node_port: int,
    fingerprint_hash: str,
    org_id: str,
    timeout_s: float = 10.0,
) -> Tuple[bytes, Dict[str, Any]]:
    q = parse.urlencode({"fingerprint": fingerprint_hash, "org_id": org_id})
    url = f"http://{node_host}:{int(node_port)}/v1/snapshots/fetch?{q}"
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
        meta_header = resp.headers.get("X-Korith-Snapshot-Meta", "")
        meta: Dict[str, Any] = {}
        if meta_header:
            try:
                meta = json.loads(meta_header)
            except Exception:
                meta = {}
    return raw, meta


def store_snapshot_bytes(path: Path, data: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)
