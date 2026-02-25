from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def snapshot_meta_path(snapshot_path: str | Path) -> Path:
    p = Path(snapshot_path)
    return p.with_name(f"{p.name}.meta.json")


def compute_snapshot_checksum(snapshot_path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(snapshot_path)
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_snapshot_metadata(
    snapshot_path: str | Path,
    *,
    fingerprint_hash: str,
    model_hash: str,
    tokenizer_hash: str,
    backend_id: str,
    n_ctx: int,
    kv_layout_version: str,
    created_at: str,
) -> Path:
    snap = Path(snapshot_path)
    meta = {
        "format": "korith.kv_snapshot.v1",
        "fingerprint_hash": fingerprint_hash,
        "model_hash": model_hash,
        "tokenizer_hash": tokenizer_hash,
        "backend_id": backend_id,
        "n_ctx": int(n_ctx),
        "kv_layout_version": kv_layout_version,
        "size_bytes": int(snap.stat().st_size if snap.exists() else 0),
        "checksum_sha256": compute_snapshot_checksum(snap) if snap.exists() else "",
        "created_at": created_at,
    }
    path = snapshot_meta_path(snap)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_meta(path: Path) -> Dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def validate_snapshot_metadata(
    snapshot_path: str | Path,
    *,
    fingerprint_hash: str,
    model_hash: str,
    tokenizer_hash: str,
    kv_layout_version: str,
    n_ctx: int,
) -> Tuple[bool, str, Dict[str, Any] | None]:
    snap = Path(snapshot_path)
    if not snap.exists():
        return False, "snapshot_missing", None
    meta = _read_meta(snapshot_meta_path(snap))
    if meta is None:
        return False, "snapshot_meta_missing", None
    if str(meta.get("format", "")) != "korith.kv_snapshot.v1":
        return False, "snapshot_meta_format_invalid", meta
    if str(meta.get("fingerprint_hash", "")) != str(fingerprint_hash):
        return False, "snapshot_fingerprint_mismatch", meta
    if str(meta.get("model_hash", "")) != str(model_hash):
        return False, "snapshot_model_mismatch", meta
    if str(meta.get("tokenizer_hash", "")) != str(tokenizer_hash):
        return False, "snapshot_tokenizer_mismatch", meta
    if str(meta.get("kv_layout_version", "")) != str(kv_layout_version):
        return False, "snapshot_layout_mismatch", meta
    if int(meta.get("n_ctx", 0) or 0) != int(n_ctx):
        return False, "snapshot_ctx_mismatch", meta
    checksum = str(meta.get("checksum_sha256", "")).strip()
    if not checksum:
        return False, "snapshot_checksum_missing", meta
    current = compute_snapshot_checksum(snap)
    if current != checksum:
        return False, "snapshot_checksum_mismatch", meta
    return True, "ok", meta

