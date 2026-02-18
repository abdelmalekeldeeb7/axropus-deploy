from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .db import get_db
from .models import APIKey, Metric

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

ALLOWED_FIELDS = {
    "api_key",
    "timestamp",
    "interval_seconds",
    "tokens_processed",
    "prefix_skipped",
    "decode_accelerated",
    "amf_hit_rate",
    "spec_acceptance_rate",
    "effective_tps",
    "baseline_tps",
    "compute_saved_pct",
    "gpu_count",
    "model_family",
    "model_size_bucket",
    "adapter_type",
    "sdk_version",
    "license_id",
    "heartbeat",
}

REQUIRED_FIELDS = {
    "api_key",
    "timestamp",
    "interval_seconds",
    "tokens_processed",
    "prefix_skipped",
    "decode_accelerated",
    "amf_hit_rate",
    "spec_acceptance_rate",
    "effective_tps",
    "baseline_tps",
    "compute_saved_pct",
    "sdk_version",
}

NUMERIC_FIELDS = {
    "interval_seconds",
    "tokens_processed",
    "prefix_skipped",
    "decode_accelerated",
    "amf_hit_rate",
    "spec_acceptance_rate",
    "effective_tps",
    "baseline_tps",
    "compute_saved_pct",
    "gpu_count",
}

SAFE_STRING_RE = re.compile(r"^[A-Za-z0-9._:\-+]{1,128}$")
IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
PATH_RE = re.compile(r"[/\\]")
MODEL_PATH_RE = re.compile(r"\.(?:gguf|safetensors|bin|pt)\b", re.IGNORECASE)


def _parse_timestamp(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("timestamp missing")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _validate_active_key(api_key: APIKey | None) -> APIKey:
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if api_key.status not in ("trial", "active"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key revoked")
    if api_key.expires_at is not None and api_key.expires_at <= datetime.now(timezone.utc).replace(tzinfo=None):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")
    return api_key


def _as_numeric(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{name} must be numeric")
    return float(value)


def _as_safe_string(name: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > 128 or SAFE_STRING_RE.fullmatch(text) is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{name} contains invalid characters")
    if PATH_RE.search(text) or IPV4_RE.search(text) or MODEL_PATH_RE.search(text):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{name} contains unsafe string content")
    return text


@router.post("")
def ingest_metrics(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict:
    unknown = set(payload.keys()) - ALLOWED_FIELDS
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Non-whitelisted fields: {', '.join(sorted(unknown))}",
        )
    missing = REQUIRED_FIELDS - set(payload.keys())
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required fields: {', '.join(sorted(missing))}",
        )

    key_value = str(payload.get("api_key", "")).strip()
    if not key_value.startswith("ax-"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid api_key format")
    api_key = db.query(APIKey).filter(APIKey.key == key_value).first()
    api_key = _validate_active_key(api_key)

    try:
        ts = _parse_timestamp(str(payload["timestamp"]))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp") from exc

    metric = Metric(
        api_key_id=api_key.id,
        timestamp=ts,
        interval_seconds=int(_as_numeric("interval_seconds", payload.get("interval_seconds"))),
        tokens_processed=int(_as_numeric("tokens_processed", payload.get("tokens_processed"))),
        prefix_skipped=int(_as_numeric("prefix_skipped", payload.get("prefix_skipped"))),
        decode_accelerated=int(_as_numeric("decode_accelerated", payload.get("decode_accelerated"))),
        amf_hit_rate=float(_as_numeric("amf_hit_rate", payload.get("amf_hit_rate"))),
        spec_acceptance_rate=float(_as_numeric("spec_acceptance_rate", payload.get("spec_acceptance_rate"))),
        effective_tps=float(_as_numeric("effective_tps", payload.get("effective_tps"))),
        baseline_tps=float(_as_numeric("baseline_tps", payload.get("baseline_tps"))),
        compute_saved_pct=float(_as_numeric("compute_saved_pct", payload.get("compute_saved_pct"))),
        gpu_count=int(_as_numeric("gpu_count", payload.get("gpu_count", 0))),
        model_family=_as_safe_string("model_family", payload.get("model_family")),
        model_size_bucket=_as_safe_string("model_size_bucket", payload.get("model_size_bucket")),
        adapter_type=_as_safe_string("adapter_type", payload.get("adapter_type")),
        sdk_version=_as_safe_string("sdk_version", payload.get("sdk_version")),
        license_id=_as_safe_string("license_id", payload.get("license_id")),
        heartbeat=1 if bool(payload.get("heartbeat", 0)) else 0,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return {"ok": True, "metric_id": metric.id}
