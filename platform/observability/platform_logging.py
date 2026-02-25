from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit_log(
    service: str,
    event: str,
    fields: Dict[str, Any],
    artifact_log: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = {
        "timestamp": utc_now(),
        "service": service,
        "event": event,
        **fields,
    }
    line = json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    if artifact_log:
        artifact_log.parent.mkdir(parents=True, exist_ok=True)
        with artifact_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return payload


def error_fields(error: Exception, replay_related: bool = False) -> Dict[str, Any]:
    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "replay_related": replay_related,
    }
