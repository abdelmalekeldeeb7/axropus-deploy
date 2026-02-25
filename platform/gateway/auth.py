from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..ledger.store import LedgerStore


class AuthError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class AuthContext:
    key_id: str
    org_id: str
    permissions: Dict[str, Any]
    rate_limit_tpm: int
    rate_limit_rpm: int


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_api_key_salt_configured() -> str:
    salt = os.environ.get("KORITH_API_KEY_SALT", "").strip()
    if not salt or salt == "korith_default_salt":
        raise RuntimeError("KORITH_API_KEY_SALT must be explicitly configured")
    return salt


def hash_api_key(raw_key: str) -> str:
    salt = require_api_key_salt_configured()
    return hashlib.sha256((salt + raw_key).encode("utf-8")).hexdigest()


def issue_api_key(key_id: str) -> str:
    secret = hashlib.sha256(f"{key_id}:{utc_now()}".encode("utf-8")).hexdigest()[:40]
    return f"kth_{key_id}_{secret}"


def parse_bearer(auth_header: Optional[str]) -> str:
    if not auth_header:
        raise AuthError(401, "missing authorization header")
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise AuthError(401, "invalid authorization header")
    return parts[1].strip()


def authenticate(headers: Dict[str, str], ledger: LedgerStore) -> AuthContext:
    try:
        raw_key = parse_bearer(headers.get("Authorization") or headers.get("authorization"))
        key_hash = hash_api_key(raw_key)
        record = ledger.get_api_key_by_hash(key_hash)
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(503, f"auth backend unavailable: {exc}") from exc

    if not record:
        raise AuthError(401, "invalid api key")
    if record.get("revoked_at"):
        raise AuthError(403, "api key revoked")

    permissions_raw = record.get("permissions_json") or "{}"
    if isinstance(permissions_raw, str):
        try:
            permissions = json.loads(permissions_raw)
        except Exception:
            permissions = {}
    else:
        permissions = permissions_raw if isinstance(permissions_raw, dict) else {}

    return AuthContext(
        key_id=record["key_id"],
        org_id=record["org_id"],
        permissions=permissions,
        rate_limit_tpm=int(record.get("rate_limit_tpm", 120000) or 120000),
        rate_limit_rpm=int(record.get("rate_limit_rpm", 600) or 600),
    )
