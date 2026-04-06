from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

try:
    from .auth import get_current_customer
    from .config import get_settings
    from .db import get_db
    from .models import APIKey, Customer
except ImportError:
    from auth import get_current_customer
    from config import get_settings
    from db import get_db
    from models import APIKey, Customer

router = APIRouter(prefix="/api/keys", tags=["keys"])


class KeyGenerateRequest(BaseModel):
    tier: Literal["trial", "standard", "enterprise"] = "trial"


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_key_value() -> str:
    return f"ax-{secrets.token_hex(16)}"


def _is_key_valid(api_key: APIKey) -> bool:
    if api_key.status == "revoked":
        return False
    if api_key.expires_at is not None and api_key.expires_at <= _utcnow_naive():
        return False
    return True


@router.get("")
def list_keys(
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> list[dict]:
    rows = (
        db.query(APIKey)
        .filter(APIKey.customer_id == customer.id)
        .order_by(APIKey.created_at.desc())
        .all()
    )
    return [
        {
            "id": row.id,
            "key": row.key,
            "status": row.status,
            "tier": row.tier,
            "created_at": row.created_at,
            "expires_at": row.expires_at,
        }
        for row in rows
    ]


@router.post("/generate")
def generate_key(
    payload: KeyGenerateRequest | None = None,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    settings = get_settings()
    active_count = (
        db.query(APIKey)
        .filter(APIKey.customer_id == customer.id, APIKey.status != "revoked")
        .count()
    )
    if active_count >= settings.max_keys_per_customer:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Max {settings.max_keys_per_customer} active keys reached",
        )

    req = payload or KeyGenerateRequest()
    tier = req.tier
    status_value = "trial" if tier == "trial" else "active"
    expires_at = (
        _utcnow_naive() + timedelta(days=settings.trial_key_days)
        if tier == "trial"
        else None
    )

    for _ in range(5):
        candidate = _new_key_value()
        exists = db.query(APIKey).filter(APIKey.key == candidate).first()
        if exists is None:
            row = APIKey(
                customer_id=customer.id,
                key=candidate,
                status=status_value,
                tier=tier,
                expires_at=expires_at,
            )
            db.add(row)
            db.commit()
            return {
                "key": row.key,
                "status": row.status,
                "tier": row.tier,
                "expires_at": row.expires_at,
            }
    raise HTTPException(status_code=500, detail="Failed to generate unique API key")


@router.delete("/{key_id}")
def revoke_key(
    key_id: int,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    row = (
        db.query(APIKey)
        .filter(APIKey.id == key_id, APIKey.customer_id == customer.id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    row.status = "revoked"
    if row.expires_at is None or row.expires_at > _utcnow_naive():
        row.expires_at = _utcnow_naive()
    db.commit()
    return {"revoked": True}


@router.get("/validate/{key_value}")
def validate_key(
    key_value: str,
    db: Session = Depends(get_db),
) -> dict:
    row = db.query(APIKey).filter(APIKey.key == key_value).first()
    if row is None:
        return {"valid": False, "tier": None, "customer_id": None}
    if not _is_key_valid(row):
        return {"valid": False, "tier": None, "customer_id": None}
    return {
        "valid": True,
        "tier": row.tier,
        "customer_id": row.customer_id,
    }

