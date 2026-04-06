from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

try:
    from .auth import get_current_customer
    from .db import get_db
    from .models import APIKey, Customer, Metric
except ImportError:
    from auth import get_current_customer
    from db import get_db
    from models import APIKey, Customer, Metric

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

GPU_COST_PER_HOUR_USD = 2.50
TOKENS_PER_GPU_HOUR = 250_000.0


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@router.get("")
def dashboard(
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    base_q = (
        db.query(Metric)
        .join(APIKey, APIKey.id == Metric.api_key_id)
        .filter(APIKey.customer_id == customer.id)
    )

    totals = base_q.with_entities(
        func.coalesce(func.sum(Metric.tokens_processed), 0),
        func.coalesce(func.sum(Metric.prefix_skipped), 0),
        func.coalesce(func.sum(Metric.decode_accelerated), 0),
    ).one()
    total_tokens = int(totals[0] or 0)
    total_prefix = int(totals[1] or 0)
    total_decode = int(totals[2] or 0)

    latest = base_q.order_by(Metric.timestamp.desc(), Metric.id.desc()).first()
    if latest is None:
        return {
            "total_tokens_processed": 0,
            "total_prefix_skipped": 0,
            "total_decode_accelerated": 0,
            "current_amf_hit_rate": 0.0,
            "current_spec_acceptance_rate": 0.0,
            "current_effective_tps": 0.0,
            "current_baseline_tps": 0.0,
            "current_compute_saved_pct": 0.0,
            "estimated_monthly_savings_usd": 0.0,
            "tokens_today": 0,
            "tokens_this_week": 0,
            "daily_breakdown": [],
            "status": "inactive",
        }

    now = _utcnow_naive()
    start_today = datetime(now.year, now.month, now.day)
    weekday = start_today.weekday()
    start_week = start_today - timedelta(days=weekday)
    start_30d = start_today - timedelta(days=29)

    tokens_today = int(
        base_q.filter(Metric.timestamp >= start_today)
        .with_entities(func.coalesce(func.sum(Metric.tokens_processed), 0))
        .scalar()
        or 0
    )
    tokens_week = int(
        base_q.filter(Metric.timestamp >= start_week)
        .with_entities(func.coalesce(func.sum(Metric.tokens_processed), 0))
        .scalar()
        or 0
    )

    monthly_rows = base_q.filter(Metric.timestamp >= start_30d).all()
    estimated_monthly_savings = 0.0
    for row in monthly_rows:
        estimated_monthly_savings += (
            float(row.tokens_processed or 0)
            * float(row.compute_saved_pct or 0.0)
            * (GPU_COST_PER_HOUR_USD / TOKENS_PER_GPU_HOUR)
        )

    breakdown_rows = (
        db.query(
            func.date(Metric.timestamp).label("day"),
            func.coalesce(func.sum(Metric.tokens_processed), 0).label("tokens"),
            func.coalesce(func.avg(Metric.compute_saved_pct), 0.0).label("savings_pct"),
        )
        .join(APIKey, APIKey.id == Metric.api_key_id)
        .filter(APIKey.customer_id == customer.id, Metric.timestamp >= start_30d)
        .group_by(func.date(Metric.timestamp))
        .order_by(func.date(Metric.timestamp))
        .all()
    )
    daily_breakdown = [
        {
            "date": str(row.day),
            "tokens": int(row.tokens or 0),
            "savings_pct": float(row.savings_pct or 0.0),
        }
        for row in breakdown_rows
    ]

    is_active = latest.timestamp is not None and latest.timestamp >= (now - timedelta(minutes=10))
    return {
        "total_tokens_processed": total_tokens,
        "total_prefix_skipped": total_prefix,
        "total_decode_accelerated": total_decode,
        "current_amf_hit_rate": float(latest.amf_hit_rate or 0.0),
        "current_spec_acceptance_rate": float(latest.spec_acceptance_rate or 0.0),
        "current_effective_tps": float(latest.effective_tps or 0.0),
        "current_baseline_tps": float(latest.baseline_tps or 0.0),
        "current_compute_saved_pct": float(latest.compute_saved_pct or 0.0),
        "estimated_monthly_savings_usd": float(round(estimated_monthly_savings, 6)),
        "tokens_today": tokens_today,
        "tokens_this_week": tokens_week,
        "daily_breakdown": daily_breakdown,
        "status": "active" if is_active else "inactive",
    }

