from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

try:
    from .auth import get_current_customer
    from .db import get_db
    from .models import APIKey, Customer, Invoice, Metric
except ImportError:
    from auth import get_current_customer
    from db import get_db
    from models import APIKey, Customer, Invoice, Metric

router = APIRouter(prefix="/api/billing", tags=["billing"])

PRICE_CENTS_PER_MILLION = 10.0  # $0.10
TRIAL_FREE_TOKENS = 10_000_000
GPU_COST_PER_HOUR_USD = 2.50
TOKENS_PER_GPU_HOUR = 250_000.0


def _period_bounds(now: datetime) -> tuple[datetime, datetime]:
    start = datetime(now.year, now.month, 1)
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime(now.year, now.month + 1, 1)
    end = next_month - timedelta(seconds=1)
    return start, end


@router.get("/summary")
def billing_summary(
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start, end = _period_bounds(now)

    rows = (
        db.query(Metric, APIKey.tier)
        .join(APIKey, APIKey.id == Metric.api_key_id)
        .filter(
            APIKey.customer_id == customer.id,
            Metric.timestamp >= start,
            Metric.timestamp <= end,
        )
        .all()
    )

    total_tokens = 0
    trial_tokens = 0
    standard_tokens = 0
    enterprise_tokens = 0
    estimated_savings = 0.0

    for metric, tier in rows:
        t = int(metric.tokens_processed or 0)
        total_tokens += t
        if tier == "trial":
            trial_tokens += t
        elif tier == "enterprise":
            enterprise_tokens += t
        else:
            standard_tokens += t
        estimated_savings += (
            float(t) * float(metric.compute_saved_pct or 0.0) * (GPU_COST_PER_HOUR_USD / TOKENS_PER_GPU_HOUR)
        )

    trial_billable = max(0, trial_tokens - TRIAL_FREE_TOKENS)
    billable_tokens = trial_billable + standard_tokens
    amount_cents = int(round((billable_tokens / 1_000_000.0) * PRICE_CENTS_PER_MILLION))

    pricing = "$0.10 per million tokens"
    if enterprise_tokens > 0 and billable_tokens == 0:
        pricing = "Custom annual pricing"

    amount_usd = amount_cents / 100.0
    if amount_usd > 0:
        roi_x = estimated_savings / amount_usd
        roi = f"For every $1 you pay Axropus, you save ${roi_x:.2f}"
    else:
        roi = "For every $1 you pay Axropus, you save $0.00"

    return {
        "current_period": {
            "start": str(start.date()),
            "end": str(end.date()),
        },
        "total_tokens": int(total_tokens),
        "amount_cents": int(amount_cents),
        "pricing": pricing,
        "estimated_savings_usd": float(round(estimated_savings, 6)),
        "roi": roi,
    }


@router.get("/invoices")
def list_invoices(
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> list[dict]:
    rows = (
        db.query(Invoice)
        .filter(Invoice.customer_id == customer.id)
        .order_by(Invoice.created_at.desc(), Invoice.id.desc())
        .all()
    )
    out: list[dict] = []
    for row in rows:
        period_start = row.period_start.isoformat() if isinstance(row.period_start, date) else None
        period_end = row.period_end.isoformat() if isinstance(row.period_end, date) else None
        out.append(
            {
                "id": row.id,
                "period_start": period_start,
                "period_end": period_end,
                "total_tokens": int(row.total_tokens or 0),
                "amount_cents": int(row.amount_cents or 0),
                "status": row.status,
                "created_at": row.created_at,
            }
        )
    return out

