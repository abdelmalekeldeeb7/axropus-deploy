"""Savings calculation and cost comparison engine.

Computes real savings from Axropus AMF acceleration versus commercial API
providers (OpenAI, Anthropic, Together AI), based on actual usage metrics
including token savings, AMF hit rates, and prefix reuse efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from .auth import get_current_customer
from .db import get_db
from .models import APIKey, Customer, Metric

logger = logging.getLogger(__name__)

router = APIRouter(tags=["economics"])


# ---------------------------------------------------------------------------
# Provider pricing (per 1M tokens)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderPricing:
    """Pricing structure for a single provider."""

    name: str
    input_per_1m: float   # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens


PROVIDER_PRICING: dict[str, ProviderPricing] = {
    "openai_gpt4o": ProviderPricing(
        name="OpenAI GPT-4o",
        input_per_1m=2.50,
        output_per_1m=10.00,
    ),
    "anthropic_claude35": ProviderPricing(
        name="Claude 3.5 Sonnet",
        input_per_1m=3.00,
        output_per_1m=15.00,
    ),
    "together_ai": ProviderPricing(
        name="Together AI",
        input_per_1m=0.90,
        output_per_1m=0.90,
    ),
    "axropus": ProviderPricing(
        name="Axropus (self-hosted + AMF)",
        input_per_1m=0.30,
        output_per_1m=0.60,
    ),
}

# Default input/output token ratio (70/30 split is typical for chat workloads)
DEFAULT_INPUT_RATIO = 0.70
DEFAULT_OUTPUT_RATIO = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_period(period: str) -> tuple[datetime, datetime]:
    """Parse a period string into (start, end) datetimes.

    Supported formats:
      - "7d"  — last 7 days
      - "30d" — last 30 days
      - "90d" — last 90 days
      - "mtd" — month to date
      - "ytd" — year to date
    """
    now = _utcnow_naive()

    if period == "mtd":
        start = datetime(now.year, now.month, 1)
        return start, now

    if period == "ytd":
        start = datetime(now.year, 1, 1)
        return start, now

    if period.endswith("d"):
        try:
            days = int(period[:-1])
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid period format: {period!r}. Use '7d', '30d', '90d', 'mtd', or 'ytd'.",
            )
        if days < 1 or days > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Period must be between 1d and 365d",
            )
        return now - timedelta(days=days), now

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid period: {period!r}. Use '7d', '30d', '90d', 'mtd', or 'ytd'.",
    )


def _fetch_usage_metrics(
    db: Session,
    customer: Customer,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Aggregate usage metrics for a customer over a time range.

    Returns dict with total_tokens, tokens_saved (from prefix skip + decode
    acceleration), average hit_rate, average prefix_reuse, and compute_saved.
    """
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

    if not rows:
        return {
            "total_tokens": 0,
            "tokens_saved": 0,
            "avg_hit_rate": 0.0,
            "avg_prefix_reuse": 0.0,
            "avg_compute_saved_pct": 0.0,
            "total_prefix_skipped": 0,
            "total_decode_accelerated": 0,
            "sample_count": 0,
        }

    total_tokens = 0
    total_prefix_skipped = 0
    total_decode_accelerated = 0
    hit_rates: list[float] = []
    compute_saved: list[float] = []

    for metric, _tier in rows:
        total_tokens += int(metric.tokens_processed or 0)
        total_prefix_skipped += int(metric.prefix_skipped or 0)
        total_decode_accelerated += int(metric.decode_accelerated or 0)
        hit_rates.append(float(metric.amf_hit_rate or 0.0))
        compute_saved.append(float(metric.compute_saved_pct or 0.0))

    tokens_saved = total_prefix_skipped + total_decode_accelerated
    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    avg_compute_saved = sum(compute_saved) / len(compute_saved) if compute_saved else 0.0
    prefix_reuse = (total_prefix_skipped / total_tokens) if total_tokens > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "tokens_saved": tokens_saved,
        "avg_hit_rate": round(avg_hit_rate, 4),
        "avg_prefix_reuse": round(prefix_reuse, 4),
        "avg_compute_saved_pct": round(avg_compute_saved, 4),
        "total_prefix_skipped": total_prefix_skipped,
        "total_decode_accelerated": total_decode_accelerated,
        "sample_count": len(rows),
    }


def _calculate_provider_cost(
    total_tokens: int,
    pricing: ProviderPricing,
    input_ratio: float = DEFAULT_INPUT_RATIO,
    output_ratio: float = DEFAULT_OUTPUT_RATIO,
) -> float:
    """Calculate cost for a provider given total token usage."""
    input_tokens = total_tokens * input_ratio
    output_tokens = total_tokens * output_ratio
    cost = (input_tokens / 1_000_000) * pricing.input_per_1m + \
           (output_tokens / 1_000_000) * pricing.output_per_1m
    return round(cost, 6)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/v1/economics/savings")
def get_savings(
    period: str = Query(default="30d", description="Time period: 7d, 30d, 90d, mtd, ytd"),
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    """Calculate savings for the specified period based on actual AMF metrics.

    Returns token-level savings, dollar savings versus cloud providers,
    and AMF efficiency metrics (hit rate, prefix reuse, compute saved).
    """
    start, end = _parse_period(period)
    usage = _fetch_usage_metrics(db, customer, start, end)
    total = usage["total_tokens"]

    # Axropus cost (actual)
    axropus_pricing = PROVIDER_PRICING["axropus"]
    axropus_cost = _calculate_provider_cost(total, axropus_pricing)

    # What it would have cost on other providers
    openai_cost = _calculate_provider_cost(total, PROVIDER_PRICING["openai_gpt4o"])
    anthropic_cost = _calculate_provider_cost(total, PROVIDER_PRICING["anthropic_claude35"])
    together_cost = _calculate_provider_cost(total, PROVIDER_PRICING["together_ai"])

    # Best-case comparison (highest-priced provider)
    max_alternative_cost = max(openai_cost, anthropic_cost, together_cost)
    dollar_savings = round(max_alternative_cost - axropus_cost, 6)
    savings_pct = round((dollar_savings / max_alternative_cost * 100), 2) if max_alternative_cost > 0 else 0.0

    return {
        "period": period,
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "total_tokens": total,
        "tokens_saved_by_amf": usage["tokens_saved"],
        "amf_metrics": {
            "hit_rate": usage["avg_hit_rate"],
            "prefix_reuse": usage["avg_prefix_reuse"],
            "compute_saved_pct": usage["avg_compute_saved_pct"],
            "prefix_skipped": usage["total_prefix_skipped"],
            "decode_accelerated": usage["total_decode_accelerated"],
        },
        "cost_usd": {
            "axropus": axropus_cost,
            "openai_gpt4o": openai_cost,
            "anthropic_claude35": anthropic_cost,
            "together_ai": together_cost,
        },
        "savings_vs_best_alternative_usd": dollar_savings,
        "savings_pct": savings_pct,
    }


@router.get("/v1/economics/comparison")
def get_comparison(
    period: str = Query(default="30d", description="Time period: 7d, 30d, 90d, mtd, ytd"),
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    """Side-by-side cost comparison against OpenAI, Anthropic, and Together AI.

    Shows what the customer's actual usage would cost on each provider,
    with per-provider savings and ROI calculations.
    """
    start, end = _parse_period(period)
    usage = _fetch_usage_metrics(db, customer, start, end)
    total = usage["total_tokens"]

    axropus_cost = _calculate_provider_cost(total, PROVIDER_PRICING["axropus"])

    comparisons: list[dict[str, Any]] = []
    for key, pricing in PROVIDER_PRICING.items():
        if key == "axropus":
            continue
        provider_cost = _calculate_provider_cost(total, pricing)
        savings = round(provider_cost - axropus_cost, 6)
        savings_pct = round((savings / provider_cost * 100), 2) if provider_cost > 0 else 0.0
        roi_x = round(savings / axropus_cost, 2) if axropus_cost > 0 else 0.0

        comparisons.append({
            "provider": pricing.name,
            "provider_key": key,
            "input_per_1m": pricing.input_per_1m,
            "output_per_1m": pricing.output_per_1m,
            "estimated_cost_usd": provider_cost,
            "axropus_cost_usd": axropus_cost,
            "savings_usd": savings,
            "savings_pct": savings_pct,
            "roi_x": roi_x,
        })

    # Sort by savings descending
    comparisons.sort(key=lambda c: c["savings_usd"], reverse=True)

    return {
        "period": period,
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "total_tokens": total,
        "axropus_cost_usd": axropus_cost,
        "axropus_pricing": {
            "input_per_1m": PROVIDER_PRICING["axropus"].input_per_1m,
            "output_per_1m": PROVIDER_PRICING["axropus"].output_per_1m,
        },
        "comparisons": comparisons,
    }


@router.get("/v1/economics/forecast")
def get_forecast(
    months: int = Query(default=12, ge=1, le=36, description="Forecast horizon in months"),
    growth_rate: float = Query(default=0.10, ge=0.0, le=5.0, description="Monthly token growth rate (0.10 = 10%)"),
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    """Project future savings based on current usage trends and growth assumptions.

    Uses the last 30 days of actual usage as the baseline, then extrapolates
    forward applying the specified monthly growth rate.
    """
    now = _utcnow_naive()
    start = now - timedelta(days=30)
    usage = _fetch_usage_metrics(db, customer, start, now)

    # Monthly baseline from last 30 days
    baseline_tokens = usage["total_tokens"]
    baseline_axropus = _calculate_provider_cost(baseline_tokens, PROVIDER_PRICING["axropus"])
    baseline_openai = _calculate_provider_cost(baseline_tokens, PROVIDER_PRICING["openai_gpt4o"])
    baseline_anthropic = _calculate_provider_cost(baseline_tokens, PROVIDER_PRICING["anthropic_claude35"])
    baseline_together = _calculate_provider_cost(baseline_tokens, PROVIDER_PRICING["together_ai"])

    monthly_projections: list[dict[str, Any]] = []
    cumulative_savings: dict[str, float] = {
        "vs_openai": 0.0,
        "vs_anthropic": 0.0,
        "vs_together": 0.0,
    }
    cumulative_tokens = 0

    for month_offset in range(1, months + 1):
        growth_multiplier = (1.0 + growth_rate) ** month_offset
        projected_tokens = int(baseline_tokens * growth_multiplier)
        cumulative_tokens += projected_tokens

        axropus = _calculate_provider_cost(projected_tokens, PROVIDER_PRICING["axropus"])
        openai = _calculate_provider_cost(projected_tokens, PROVIDER_PRICING["openai_gpt4o"])
        anthropic = _calculate_provider_cost(projected_tokens, PROVIDER_PRICING["anthropic_claude35"])
        together = _calculate_provider_cost(projected_tokens, PROVIDER_PRICING["together_ai"])

        month_savings_openai = round(openai - axropus, 2)
        month_savings_anthropic = round(anthropic - axropus, 2)
        month_savings_together = round(together - axropus, 2)

        cumulative_savings["vs_openai"] = round(cumulative_savings["vs_openai"] + month_savings_openai, 2)
        cumulative_savings["vs_anthropic"] = round(cumulative_savings["vs_anthropic"] + month_savings_anthropic, 2)
        cumulative_savings["vs_together"] = round(cumulative_savings["vs_together"] + month_savings_together, 2)

        target_date = now + timedelta(days=30 * month_offset)
        monthly_projections.append({
            "month": month_offset,
            "date": str(target_date.date()),
            "projected_tokens": projected_tokens,
            "cost_axropus": axropus,
            "cost_openai": openai,
            "cost_anthropic": anthropic,
            "cost_together": together,
            "monthly_savings_vs_openai": month_savings_openai,
            "monthly_savings_vs_anthropic": month_savings_anthropic,
            "monthly_savings_vs_together": month_savings_together,
            "cumulative_savings_vs_openai": cumulative_savings["vs_openai"],
            "cumulative_savings_vs_anthropic": cumulative_savings["vs_anthropic"],
            "cumulative_savings_vs_together": cumulative_savings["vs_together"],
        })

    # Summary
    total_axropus = sum(p["cost_axropus"] for p in monthly_projections)
    total_openai = sum(p["cost_openai"] for p in monthly_projections)
    total_anthropic = sum(p["cost_anthropic"] for p in monthly_projections)

    return {
        "forecast_months": months,
        "growth_rate_pct": round(growth_rate * 100, 1),
        "baseline_monthly_tokens": baseline_tokens,
        "baseline_amf_metrics": {
            "hit_rate": usage["avg_hit_rate"],
            "prefix_reuse": usage["avg_prefix_reuse"],
            "compute_saved_pct": usage["avg_compute_saved_pct"],
        },
        "total_projected_tokens": cumulative_tokens,
        "total_projected_cost_axropus": round(total_axropus, 2),
        "total_savings_vs_openai": cumulative_savings["vs_openai"],
        "total_savings_vs_anthropic": cumulative_savings["vs_anthropic"],
        "total_savings_vs_together": cumulative_savings["vs_together"],
        "monthly_projections": monthly_projections,
    }
