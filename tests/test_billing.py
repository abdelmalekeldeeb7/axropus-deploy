from __future__ import annotations

from datetime import datetime, timezone


def _signup(client, email: str) -> dict:
    res = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": "StrongPass123",
            "company_name": "Axropus",
        },
    )
    assert res.status_code == 200
    return res.json()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _post_metric(client, api_key: str, tokens: int, saved_pct: float) -> None:
    payload = {
        "api_key": api_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interval_seconds": 60,
        "tokens_processed": tokens,
        "prefix_skipped": int(tokens * 0.6),
        "decode_accelerated": int(tokens * 0.4),
        "amf_hit_rate": 0.98,
        "spec_acceptance_rate": 0.72,
        "effective_tps": 100.0,
        "baseline_tps": 70.0,
        "compute_saved_pct": saved_pct,
        "sdk_version": "0.1.0",
    }
    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 200


def test_billing_amount_calculation(client):
    signup = _signup(client, "billing-amount@axropus.com")
    headers = _auth(signup["token"])

    gen = client.post("/api/keys/generate", headers=headers, json={"tier": "standard"})
    assert gen.status_code == 200
    standard_key = gen.json()["key"]

    _post_metric(client, standard_key, tokens=2_000_000, saved_pct=0.5)

    summary = client.get("/api/billing/summary", headers=headers)
    assert summary.status_code == 200
    body = summary.json()
    assert body["total_tokens"] == 2_000_000
    assert body["amount_cents"] == 20
    assert body["pricing"] == "$0.10 per million tokens"


def test_trial_tier_free_tokens(client):
    signup = _signup(client, "billing-trial@axropus.com")
    headers = _auth(signup["token"])
    trial_key = signup["api_key"]

    _post_metric(client, trial_key, tokens=11_000_000, saved_pct=0.7)

    summary = client.get("/api/billing/summary", headers=headers)
    assert summary.status_code == 200
    body = summary.json()
    assert body["total_tokens"] == 11_000_000
    # First 10M free => 1M billable => 10 cents
    assert body["amount_cents"] == 10


def test_roi_calculation(client):
    signup = _signup(client, "billing-roi@axropus.com")
    headers = _auth(signup["token"])
    gen = client.post("/api/keys/generate", headers=headers, json={"tier": "standard"})
    standard_key = gen.json()["key"]

    _post_metric(client, standard_key, tokens=2_000_000, saved_pct=0.5)
    summary = client.get("/api/billing/summary", headers=headers)
    body = summary.json()
    # savings = 2,000,000 * 0.5 * (2.5 / 250000) = 10 USD
    # amount = 20 cents = 0.20 USD => ROI = 50x
    assert body["estimated_savings_usd"] == 10.0
    assert "$50.00" in body["roi"]

