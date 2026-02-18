from __future__ import annotations

from datetime import datetime, timezone


def _signup(client, email: str = "metrics@axropus.com") -> dict:
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


def _base_metrics_payload(api_key: str) -> dict:
    return {
        "api_key": api_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interval_seconds": 60,
        "tokens_processed": 1_847_293,
        "prefix_skipped": 1_293_805,
        "decode_accelerated": 553_488,
        "amf_hit_rate": 0.98,
        "spec_acceptance_rate": 0.72,
        "effective_tps": 105.6,
        "baseline_tps": 70.9,
        "compute_saved_pct": 0.78,
        "sdk_version": "0.1.0",
    }


def test_metrics_ingestion_with_valid_payload(client):
    signup = _signup(client, "metrics-valid@axropus.com")
    payload = _base_metrics_payload(signup["api_key"])
    payload.update(
        {
            "gpu_count": 8,
            "model_family": "llama",
            "model_size_bucket": "70B",
            "adapter_type": "vllm",
            "license_id": "lic-123",
            "heartbeat": 1,
        }
    )
    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert isinstance(body["metric_id"], int)


def test_metrics_rejection_with_invalid_api_key(client):
    payload = _base_metrics_payload("ax-invalid")
    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 401


def test_metrics_rejection_with_non_whitelisted_fields(client):
    signup = _signup(client, "metrics-whitelist@axropus.com")
    payload = _base_metrics_payload(signup["api_key"])
    payload["prompt_text"] = "secret-customer-data"
    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 400
    assert "Non-whitelisted fields" in res.json()["detail"]


def test_metrics_rejection_with_string_data_in_numeric_fields(client):
    signup = _signup(client, "metrics-numeric@axropus.com")
    payload = _base_metrics_payload(signup["api_key"])
    payload["tokens_processed"] = "not-numeric"
    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 400
    assert "tokens_processed must be numeric" in res.json()["detail"]


def test_dashboard_calculation_correctness(client):
    signup = _signup(client, "metrics-dashboard@axropus.com")
    token = signup["token"]
    key = signup["api_key"]

    p1 = _base_metrics_payload(key)
    p1["tokens_processed"] = 1_000_000
    p1["prefix_skipped"] = 700_000
    p1["decode_accelerated"] = 300_000
    p1["compute_saved_pct"] = 0.75
    r1 = client.post("/api/metrics", json=p1)
    assert r1.status_code == 200

    p2 = _base_metrics_payload(key)
    p2["tokens_processed"] = 2_000_000
    p2["prefix_skipped"] = 1_200_000
    p2["decode_accelerated"] = 800_000
    p2["compute_saved_pct"] = 0.80
    p2["effective_tps"] = 120.0
    p2["baseline_tps"] = 80.0
    p2["amf_hit_rate"] = 0.99
    p2["spec_acceptance_rate"] = 0.74
    r2 = client.post("/api/metrics", json=p2)
    assert r2.status_code == 200

    dash = client.get("/api/dashboard", headers={"Authorization": f"Bearer {token}"})
    assert dash.status_code == 200
    body = dash.json()
    assert body["total_tokens_processed"] == 3_000_000
    assert body["total_prefix_skipped"] == 1_900_000
    assert body["total_decode_accelerated"] == 1_100_000
    assert body["current_effective_tps"] == 120.0
    assert body["current_baseline_tps"] == 80.0
    assert body["current_amf_hit_rate"] == 0.99
    assert body["current_spec_acceptance_rate"] == 0.74
    assert body["current_compute_saved_pct"] == 0.8
    assert body["status"] in ("active", "inactive")
    assert body["tokens_today"] >= 3_000_000
    assert body["tokens_this_week"] >= 3_000_000

