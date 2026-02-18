from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _signup(client, email: str = "keys@axropus.com") -> dict:
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


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_api_key_generation(client):
    signup = _signup(client, "gen@axropus.com")

    res = client.post(
        "/api/keys/generate",
        json={"tier": "standard"},
        headers=_auth_headers(signup["token"]),
    )
    assert res.status_code == 200
    body = res.json()

    assert body["key"].startswith("ax-")
    assert body["status"] == "active"
    assert body["tier"] == "standard"
    assert body["expires_at"] is None


def test_api_key_validation(client):
    signup = _signup(client, "validate@axropus.com")
    key = signup["api_key"]

    res = client.get(f"/api/keys/validate/{key}")
    assert res.status_code == 200
    body = res.json()
    assert body["valid"] is True
    assert body["tier"] == "trial"
    assert body["customer_id"] == signup["customer_id"]


def test_api_key_revocation(client):
    signup = _signup(client, "revoke@axropus.com")

    gen = client.post(
        "/api/keys/generate",
        json={"tier": "standard"},
        headers=_auth_headers(signup["token"]),
    ).json()

    rows = client.get("/api/keys", headers=_auth_headers(signup["token"])).json()
    key_row = next(item for item in rows if item["key"] == gen["key"])
    key_id = key_row["id"]

    revoke = client.delete(f"/api/keys/{key_id}", headers=_auth_headers(signup["token"]))
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True

    validate = client.get(f"/api/keys/validate/{gen['key']}")
    assert validate.status_code == 200
    assert validate.json()["valid"] is False


def test_trial_key_expiration(client):
    signup = _signup(client, "expire@axropus.com")
    trial_key = signup["api_key"]

    from backend.db import SessionLocal
    from backend.models import APIKey

    db = SessionLocal()
    try:
        row = db.query(APIKey).filter(APIKey.key == trial_key).first()
        assert row is not None
        row.expires_at = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        )
        db.commit()
    finally:
        db.close()

    validate = client.get(f"/api/keys/validate/{trial_key}")
    assert validate.status_code == 200
    body = validate.json()
    assert body["valid"] is False
    assert body["tier"] is None
    assert body["customer_id"] is None

