from __future__ import annotations

import os

import jwt


def _signup_payload(email: str = "user@axropus.com") -> dict:
    return {
        "email": email,
        "password": "StrongPass123",
        "company_name": "Axropus Labs",
    }


def test_signup_creates_customer_and_api_key(client):
    res = client.post("/api/auth/signup", json=_signup_payload())
    assert res.status_code == 200
    body = res.json()

    assert isinstance(body["customer_id"], int)
    assert body["customer_id"] > 0
    assert body["api_key"].startswith("ax-")
    assert len(body["api_key"]) == 35
    assert isinstance(body["token"], str)

    keys = client.get(
        "/api/keys",
        headers={"Authorization": f"Bearer {body['token']}"},
    )
    assert keys.status_code == 200
    key_rows = keys.json()
    assert len(key_rows) == 1
    assert key_rows[0]["key"] == body["api_key"]
    assert key_rows[0]["tier"] == "trial"


def test_login_returns_valid_jwt(client):
    signup = client.post("/api/auth/signup", json=_signup_payload("login@axropus.com")).json()
    customer_id = signup["customer_id"]

    res = client.post(
        "/api/auth/login",
        json={"email": "login@axropus.com", "password": "StrongPass123"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["customer_id"] == customer_id

    claims = jwt.decode(
        body["token"],
        os.environ["AXROPUS_JWT_SECRET"],
        algorithms=["HS256"],
    )
    assert int(claims["sub"]) == customer_id
    assert "exp" in claims


def test_invalid_password_rejected(client):
    client.post("/api/auth/signup", json=_signup_payload("wrong-pass@axropus.com"))

    res = client.post(
        "/api/auth/login",
        json={"email": "wrong-pass@axropus.com", "password": "invalid"},
    )
    assert res.status_code == 401
    assert "Invalid credentials" in res.json()["detail"]


def test_duplicate_email_rejected(client):
    first = client.post("/api/auth/signup", json=_signup_payload("dupe@axropus.com"))
    assert first.status_code == 200

    second = client.post("/api/auth/signup", json=_signup_payload("dupe@axropus.com"))
    assert second.status_code == 400
    assert "Email already registered" in second.json()["detail"]

