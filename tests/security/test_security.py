from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt


class _FakeChannel:
    def __init__(self, code: int) -> None:
        self._code = int(code)

    def recv_exit_status(self) -> int:
        return self._code


class _FakeStream:
    def __init__(self, content: str, code: int) -> None:
        self._content = content.encode("utf-8")
        self.channel = _FakeChannel(code)

    def read(self) -> bytes:
        return self._content


class _MockSSHClient:
    def __init__(self, command_map: dict[str, tuple[int, str, str]], connect_error: Exception | None = None) -> None:
        self._command_map = command_map
        self._connect_error = connect_error

    def set_missing_host_key_policy(self, _policy) -> None:
        return None

    def connect(self, **_kwargs) -> None:
        if self._connect_error is not None:
            raise self._connect_error

    def exec_command(self, command: str, timeout: int = 240):
        _ = timeout
        for needle, result in self._command_map.items():
            if needle in command:
                code, out, err = result
                return (None, _FakeStream(out, code), _FakeStream(err, code))
        return (None, _FakeStream("", 0), _FakeStream("", 0))

    def close(self) -> None:
        return None


def _signup(client, email: str) -> dict:
    res = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": "StrongPass123",
            "company_name": "Axropus Security",
        },
    )
    assert res.status_code == 200
    return res.json()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _metric_payload(api_key: str) -> dict:
    return {
        "api_key": api_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interval_seconds": 60,
        "tokens_processed": 1000,
        "prefix_skipped": 800,
        "decode_accelerated": 200,
        "amf_hit_rate": 0.99,
        "spec_acceptance_rate": 0.7,
        "effective_tps": 100.0,
        "baseline_tps": 70.0,
        "compute_saved_pct": 0.3,
        "sdk_version": "0.1.0",
    }


def _dump_db_strings() -> str:
    from backend.db import SessionLocal
    from backend.models import APIKey, Customer, Deployment, Invoice, Metric

    db = SessionLocal()
    try:
        blob: list[str] = []
        for model in (Customer, APIKey, Deployment, Metric, Invoice):
            for row in db.query(model).all():
                blob.append(str(row.__dict__))
        return "\n".join(blob)
    finally:
        db.close()


def test_ssh_credentials_not_in_database_after_deploy(client, monkeypatch):
    signup = _signup(client, "sec-db@axropus.com")
    token = signup["token"]

    secret_password = "UltraSecretPass!"
    secret_key = "-----BEGIN PRIVATE KEY-----DO-NOT-STORE-----END PRIVATE KEY-----"

    monkeypatch.setattr(
        "backend.deploy.paramiko.SSHClient",
        lambda: _MockSSHClient({}, connect_error=RuntimeError("auth error")),
    )
    monkeypatch.setattr("backend.deploy.paramiko.AutoAddPolicy", lambda: object())

    deploy = client.post(
        "/api/deploy",
        headers=_auth(token),
        json={
            "host": "10.0.1.60",
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": secret_password,
            "ssh_key": secret_key,
            "api_key_id": 1,
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "70B",
        },
    )
    assert deploy.status_code == 200

    import time

    dep_id = deploy.json()["deployment_id"]
    deadline = time.time() + 2.0
    while time.time() < deadline:
        status = client.get(f"/api/deploy/status/{dep_id}", headers=_auth(token))
        if status.json().get("status") in ("active", "failed"):
            break
        time.sleep(0.02)

    db_blob = _dump_db_strings()
    assert secret_password not in db_blob
    assert secret_key not in db_blob


def test_ssh_credentials_not_in_log_file_or_stream(client, monkeypatch):
    signup = _signup(client, "sec-logs@axropus.com")
    token = signup["token"]

    secret_password = "DontLeakMe"
    secret_key = "-----BEGIN PRIVATE KEY-----LEAKCHECK-----END PRIVATE KEY-----"

    monkeypatch.setattr(
        "backend.deploy.paramiko.SSHClient",
        lambda: _MockSSHClient({}, connect_error=RuntimeError("connection denied")),
    )
    monkeypatch.setattr("backend.deploy.paramiko.AutoAddPolicy", lambda: object())

    deploy = client.post(
        "/api/deploy",
        headers=_auth(token),
        json={
            "host": "10.0.1.61",
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": secret_password,
            "ssh_key": secret_key,
            "api_key_id": 1,
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "70B",
        },
    )
    dep_id = deploy.json()["deployment_id"]

    import time

    deadline = time.time() + 2.0
    last = {}
    while time.time() < deadline:
        res = client.get(f"/api/deploy/status/{dep_id}", headers=_auth(token))
        last = res.json()
        if last.get("status") in ("active", "failed"):
            break
        time.sleep(0.02)

    messages = "\n".join(item["message"] for item in last.get("logs", []))
    assert secret_password not in messages
    assert secret_key not in messages


def test_metrics_rejects_non_whitelisted_fields(client):
    signup = _signup(client, "sec-whitelist@axropus.com")
    payload = _metric_payload(signup["api_key"])
    payload["prompt_text"] = "customer-private-prompt"

    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 400
    assert "Non-whitelisted fields" in res.json()["detail"]


def test_metrics_rejects_strings_in_numeric_fields(client):
    signup = _signup(client, "sec-numeric@axropus.com")
    payload = _metric_payload(signup["api_key"])
    payload["effective_tps"] = "fast"

    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 400
    assert "must be numeric" in res.json()["detail"]


def test_jwt_tokens_expire_correctly(client):
    signup = _signup(client, "sec-jwt@axropus.com")
    expired = jwt.encode(
        {
            "sub": str(signup["customer_id"]),
            "exp": int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()),
        },
        "test-secret",
        algorithm="HS256",
    )

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert me.status_code == 401


def test_revoked_api_keys_are_rejected(client):
    signup = _signup(client, "sec-revoke@axropus.com")
    token = signup["token"]

    keys = client.get("/api/keys", headers=_auth(token)).json()
    trial_id = keys[0]["id"]
    revoke = client.delete(f"/api/keys/{trial_id}", headers=_auth(token))
    assert revoke.status_code == 200

    payload = _metric_payload(signup["api_key"])
    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 401


def test_rate_limiting_on_key_generation(client):
    signup = _signup(client, "sec-rate@axropus.com")
    token = signup["token"]

    # Max active keys is 5; signup already created 1 trial key.
    for _ in range(4):
        ok = client.post("/api/keys/generate", headers=_auth(token), json={"tier": "standard"})
        assert ok.status_code == 200

    blocked = client.post("/api/keys/generate", headers=_auth(token), json={"tier": "standard"})
    assert blocked.status_code == 429


def test_login_rate_limiting(client):
    last_status = None
    for _ in range(25):
        res = client.post(
            "/api/auth/login",
            json={"email": "sec-login-rate@axropus.com", "password": "wrong-pass"},
        )
        last_status = res.status_code
        if res.status_code == 429:
            break

    assert last_status == 429


def test_health_live_and_ready(client):
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "live"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_zero_data_no_customer_prompts_or_model_paths_stored(client):
    signup = _signup(client, "sec-zerodata@axropus.com")

    prompt = "PRIVATE_PROMPT_NEVER_STORE"
    payload = _metric_payload(signup["api_key"])
    payload["prompt"] = prompt
    res = client.post("/api/metrics", json=payload)
    assert res.status_code == 400

    db_blob = _dump_db_strings()
    assert prompt not in db_blob
    assert ".gguf" not in db_blob.lower()
    assert "/models/" not in db_blob.lower()


def test_zero_data_metrics_reject_file_paths_and_ip_addresses(client):
    signup = _signup(client, "sec-paths@axropus.com")

    bad_path = _metric_payload(signup["api_key"])
    bad_path["model_size_bucket"] = "/tmp/model.gguf"
    path_res = client.post("/api/metrics", json=bad_path)
    assert path_res.status_code == 400

    bad_ip = _metric_payload(signup["api_key"])
    bad_ip["license_id"] = "10.1.2.3"
    ip_res = client.post("/api/metrics", json=bad_ip)
    assert ip_res.status_code == 400


def test_zero_data_no_plaintext_ip_stored_after_deploy(client, monkeypatch):
    signup = _signup(client, "sec-ip@axropus.com")
    token = signup["token"]
    host_ip = "10.0.9.99"

    monkeypatch.setattr(
        "backend.deploy.paramiko.SSHClient",
        lambda: _MockSSHClient({}, connect_error=RuntimeError("fail")),
    )
    monkeypatch.setattr("backend.deploy.paramiko.AutoAddPolicy", lambda: object())

    dep = client.post(
        "/api/deploy",
        headers=_auth(token),
        json={
            "host": host_ip,
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": "x",
            "api_key_id": 1,
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "70B",
        },
    )
    assert dep.status_code == 200

    import time

    dep_id = dep.json()["deployment_id"]
    deadline = time.time() + 2.0
    while time.time() < deadline:
        status = client.get(f"/api/deploy/status/{dep_id}", headers=_auth(token)).json()
        if status.get("status") in ("active", "failed"):
            break
        time.sleep(0.02)

    db_blob = _dump_db_strings()
    assert host_ip not in db_blob
    assert re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", db_blob) is None


def test_audit_log_integrity_verification(tmp_path: Path):
    sdk_root = Path("/home/korith/axropus-sdk")
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))

    from axropus.zero_data.audit_log import AuditLog

    audit = AuditLog(log_dir=str(tmp_path / "audit"))
    audit.log("sdk_init", {"mode": "standard"})
    audit.log("engine_wrapped", {"runtime": "vllm"})
    assert audit.verify_integrity() is True

    log_file = tmp_path / "audit" / "audit.jsonl"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[-1])
    tampered["details"] = {"runtime": "tampered"}
    lines[-1] = json.dumps(tampered)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert audit.verify_integrity() is False
