from __future__ import annotations

import time
from datetime import datetime, timezone

from starlette.websockets import WebSocketDisconnect


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
    def __init__(self, command_map: dict[str, tuple[int, str, str]]) -> None:
        self._command_map = command_map

    def set_missing_host_key_policy(self, _policy) -> None:
        return None

    def connect(self, **_kwargs) -> None:
        return None

    def exec_command(self, command: str, timeout: int = 240):
        _ = timeout
        for needle, result in self._command_map.items():
            if needle in command:
                code, out, err = result
                return (None, _FakeStream(out, code), _FakeStream(err, code))
        return (None, _FakeStream("", 0), _FakeStream("", 0))

    def close(self) -> None:
        return None


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _metric_payload(api_key: str, tokens: int, saved_pct: float) -> dict:
    return {
        "api_key": api_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interval_seconds": 60,
        "tokens_processed": tokens,
        "prefix_skipped": int(tokens * 0.7),
        "decode_accelerated": int(tokens * 0.3),
        "amf_hit_rate": 0.99,
        "spec_acceptance_rate": 0.72,
        "effective_tps": 105.6,
        "baseline_tps": 70.9,
        "compute_saved_pct": saved_pct,
        "sdk_version": "0.1.0",
        "adapter_type": "vllm",
        "model_family": "llama",
        "model_size_bucket": "70B",
        "heartbeat": 1,
    }


def test_full_platform_flow(client, monkeypatch) -> None:
    # a) Sign up and get API key
    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "e2e@axropus.com",
            "password": "StrongPass123",
            "company_name": "Axropus E2E",
        },
    )
    assert signup.status_code == 200
    signup_body = signup.json()
    token = signup_body["token"]

    keys = client.get("/api/keys", headers=_auth(token))
    assert keys.status_code == 200
    trial_key_row = keys.json()[0]

    # Create a standard key for billable usage later.
    standard = client.post(
        "/api/keys/generate",
        headers=_auth(token),
        json={"tier": "standard"},
    )
    assert standard.status_code == 200
    standard_key = standard.json()["key"]

    # b) Configure deployment with mocked SSH runtime
    command_map = {
        "--version": (0, "Python 3.12.3\n", ""),
        "-m pip install --upgrade axropus": (0, "Successfully installed axropus-0.1.0\n", ""),
        "ps aux | grep -E 'vllm|sglang|triton'": (0, "python -m vllm.entrypoints.openai.api_server\n", ""),
        "import vllm; print(vllm.__version__)": (0, "0.6.2\n", ""),
        "ps aux | grep -E -- '--model|model='": (0, "--model meta-llama/Llama-3.1-70B\n", ""),
        "huggingface-cli download": (0, "downloaded\n", ""),
        "cat > /tmp/axropus_install.sh": (0, "", ""),
        "bash /tmp/axropus_install.sh": (0, "installed\n", ""),
        "test -f ~/.axropus/config.json": (0, "active\n", ""),
    }
    monkeypatch.setattr("backend.deploy.paramiko.SSHClient", lambda: _MockSSHClient(command_map))
    monkeypatch.setattr("backend.deploy.paramiko.AutoAddPolicy", lambda: object())

    deploy = client.post(
        "/api/deploy",
        headers=_auth(token),
        json={
            "host": "10.0.1.50",
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": "secret-pass",
            "api_key_id": trial_key_row["id"],
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "70B",
        },
    )
    assert deploy.status_code == 200
    dep_id = deploy.json()["deployment_id"]

    # c) Deploy and watch progress complete via websocket stream
    streamed_messages: list[dict] = []
    with client.websocket_connect(f"/api/deploy/stream/{dep_id}") as ws:
        deadline = time.time() + 6.0
        while time.time() < deadline:
            try:
                msg = ws.receive_json()
            except WebSocketDisconnect:
                break
            streamed_messages.append(msg)
            if "DEPLOYMENT COMPLETE" in str(msg.get("message", "")):
                break

    assert any("DEPLOYMENT COMPLETE" in str(m.get("message", "")) for m in streamed_messages)

    status = client.get(f"/api/deploy/status/{dep_id}", headers=_auth(token))
    assert status.status_code == 200
    assert status.json()["status"] == "active"

    # d) SDK-equivalent metrics payload sent to backend
    metrics = client.post("/api/metrics", json=_metric_payload(standard_key, tokens=2_000_000, saved_pct=0.78))
    assert metrics.status_code == 200

    # e) Dashboard returns expected savings shape
    dashboard = client.get("/api/dashboard", headers=_auth(token))
    assert dashboard.status_code == 200
    dash = dashboard.json()
    assert dash["total_tokens_processed"] == 2_000_000
    assert dash["current_compute_saved_pct"] == 0.78
    assert dash["estimated_monthly_savings_usd"] > 0
    assert dash["status"] in ("active", "inactive")

    # f) Billing returns expected amount for standard key usage
    billing = client.get("/api/billing/summary", headers=_auth(token))
    assert billing.status_code == 200
    summary = billing.json()
    assert summary["total_tokens"] == 2_000_000
    assert summary["amount_cents"] == 20
    assert "save" in summary["roi"].lower()

    # g) Teardown/cleanup: revoke key and verify invalidation
    key_rows = client.get("/api/keys", headers=_auth(token)).json()
    standard_row = next(row for row in key_rows if row["key"] == standard_key)
    revoke = client.delete(f"/api/keys/{standard_row['id']}", headers=_auth(token))
    assert revoke.status_code == 200

    validate = client.get(f"/api/keys/validate/{standard_key}")
    assert validate.status_code == 200
    assert validate.json()["valid"] is False
