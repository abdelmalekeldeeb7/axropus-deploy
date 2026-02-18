from __future__ import annotations

import time


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
        self.closed = False

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
                return (
                    None,
                    _FakeStream(out, code),
                    _FakeStream(err, code),
                )
        return (None, _FakeStream("", 0), _FakeStream("", 0))

    def close(self) -> None:
        self.closed = True


def _signup_and_headers(client, email: str) -> tuple[dict, dict]:
    signup = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": "StrongPass123",
            "company_name": "Axropus",
        },
    )
    assert signup.status_code == 200
    body = signup.json()
    return body, {"Authorization": f"Bearer {body['token']}"}


def _wait_for_deploy_done(client, deployment_id: int, headers: dict, timeout_s: float = 8.0) -> dict:
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        res = client.get(f"/api/deploy/status/{deployment_id}", headers=headers)
        assert res.status_code == 200
        last = res.json()
        if str(last.get("status")) in ("active", "failed"):
            return last
        time.sleep(0.05)
    return last


def _patch_paramiko(monkeypatch, mock_client):
    monkeypatch.setattr("backend.deploy.paramiko.SSHClient", lambda: mock_client)
    monkeypatch.setattr("backend.deploy.paramiko.AutoAddPolicy", lambda: object())


def test_successful_deployment_flow(client, monkeypatch):
    signup, headers = _signup_and_headers(client, "deploy-success@axropus.com")
    _ = signup
    commands = {
        "--version": (0, "Python 3.12.3\n", ""),
        "-m pip install --upgrade axropus": (0, "Successfully installed axropus-0.1.0\n", ""),
        "ps aux | grep -E 'vllm|sglang|triton'": (0, "python -m vllm.entrypoints.openai.api_server\n", ""),
        "import vllm; print(vllm.__version__)": (0, "0.6.2\n", ""),
        "ps aux | grep -E -- '--model|model='": (0, "--model meta-llama/Llama-3.1-70B\n", ""),
        "huggingface-cli download": (0, "downloaded\n", ""),
        "cat > /tmp/axropus_install.sh": (0, "", ""),
        "bash /tmp/axropus_install.sh": (0, "[AXROPUS] installer finished\n", ""),
        "test -f ~/.axropus/config.json": (0, "active\n", ""),
    }
    _patch_paramiko(monkeypatch, _MockSSHClient(commands))

    deploy = client.post(
        "/api/deploy",
        headers=headers,
        json={
            "host": "10.0.1.50",
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": "safe-pass",
            "api_key_id": 1,
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "70B",
        },
    )
    assert deploy.status_code == 200
    body = deploy.json()
    assert isinstance(body["deployment_id"], int)
    assert body["ws_url"].endswith(str(body["deployment_id"]))

    status = _wait_for_deploy_done(client, body["deployment_id"], headers)
    assert status["status"] == "active"
    messages = [item["message"] for item in status["logs"]]
    assert any("DEPLOYMENT COMPLETE" in m for m in messages)


def test_failed_ssh_connection_handling(client, monkeypatch):
    _, headers = _signup_and_headers(client, "deploy-fail-ssh@axropus.com")
    _patch_paramiko(monkeypatch, _MockSSHClient({}, connect_error=RuntimeError("conn failed")))

    deploy = client.post(
        "/api/deploy",
        headers=headers,
        json={
            "host": "10.0.1.51",
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": "bad-pass",
            "api_key_id": 1,
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "8B",
        },
    )
    assert deploy.status_code == 200
    dep_id = deploy.json()["deployment_id"]
    status = _wait_for_deploy_done(client, dep_id, headers)
    assert status["status"] == "failed"
    assert any("Deployment failed" in row["message"] for row in status["logs"])


def test_python_version_too_low_handling(client, monkeypatch):
    _, headers = _signup_and_headers(client, "deploy-low-python@axropus.com")
    commands = {
        "--version": (0, "Python 3.9.18\n", ""),
    }
    _patch_paramiko(monkeypatch, _MockSSHClient(commands))

    deploy = client.post(
        "/api/deploy",
        headers=headers,
        json={
            "host": "10.0.1.52",
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": "p",
            "api_key_id": 1,
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "8B",
        },
    )
    assert deploy.status_code == 200
    dep_id = deploy.json()["deployment_id"]
    status = _wait_for_deploy_done(client, dep_id, headers)
    assert status["status"] == "failed"
    assert any("below required 3.10" in row["message"] for row in status["logs"])


def test_runtime_not_detected_handling(client, monkeypatch):
    _, headers = _signup_and_headers(client, "deploy-runtime-miss@axropus.com")
    commands = {
        "--version": (0, "Python 3.12.1\n", ""),
        "-m pip install --upgrade axropus": (0, "ok\n", ""),
        "ps aux | grep -E 'vllm|sglang|triton'": (0, "", ""),
        "import vllm; print(vllm.__version__)": (1, "", "ModuleNotFoundError"),
    }
    _patch_paramiko(monkeypatch, _MockSSHClient(commands))

    deploy = client.post(
        "/api/deploy",
        headers=headers,
        json={
            "host": "10.0.1.53",
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": "p",
            "api_key_id": 1,
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "8B",
        },
    )
    dep_id = deploy.json()["deployment_id"]
    status = _wait_for_deploy_done(client, dep_id, headers)
    assert status["status"] == "failed"
    assert any("Runtime not detected" in row["message"] for row in status["logs"])


def test_partial_failure_recovery(client, monkeypatch):
    _, headers = _signup_and_headers(client, "deploy-partial@axropus.com")
    commands = {
        "--version": (0, "Python 3.12.0\n", ""),
        "-m pip install --upgrade axropus": (0, "axropus-0.1.0\n", ""),
        "ps aux | grep -E 'vllm|sglang|triton'": (0, "vllm", ""),
        "import vllm; print(vllm.__version__)": (0, "0.6.2", ""),
        "ps aux | grep -E -- '--model|model='": (0, "--model llama", ""),
        "huggingface-cli download": (1, "", "download failed"),
    }
    _patch_paramiko(monkeypatch, _MockSSHClient(commands))

    deploy = client.post(
        "/api/deploy",
        headers=headers,
        json={
            "host": "10.0.1.54",
            "port": 22,
            "username": "ubuntu",
            "auth_method": "password",
            "password": "p",
            "api_key_id": 1,
            "runtime": "vllm",
            "model_family": "llama",
            "model_size": "70B",
        },
    )
    dep_id = deploy.json()["deployment_id"]
    status = _wait_for_deploy_done(client, dep_id, headers)
    assert status["status"] == "failed"
    messages = [row["message"] for row in status["logs"]]
    assert any(
        ("Installing axropus SDK" in msg) or ("axropus" in msg.lower() and "installed" in msg.lower())
        for msg in messages
    )
    assert any("Draft model download failed" in msg for msg in messages)


def test_credentials_never_in_logs_or_db(client, monkeypatch):
    _, headers = _signup_and_headers(client, "deploy-credentials@axropus.com")
    secret_password = "SuperSecretPass123"
    secret_key = "-----BEGIN PRIVATE KEY-----TOPSECRET-----END PRIVATE KEY-----"
    _patch_paramiko(monkeypatch, _MockSSHClient({}, connect_error=RuntimeError("auth error")))

    deploy = client.post(
        "/api/deploy",
        headers=headers,
        json={
            "host": "10.0.1.55",
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
    status = _wait_for_deploy_done(client, dep_id, headers)

    log_blob = "\n".join(row["message"] for row in status["logs"])
    assert secret_password not in log_blob
    assert secret_key not in log_blob

    from backend.db import SessionLocal
    from backend.models import Deployment

    db = SessionLocal()
    try:
        row = db.get(Deployment, dep_id)
        assert row is not None
        row_dump = f"{row.runtime} {row.model_family} {row.model_size}"
        assert secret_password not in row_dump
        assert secret_key not in row_dump
    finally:
        db.close()
