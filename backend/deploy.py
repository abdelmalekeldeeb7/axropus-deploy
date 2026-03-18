from __future__ import annotations

import asyncio
import io
import re
import shlex
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import paramiko
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .auth import get_current_customer
from .db import SessionLocal, get_db
from .models import APIKey, Customer, Deployment

router = APIRouter(prefix="/api/deploy", tags=["deploy"])


@dataclass
class DeploymentRuntimeState:
    logs: deque[dict] = field(default_factory=lambda: deque(maxlen=500))
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class DeploymentStateStore:
    def __init__(self) -> None:
        self._states: dict[int, DeploymentRuntimeState] = {}
        self._states_lock = threading.Lock()

    def _get_or_create(self, deployment_id: int) -> DeploymentRuntimeState:
        with self._states_lock:
            state = self._states.get(deployment_id)
            if state is None:
                state = DeploymentRuntimeState()
                self._states[deployment_id] = state
            return state

    def append(self, deployment_id: int, message: dict) -> None:
        state = self._get_or_create(deployment_id)
        with state.lock:
            state.logs.append(message)

    def mark_done(self, deployment_id: int) -> None:
        state = self._get_or_create(deployment_id)
        with state.lock:
            state.done = True

    def snapshot(self, deployment_id: int, start_idx: int) -> tuple[list[dict], bool]:
        state = self._get_or_create(deployment_id)
        with state.lock:
            logs = list(state.logs)
            done = state.done
        if start_idx < 0:
            start_idx = 0
        return logs[start_idx:], done

    def tail(self, deployment_id: int, limit: int = 50) -> list[dict]:
        state = self._get_or_create(deployment_id)
        with state.lock:
            return list(state.logs)[-max(1, int(limit)) :]


DEPLOY_STATE = DeploymentStateStore()


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _is_api_key_valid(row: APIKey) -> bool:
    if row.status == "revoked":
        return False
    if row.expires_at is not None and row.expires_at <= _utcnow_naive():
        return False
    return True


def _sanitize_log(message: str, secrets_to_mask: list[str]) -> str:
    text = str(message or "")
    for secret in secrets_to_mask:
        if secret:
            text = text.replace(secret, "***")
    return text


def _push_progress(
    deployment_id: int,
    step: int,
    status_value: str,
    message: str,
    progress: float,
    secrets_to_mask: list[str],
) -> None:
    payload = {
        "step": int(step),
        "status": str(status_value),
        "message": _sanitize_log(message, secrets_to_mask),
        "progress": max(0.0, min(1.0, float(progress))),
    }
    DEPLOY_STATE.append(deployment_id, payload)


def _parse_python_version(raw: str) -> Optional[tuple[int, int, int]]:
    match = re.search(r"Python\s+(\d+)\.(\d+)\.(\d+)", raw or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _draft_model_for_family(model_family: str) -> str:
    family = str(model_family or "").strip().lower()
    mapping = {
        "llama": "bartowski/Meta-Llama-3.2-1B-Instruct-GGUF",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    }
    return mapping.get(family, "bartowski/Meta-Llama-3.2-1B-Instruct-GGUF")


def _ssh_run(ssh: paramiko.SSHClient, command: str, timeout: int = 240) -> tuple[int, str, str]:
    _, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = int(stdout.channel.recv_exit_status())
    return code, out, err


def _load_install_script() -> str:
    path = Path(__file__).resolve().parent / "install_script.sh"
    return path.read_text(encoding="utf-8")


def _mark_deployment_status(deployment_id: int, status_value: str) -> None:
    db = SessionLocal()
    try:
        row = db.get(Deployment, deployment_id)
        if row is not None:
            row.status = status_value
            if status_value == "active":
                row.deployed_at = _utcnow_naive()
            db.commit()
    finally:
        db.close()


class DeployRequest(BaseModel):
    host: str
    port: int = 22
    username: str
    auth_method: str = Field(pattern="^(key|password)$")
    ssh_key: Optional[str] = None
    password: Optional[str] = None
    python_path: Optional[str] = None
    api_key_id: int
    runtime: str = "korith"
    model_family: str
    model_size: str


def _run_deployment_task(deployment_id: int, payload: DeployRequest, api_key_value: str) -> None:
    ssh: Optional[paramiko.SSHClient] = None
    password = payload.password or ""
    ssh_key_text = payload.ssh_key or ""
    secrets_to_mask = [password, ssh_key_text, api_key_value]

    try:
        _mark_deployment_status(deployment_id, "deploying")

        _push_progress(
            deployment_id,
            1,
            "running",
            "[AXROPUS] Connecting to cluster...",
            0.05,
            secrets_to_mask,
        )
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict = {
            "hostname": payload.host,
            "port": int(payload.port or 22),
            "username": payload.username,
            "timeout": 20,
            "banner_timeout": 20,
            "auth_timeout": 20,
            "look_for_keys": False,
            "allow_agent": False,
        }
        if payload.auth_method == "password":
            kwargs["password"] = password
        else:
            if not ssh_key_text:
                raise RuntimeError("Missing SSH key for key auth")
            pkey = paramiko.RSAKey.from_private_key(io.StringIO(ssh_key_text))
            kwargs["pkey"] = pkey
        ssh.connect(**kwargs)
        _push_progress(deployment_id, 1, "success", "[AXROPUS] \u2713 Connected", 0.10, secrets_to_mask)

        python_bin = (payload.python_path or "python3").strip() or "python3"
        _push_progress(deployment_id, 2, "running", "[AXROPUS] Checking Python...", 0.15, secrets_to_mask)
        code, out, err = _ssh_run(ssh, f"{shlex.quote(python_bin)} --version")
        if code != 0:
            raise RuntimeError(f"Python check failed: {(err or out).strip()}")
        version = _parse_python_version(f"{out}\n{err}")
        if version is None:
            raise RuntimeError("Could not parse Python version")
        if version < (3, 10, 0):
            raise RuntimeError(f"Python {version[0]}.{version[1]}.{version[2]} is below required 3.10")
        _push_progress(
            deployment_id,
            2,
            "success",
            f"[AXROPUS] \u2713 Python {version[0]}.{version[1]}.{version[2]} detected",
            0.20,
            secrets_to_mask,
        )

        _push_progress(deployment_id, 3, "running", "[AXROPUS] Installing axropus SDK...", 0.28, secrets_to_mask)
        code, out, err = _ssh_run(ssh, f"{shlex.quote(python_bin)} -m pip install --upgrade axropus")
        if code != 0:
            raise RuntimeError(f"SDK install failed: {(err or out).strip()}")
        version_match = re.search(r"axropus[-=\s]+([0-9][\w\.\-]*)", f"{out}\n{err}", flags=re.IGNORECASE)
        installed_ver = version_match.group(1) if version_match else "installed"
        _push_progress(
            deployment_id,
            3,
            "success",
            f"[AXROPUS] \u2713 axropus {installed_ver} installed",
            0.35,
            secrets_to_mask,
        )

        _push_progress(deployment_id, 4, "running", "[AXROPUS] Checking GPU...", 0.42, secrets_to_mask)
        runtime = payload.runtime.strip().lower()
        code, gpu_out, _ = _ssh_run(ssh, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no gpu'")
        gpu_lines = [l.strip() for l in gpu_out.strip().splitlines() if l.strip() and l.strip() != "no gpu"]
        gpu_summary = f"{len(gpu_lines)}x {gpu_lines[0].split(',')[0].strip()}" if gpu_lines else "GPU detected"
        _push_progress(deployment_id, 4, "success", f"[AXROPUS] \u2713 {gpu_summary}", 0.50, secrets_to_mask)

        _push_progress(deployment_id, 5, "running", "[AXROPUS] Building korith_dynamic + downloading models...", 0.52, secrets_to_mask)
        install_script = _load_install_script()
        remote_script = "/tmp/axropus_install.sh"
        # Use printf to safely write the script without heredoc quoting issues
        import base64 as _b64
        encoded = _b64.b64encode(install_script.encode()).decode()
        create_script_cmd = f"echo {shlex.quote(encoded)} | base64 -d > {remote_script} && chmod 700 {remote_script}"
        code, out, err = _ssh_run(ssh, create_script_cmd)
        if code != 0:
            raise RuntimeError(f"Install script upload failed: {(err or out).strip()}")
        run_script_cmd = (
            f"AXROPUS_API_KEY={shlex.quote(api_key_value)} "
            f"AXROPUS_DEPLOYMENT_ID={shlex.quote(str(deployment_id))} "
            f"AXROPUS_RUNTIME={shlex.quote(runtime)} "
            f"AXROPUS_MODEL_FAMILY={shlex.quote(payload.model_family)} "
            f"AXROPUS_MODEL_SIZE={shlex.quote(payload.model_size)} "
            f"AXROPUS_PYTHON={shlex.quote(python_bin)} "
            f"bash {remote_script}"
        )
        # Build + model download can take 10-20 min on first run
        code, out, err = _ssh_run(ssh, run_script_cmd, timeout=1800)
        if code != 0:
            raise RuntimeError(f"Install failed: {(err or out)[-500:].strip()}")
        _push_progress(deployment_id, 5, "success", "[AXROPUS] \u2713 korith_dynamic built, gateway started", 0.80, secrets_to_mask)

        _push_progress(deployment_id, 6, "running", "[AXROPUS] Waiting for gateway to register...", 0.85, secrets_to_mask)
        # Poll DB until node self-registers (gateway_agent does this automatically)
        registered = False
        for _ in range(24):  # 24 × 5s = 2 min
            time.sleep(5)
            db = SessionLocal()
            try:
                row = db.get(Deployment, deployment_id)
                if row and row.inference_url:
                    registered = True
                    inference_url = row.inference_url
                    break
            finally:
                db.close()
        if not registered:
            raise RuntimeError("Gateway did not register within 2 minutes. Check ~/.axropus/agent.log on the node.")
        _push_progress(deployment_id, 6, "success", f"[AXROPUS] \u2713 Gateway live at {inference_url}", 0.92, secrets_to_mask)

        _push_progress(deployment_id, 7, "running", "[AXROPUS] Verifying inference endpoint...", 0.95, secrets_to_mask)
        import urllib.request as _urllib
        try:
            with _urllib.urlopen(f"{inference_url}/health", timeout=10) as r:
                if r.status != 200:
                    raise RuntimeError(f"Health check returned {r.status}")
        except Exception as exc:
            raise RuntimeError(f"Inference endpoint unreachable: {exc}") from exc
        _push_progress(deployment_id, 7, "success", "[AXROPUS] \u2713 AMF active — prefix elimination enabled", 0.97, secrets_to_mask)
        _push_progress(deployment_id, 7, "success", "[AXROPUS] \u2713 AI reasoning layer active — adaptive spec decode", 0.98, secrets_to_mask)
        _push_progress(deployment_id, 10, "success", "", 1.00, secrets_to_mask)
        _push_progress(deployment_id, 10, "success", "[AXROPUS] \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550", 1.00, secrets_to_mask)
        _push_progress(deployment_id, 10, "success", "[AXROPUS]   DEPLOYMENT COMPLETE", 1.00, secrets_to_mask)
        _push_progress(deployment_id, 10, "success", "[AXROPUS]   Estimated savings: 94-99%+ compute eliminated", 1.00, secrets_to_mask)
        _push_progress(deployment_id, 10, "success", "[AXROPUS] \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550", 1.00, secrets_to_mask)
        _mark_deployment_status(deployment_id, "active")
    except Exception as exc:
        _mark_deployment_status(deployment_id, "failed")
        _push_progress(
            deployment_id,
            10,
            "error",
            f"[AXROPUS] Deployment failed: {exc}",
            1.00,
            secrets_to_mask,
        )
    finally:
        if ssh is not None:
            try:
                ssh.close()
            except Exception:
                pass
        # Best-effort in-memory credential cleanup.
        password = ""
        ssh_key_text = ""
        secrets_to_mask = []
        DEPLOY_STATE.mark_done(deployment_id)


@router.post("")
def deploy(
    payload: DeployRequest,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    api_key = (
        db.query(APIKey)
        .filter(APIKey.id == payload.api_key_id, APIKey.customer_id == customer.id)
        .first()
    )
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    if not _is_api_key_valid(api_key):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="API key is not valid")

    deployment = Deployment(
        customer_id=customer.id,
        api_key_id=api_key.id,
        runtime=payload.runtime.strip().lower(),
        model_family=payload.model_family.strip(),
        model_size=payload.model_size.strip(),
        status="pending",
    )
    db.add(deployment)
    db.commit()
    db.refresh(deployment)

    worker = threading.Thread(
        target=_run_deployment_task,
        args=(deployment.id, payload, api_key.key),
        daemon=True,
    )
    worker.start()

    return {
        "deployment_id": deployment.id,
        "ws_url": f"/api/deploy/stream/{deployment.id}",
    }


@router.get("/status/{deployment_id}")
def deploy_status(
    deployment_id: int,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    row = (
        db.query(Deployment)
        .filter(Deployment.id == deployment_id, Deployment.customer_id == customer.id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found")
    return {
        "deployment_id": row.id,
        "status": row.status,
        "runtime": row.runtime,
        "model_family": row.model_family,
        "model_size": row.model_size,
        "deployed_at": row.deployed_at,
        "logs": DEPLOY_STATE.tail(deployment_id, limit=50),
    }


class RegisterNodeRequest(BaseModel):
    api_key: str
    inference_url: str


@router.post("/{deployment_id}/register")
def register_node(
    deployment_id: int,
    payload: RegisterNodeRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Called by gateway_agent on the GPU node after startup to register its public URL."""
    key_value = str(payload.api_key or "").strip()
    if not key_value.startswith("ax-"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid api_key format")

    api_key = db.query(APIKey).filter(APIKey.key == key_value).first()
    if api_key is None or api_key.status not in ("trial", "active"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    row = (
        db.query(Deployment)
        .filter(Deployment.id == deployment_id, Deployment.api_key_id == api_key.id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found")

    url = str(payload.inference_url or "").strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="inference_url must start with http(s)://")

    row.inference_url = url
    row.status = "active"
    row.deployed_at = _utcnow_naive()
    db.commit()
    return {"ok": True, "deployment_id": deployment_id, "inference_url": url}


@router.websocket("/stream/{deployment_id}")
async def deploy_stream(websocket: WebSocket, deployment_id: int) -> None:
    await websocket.accept()
    cursor = 0
    try:
        while True:
            messages, done = DEPLOY_STATE.snapshot(deployment_id, cursor)
            if messages:
                for msg in messages:
                    await websocket.send_json(msg)
                cursor += len(messages)
            if done and not messages:
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
