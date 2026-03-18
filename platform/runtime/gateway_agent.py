#!/usr/bin/env python3
"""
gateway_agent.py — Axropus node agent.

Replaces `axropus agent start`. Runs on the customer's GPU node after install_script.sh:
  1. Finds the model GGUF on disk
  2. Sets env vars for the platform runtime
  3. Starts workers + router in background threads
  4. Self-registers public IP + port with axropus.app backend
  5. Sends metrics heartbeat every 60s
  6. (Optional) starts AI reasoning orchestrator if deps available
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gateway_agent")

_GATEWAY_PORT = int(os.environ.get("AXROPUS_GATEWAY_PORT", "8080"))
_METRICS_INTERVAL = int(os.environ.get("AXROPUS_METRICS_INTERVAL", "60"))
_REGISTER_RETRIES = int(os.environ.get("AXROPUS_REGISTER_RETRIES", "12"))  # 12 × 10s = 2min
_SDK_VERSION = "0.2.0"


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_public_ip() -> str:
    for url in ("https://ifconfig.me", "https://api.ipify.org", "https://checkip.amazonaws.com"):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.read().decode().strip()
        except Exception:
            pass
    # Fallback: outbound IP via UDP trick (no packet sent)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _find_model(models_dir: Path, family: str, size: str) -> Path | None:
    """Scan models_dir for a GGUF matching family + size hints."""
    if not models_dir.exists():
        return None
    family_lc = family.lower()
    size_lc = size.lower()
    # Prefer exact match, then any GGUF
    candidates = list(models_dir.glob("*.gguf"))
    for p in candidates:
        name = p.name.lower()
        if family_lc in name and size_lc in name:
            return p
    for p in candidates:
        if family_lc in p.name.lower():
            return p
    return candidates[0] if candidates else None


def _find_draft_model(models_dir: Path) -> Path | None:
    for p in models_dir.glob("*.gguf"):
        if "1b" in p.name.lower() or "draft" in p.name.lower():
            return p
    return None


def _post_json(url: str, payload: dict, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── self-registration ─────────────────────────────────────────────────────────

def _register_with_backend(backend_url: str, api_key: str, deployment_id: str, inference_url: str) -> bool:
    url = f"{backend_url.rstrip('/')}/api/deploy/{deployment_id}/register"
    try:
        resp = _post_json(url, {"api_key": api_key, "inference_url": inference_url})
        return bool(resp.get("ok"))
    except Exception as exc:
        log.warning("Registration failed: %s", exc)
        return False


def _wait_and_register(backend_url: str, api_key: str, deployment_id: str, port: int) -> None:
    public_ip = _get_public_ip()
    inference_url = f"http://{public_ip}:{port}"
    log.info("Public inference URL: %s", inference_url)

    for attempt in range(1, _REGISTER_RETRIES + 1):
        # Wait for gateway to be healthy first
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        log.info("Waiting for gateway to be ready (attempt %d/%d)...", attempt, _REGISTER_RETRIES)
        time.sleep(10)
    else:
        log.error("Gateway never became healthy — skipping registration")
        return

    for attempt in range(1, 4):
        if _register_with_backend(backend_url, api_key, deployment_id, inference_url):
            log.info("Registered with backend: %s", inference_url)
            return
        log.warning("Registration attempt %d/3 failed, retrying...", attempt)
        time.sleep(5)
    log.error("Failed to register with backend after 3 attempts")


# ── metrics heartbeat ─────────────────────────────────────────────────────────

def _metrics_loop(backend_url: str, api_key: str, router_ref: list, gpu_count: int,
                  model_family: str, model_size: str) -> None:
    url = f"{backend_url.rstrip('/')}/api/metrics"
    while True:
        time.sleep(_METRICS_INTERVAL)
        try:
            router = router_ref[0] if router_ref else None
            amf_hit_rate = 0.0
            spec_rate = 0.0
            effective_tps = 0.0
            baseline_tps = 0.0
            compute_saved = 0.0
            tokens_processed = 0
            prefix_skipped = 0
            decode_accelerated = 0

            if router is not None and hasattr(router, "get_workers"):
                workers = router.get_workers()
                hit_rates = [w.get("amf_hit_rate", 0.0) for w in workers if "amf_hit_rate" in w]
                if hit_rates:
                    amf_hit_rate = sum(hit_rates) / len(hit_rates)
                compute_saved = amf_hit_rate  # AMF hit rate ≈ compute saved fraction

            payload = {
                "api_key": api_key,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "interval_seconds": _METRICS_INTERVAL,
                "tokens_processed": tokens_processed,
                "prefix_skipped": prefix_skipped,
                "decode_accelerated": decode_accelerated,
                "amf_hit_rate": round(amf_hit_rate, 4),
                "spec_acceptance_rate": round(spec_rate, 4),
                "effective_tps": round(effective_tps, 2),
                "baseline_tps": round(baseline_tps, 2),
                "compute_saved_pct": round(compute_saved * 100, 2),
                "gpu_count": gpu_count,
                "model_family": model_family[:64] if model_family else None,
                "model_size_bucket": model_size[:64] if model_size else None,
                "adapter_type": "korith_local",
                "sdk_version": _SDK_VERSION,
                "heartbeat": 1,
            }
            _post_json(url, payload, timeout=10)
            log.info("Metrics sent: amf_hit_rate=%.3f compute_saved=%.1f%%",
                     amf_hit_rate, compute_saved * 100)
        except Exception as exc:
            log.warning("Metrics heartbeat failed: %s", exc)


# ── AI reasoning ──────────────────────────────────────────────────────────────

def _start_reasoning(router_ref: list) -> object | None:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from platform.reasoning.metrics_collector import MetricsCollector
        from platform.reasoning.nemoclaw_model import NemoClawReasoningModel
        from platform.reasoning.decision_executor import DecisionExecutor
        from platform.reasoning.reward_calculator import RewardCalculator
        from platform.reasoning.learning_loop import LearningLoop
        from platform.reasoning.orchestrator import Orchestrator

        log.info("[reasoning] Initialising NemoClaw reasoning layer...")

        nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "")
        nemoclaw_url   = os.environ.get("NEMOCLAW_BASE_URL", "https://integrate.api.nvidia.com/v1")

        collector   = MetricsCollector()
        model_r     = NemoClawReasoningModel(
            api_key=nvidia_api_key,
            base_url=nemoclaw_url,
            max_inference_ms=100.0,
            enable_reward_feedback=True,
        )
        executor    = DecisionExecutor(coordinator_client=None)
        reward_calc = RewardCalculator(
            db_path=str(Path.home() / ".axropus" / "rewards.db")
        )
        # LearningLoop kept for reward record-keeping but RL is handled by NemoClaw
        loop = LearningLoop(reasoning_model=model_r, reward_calculator=reward_calc, rl_level=0)
        orchestrator = Orchestrator(
            collector, model_r, executor, reward_calc, loop,
            enabled=True, ab_testing=False,
        )
        orchestrator.start()

        if nvidia_api_key:
            log.info("[reasoning] NemoClaw active — RL loop built-in, reward feedback enabled")
        else:
            log.warning("[reasoning] NemoClaw running without API key — set NVIDIA_API_KEY for RL")
        return orchestrator
    except Exception as exc:
        log.warning("[reasoning] Could not start NemoClaw layer: %s", exc)
        log.warning("[reasoning] Continuing without AI reasoning (AMF still active)")
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Axropus gateway agent")
    parser.add_argument("--config", default=str(Path.home() / ".axropus" / "gateway_config.json"))
    args = parser.parse_args()

    config_path = Path(args.config)
    config: dict = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    api_key = os.environ.get("AXROPUS_API_KEY") or config.get("api_key", "")
    deployment_id = os.environ.get("AXROPUS_DEPLOYMENT_ID") or config.get("deployment_id", "")
    backend_url = os.environ.get("AXROPUS_BACKEND_URL") or config.get("backend_url", "https://axropus-backend.up.railway.app")
    model_family = config.get("model_family", "llama")
    model_size = config.get("model_size", "70b")
    use_reasoning = str(config.get("reasoning", True)).lower() not in ("false", "0", "no")

    repo_dir = Path(__file__).parent.parent.parent
    models_dir = Path(os.environ.get("AXROPUS_MODELS_DIR", str(Path.home() / ".axropus" / "models")))
    binary = Path(os.environ.get("AXROPUS_BINARY", str(repo_dir / "build" / "korith_dynamic")))

    # Set env vars for the platform runtime
    if not os.environ.get("KORITH_DEFAULT_MODEL_PATH"):
        model_path = _find_model(models_dir, model_family, model_size)
        if model_path:
            os.environ["KORITH_DEFAULT_MODEL_PATH"] = str(model_path)
            log.info("Model: %s", model_path)
        else:
            log.warning("No model found in %s — workers will fail to start", models_dir)

    draft = _find_draft_model(models_dir)
    if draft and not os.environ.get("KORITH_DRAFT_MODEL_PATH"):
        os.environ["KORITH_DRAFT_MODEL_PATH"] = str(draft)
        log.info("Draft model: %s", draft)

    if binary.exists() and not os.environ.get("KORITH_BINARY"):
        os.environ["KORITH_BINARY"] = str(binary)

    os.environ.setdefault("KORITH_API_KEY_SALT", api_key[:32] if api_key else "axropus-default-salt")
    os.environ.setdefault("KORITH_DATA_DIR", str(Path.home() / ".axropus" / "data"))
    os.environ.setdefault("KORITH_AMF_PATH", str(Path.home() / ".axropus" / "amf"))
    os.environ.setdefault("KORITH_KV_FP8", "1")

    Path(os.environ["KORITH_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["KORITH_AMF_PATH"]).mkdir(parents=True, exist_ok=True)

    # Detect GPU count
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_count = len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except Exception:
        gpu_count = 1

    if gpu_count > 1:
        split = ",".join([f"{1.0/gpu_count:.4f}"] * gpu_count)
        os.environ.setdefault("KORITH_TENSOR_SPLIT", split)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(gpu_count)))
        log.info("Multi-GPU: %d GPUs, tensor_split=%s", gpu_count, split)

    # Start router (gateway + workers) in background thread
    router_ref: list = []

    def _start_router():
        try:
            sys.path.insert(0, str(repo_dir))
            from platform.runtime.router_service import run_router
            # run_router calls serve() which blocks — run in thread
            # Capture the router object via monkey-patch before serve blocks
            import platform.gateway.http_server as _hs
            _orig_serve = _hs.serve
            def _patched_serve(router, host="0.0.0.0", port=8000):
                router_ref.append(router)
                return _orig_serve(router, host=host, port=port)
            _hs.serve = _patched_serve
            run_router(host="0.0.0.0", port=_GATEWAY_PORT)
        except Exception as exc:
            log.error("Router failed: %s", exc, exc_info=True)

    router_thread = threading.Thread(target=_start_router, daemon=True)
    router_thread.start()
    log.info("Gateway starting on port %d...", _GATEWAY_PORT)

    # Start AI reasoning (optional)
    if use_reasoning:
        orchestrator = _start_reasoning(router_ref)
    else:
        orchestrator = None

    # Self-register with backend (waits for gateway to be healthy)
    if api_key and deployment_id:
        reg_thread = threading.Thread(
            target=_wait_and_register,
            args=(backend_url, api_key, deployment_id, _GATEWAY_PORT),
            daemon=True,
        )
        reg_thread.start()
    else:
        log.warning("No api_key/deployment_id — skipping backend registration")

    # Metrics heartbeat
    metrics_thread = threading.Thread(
        target=_metrics_loop,
        args=(backend_url, api_key, router_ref, gpu_count, model_family, model_size),
        daemon=True,
    )
    metrics_thread.start()

    # Block until killed
    def _handle_signal(sig, frame):
        log.info("Shutting down gateway agent...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info("Gateway agent running. Ctrl+C to stop.")
    router_thread.join()


if __name__ == "__main__":
    main()
