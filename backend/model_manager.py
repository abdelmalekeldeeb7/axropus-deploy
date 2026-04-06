"""Model lifecycle management — download, deploy, undeploy, and monitor models.

Manages the full lifecycle of model deployments on the Axropus Platform Hub,
including vLLM worker orchestration with AMF (Adaptive Memory Fusion)
acceleration, GPU auto-detection, and VRAM pool warming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

try:
    from .db import SessionLocal
    from .model_registry import ModelSpec, get_model_or_raise
except ImportError:
    from db import SessionLocal
    from model_registry import ModelSpec, get_model_or_raise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CACHE_DIR = Path("/data/models")
VLLM_HOST = "0.0.0.0"
VLLM_DEFAULT_PORT_BASE = 8100
MAX_CONCURRENT_DOWNLOADS = 2


class ModelStatus(str, Enum):
    """Lifecycle states for a model deployment."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------

def _detect_gpus() -> list[dict[str, Any]]:
    """Auto-detect NVIDIA GPUs via nvidia-smi. Returns list of GPU info dicts."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return []
    except FileNotFoundError:
        logger.warning("nvidia-smi not found — no GPU detection available")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return []

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "total_vram_mb": int(parts[2]),
            "free_vram_mb": int(parts[3]),
        })
    return gpus


def _select_amf_quant(spec: ModelSpec, available_vram_gb: float) -> str:
    """Choose the best quantization mode given available VRAM."""
    if available_vram_gb >= spec.min_vram_gb * 2:
        return "FP16"
    if available_vram_gb >= spec.min_vram_gb * 1.2:
        return spec.default_quant
    if available_vram_gb >= spec.min_vram_gb:
        return "GPTQ-4bit" if "AWQ" in spec.default_quant else spec.default_quant
    # Below minimum — try aggressive quantization
    return "GPTQ-3bit"


def _utcnow_naive() -> datetime:
    """Return current UTC time without tzinfo (for SQLite compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages model downloads, vLLM worker processes, and deployment state.

    State is persisted in the ``model_deployments`` table via SQLAlchemy.
    Active vLLM subprocesses are tracked in-memory; on restart the manager
    detects orphaned entries and marks them as stopped.
    """

    def __init__(self) -> None:
        self._processes: dict[int, subprocess.Popen] = {}
        self._download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        self._next_port = VLLM_DEFAULT_PORT_BASE

    # ── Database helpers ──────────────────────────────────────────────────

    @staticmethod
    def _get_db() -> Session:
        return SessionLocal()

    def _update_status(self, deployment_id: int, status: ModelStatus, *, error: str | None = None) -> None:
        """Persist status change to the database."""
        db = self._get_db()
        try:
            from .models import ModelDeployment  # deferred to avoid circular import
            row = db.get(ModelDeployment, deployment_id)
            if row is None:
                logger.error("ModelDeployment %d not found for status update", deployment_id)
                return
            row.status = status.value
            if error:
                row.error_message = error
            if status == ModelStatus.RUNNING:
                row.deployed_at = _utcnow_naive()
            if status in (ModelStatus.STOPPED, ModelStatus.FAILED):
                row.stopped_at = _utcnow_naive()
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Failed to update deployment %d status to %s", deployment_id, status.value)
        finally:
            db.close()

    # ── Download ──────────────────────────────────────────────────────────

    async def download_model(self, model_id: str) -> Path:
        """Download model weights from HuggingFace Hub.

        Uses a semaphore to limit concurrent downloads. Returns the local
        cache path where the model was stored.

        Raises:
            KeyError: If *model_id* is not in the registry.
            RuntimeError: If download fails.
        """
        spec = get_model_or_raise(model_id)
        cache_path = MODEL_CACHE_DIR / spec.id
        if cache_path.exists() and any(cache_path.iterdir()):
            logger.info("Model %s already cached at %s", model_id, cache_path)
            return cache_path

        async with self._download_semaphore:
            logger.info("Downloading model %s from %s ...", model_id, spec.source)
            cache_path.mkdir(parents=True, exist_ok=True)
            try:
                proc = await asyncio.create_subprocess_exec(
                    "huggingface-cli",
                    "download",
                    spec.source,
                    "--local-dir",
                    str(cache_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"huggingface-cli download failed (rc={proc.returncode}): "
                        f"{stderr.decode(errors='replace')[-500:]}"
                    )
                logger.info("Model %s downloaded successfully to %s", model_id, cache_path)
                return cache_path
            except Exception:
                # Clean up partial download
                if cache_path.exists():
                    shutil.rmtree(cache_path, ignore_errors=True)
                raise

    # ── Deploy ────────────────────────────────────────────────────────────

    async def deploy_model(
        self,
        deployment_id: int,
        model_id: str,
        *,
        port: int | None = None,
        tensor_parallel: int | None = None,
        max_model_len: int | None = None,
    ) -> dict[str, Any]:
        """Deploy a model by starting a vLLM worker with AMF acceleration.

        Steps:
          1. Auto-detect GPU(s) and compute available VRAM.
          2. Configure AMF quantization mode based on hardware.
          3. Download model weights if not cached.
          4. Start vLLM subprocess with AMF extensions.
          5. Warm the VRAM pool for fast first-token latency.

        Returns a dict with deployment details (port, quant, gpus).
        """
        spec = get_model_or_raise(model_id)
        self._update_status(deployment_id, ModelStatus.DOWNLOADING)

        # Step 1: GPU detection
        gpus = _detect_gpus()
        total_vram_gb = sum(g["free_vram_mb"] for g in gpus) / 1024.0 if gpus else 0
        if total_vram_gb < spec.min_vram_gb:
            error_msg = (
                f"Insufficient VRAM: {total_vram_gb:.1f} GB available, "
                f"{spec.min_vram_gb} GB required for {spec.name}"
            )
            self._update_status(deployment_id, ModelStatus.FAILED, error=error_msg)
            raise RuntimeError(error_msg)

        # Step 2: AMF quant mode
        quant_mode = _select_amf_quant(spec, total_vram_gb)
        tp_size = tensor_parallel or len(gpus) or 1

        # Step 3: Download
        try:
            model_path = await self.download_model(model_id)
        except Exception as exc:
            self._update_status(deployment_id, ModelStatus.FAILED, error=str(exc))
            raise

        # Step 4: Start vLLM process
        self._update_status(deployment_id, ModelStatus.DEPLOYING)
        serve_port = port or self._next_port
        self._next_port = serve_port + 1

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_path),
            "--host", VLLM_HOST,
            "--port", str(serve_port),
            "--tensor-parallel-size", str(tp_size),
            "--dtype", "auto",
            "--trust-remote-code",
            "--enable-prefix-caching",
        ]

        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])
        elif spec.context_window > 32_768:
            # Default cap to manage VRAM on constrained hardware
            cmd.extend(["--max-model-len", str(min(spec.context_window, 65536))])

        if quant_mode.startswith("AWQ"):
            cmd.extend(["--quantization", "awq"])
        elif quant_mode.startswith("GPTQ"):
            cmd.extend(["--quantization", "gptq"])

        logger.info(
            "Starting vLLM for %s (tp=%d, quant=%s, port=%d): %s",
            model_id, tp_size, quant_mode, serve_port, " ".join(cmd),
        )

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=None,  # inherit environment
            )
            self._processes[deployment_id] = process
        except Exception as exc:
            self._update_status(deployment_id, ModelStatus.FAILED, error=f"Failed to start vLLM: {exc}")
            raise

        # Wait briefly to check for immediate crash
        try:
            await asyncio.sleep(3)
            if process.poll() is not None:
                stderr_out = ""
                if process.stderr:
                    stderr_out = process.stderr.read().decode(errors="replace")[-500:]
                error_msg = f"vLLM exited immediately (rc={process.returncode}): {stderr_out}"
                self._update_status(deployment_id, ModelStatus.FAILED, error=error_msg)
                self._processes.pop(deployment_id, None)
                raise RuntimeError(error_msg)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Error checking vLLM startup: %s", exc)

        # Step 5: Warm VRAM pool
        await self.warm_vram_pool(serve_port, model_id)

        self._update_status(deployment_id, ModelStatus.RUNNING)

        # Persist port and config in DB
        db = self._get_db()
        try:
            from .models import ModelDeployment
            row = db.get(ModelDeployment, deployment_id)
            if row:
                row.port = serve_port
                row.quant_mode = quant_mode
                row.tensor_parallel = tp_size
                row.gpu_count = len(gpus)
                db.commit()
        finally:
            db.close()

        return {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "port": serve_port,
            "quant_mode": quant_mode,
            "tensor_parallel": tp_size,
            "gpu_count": len(gpus),
            "status": ModelStatus.RUNNING.value,
        }

    # ── Undeploy ──────────────────────────────────────────────────────────

    async def undeploy_model(self, deployment_id: int) -> dict[str, Any]:
        """Stop a running vLLM worker and free resources.

        Returns a summary dict with the final status.
        """
        self._update_status(deployment_id, ModelStatus.STOPPING)
        process = self._processes.pop(deployment_id, None)

        if process is not None and process.poll() is None:
            logger.info("Terminating vLLM process for deployment %d (pid=%d)", deployment_id, process.pid)
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM process %d did not terminate; killing", process.pid)
                process.kill()
                process.wait(timeout=10)
        else:
            logger.info("No active process found for deployment %d", deployment_id)

        self._update_status(deployment_id, ModelStatus.STOPPED)
        return {"deployment_id": deployment_id, "status": ModelStatus.STOPPED.value}

    # ── Status ────────────────────────────────────────────────────────────

    def get_model_status(self, deployment_id: int) -> dict[str, Any]:
        """Get the current status of a deployment, including process health."""
        db = self._get_db()
        try:
            from .models import ModelDeployment
            row = db.get(ModelDeployment, deployment_id)
            if row is None:
                return {"error": "Deployment not found", "deployment_id": deployment_id}

            result: dict[str, Any] = {
                "deployment_id": row.id,
                "model_id": row.model_id,
                "status": row.status,
                "port": row.port,
                "quant_mode": row.quant_mode,
                "tensor_parallel": row.tensor_parallel,
                "gpu_count": row.gpu_count,
                "deployed_at": str(row.deployed_at) if row.deployed_at else None,
                "stopped_at": str(row.stopped_at) if row.stopped_at else None,
                "error_message": row.error_message,
            }

            # Cross-check with live process
            process = self._processes.get(deployment_id)
            if process is not None:
                result["process_alive"] = process.poll() is None
                result["pid"] = process.pid
            else:
                result["process_alive"] = False

            return result
        finally:
            db.close()

    def list_deployed_models(self) -> list[dict[str, Any]]:
        """List all model deployments and their current status."""
        db = self._get_db()
        try:
            from .models import ModelDeployment
            rows = db.query(ModelDeployment).order_by(ModelDeployment.id.desc()).all()
            results: list[dict[str, Any]] = []
            for row in rows:
                process = self._processes.get(row.id)
                results.append({
                    "deployment_id": row.id,
                    "model_id": row.model_id,
                    "status": row.status,
                    "port": row.port,
                    "quant_mode": row.quant_mode,
                    "gpu_count": row.gpu_count,
                    "deployed_at": str(row.deployed_at) if row.deployed_at else None,
                    "process_alive": process is not None and process.poll() is None,
                })
            return results
        finally:
            db.close()

    # ── VRAM warm-up ──────────────────────────────────────────────────────

    async def warm_vram_pool(self, port: int, model_id: str) -> None:
        """Send a small prefill request to warm CUDA memory allocators.

        This reduces first-token latency for real requests by pre-allocating
        KV-cache buffers and triggering JIT compilation.
        """
        import httpx

        url = f"http://127.0.0.1:{port}/v1/completions"
        payload = {
            "model": model_id,
            "prompt": "Warmup request.",
            "max_tokens": 1,
            "temperature": 0.0,
        }

        for attempt in range(1, 61):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        logger.info("VRAM pool warmed for %s on port %d (attempt %d)", model_id, port, attempt)
                        return
                    logger.debug("Warm-up attempt %d returned status %d", attempt, resp.status_code)
            except Exception:
                pass
            await asyncio.sleep(2)

        logger.warning("VRAM warm-up did not succeed after 60 attempts for %s on port %d", model_id, port)

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Terminate all managed vLLM processes. Called during app shutdown."""
        for deployment_id in list(self._processes):
            try:
                await self.undeploy_model(deployment_id)
            except Exception:
                logger.exception("Error shutting down deployment %d", deployment_id)


# Module-level singleton
model_manager = ModelManager()
