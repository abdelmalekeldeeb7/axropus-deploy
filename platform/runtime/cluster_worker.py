from __future__ import annotations

import json
import importlib.util
import math
import os
import shutil
import sysconfig
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..adapters.registry import AdapterRegistry
from ..engine.snapshot import snapshot_meta_path, validate_snapshot_metadata, write_snapshot_metadata
from ..artifacts.adapter import ArtifactStoreAdapter
from ..cluster.node_registry import NodeRegistry
from ..cluster.snapshot_index import SnapshotIndex
from ..cluster.snapshot_transfer import fetch_snapshot_bytes
from ..kernels.registry import resolve_kernel_backend
from ..ledger.store import LedgerStore
from ..observability.metrics import GLOBAL_METRICS
from ..observability.platform_logging import emit_log, error_fields
from ..queue.base import QueueBase
from .decode_cache_store import DecodeCacheStore
from .registry import WorkerRegistry
from .restore_store import RestoreStore

try:
    from ..reasoning.orchestrator import Orchestrator as _Orchestrator
except Exception:
    _Orchestrator = None  # type: ignore[assignment,misc]

_TRUTHY = ("1", "true", "yes", "on")


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in _TRUTHY


def apply_decode_opt_profile_env_defaults() -> bool:
    if not _env_truthy("KORITH_DECODE_OPT_PROFILE", default=False):
        return False

    defaults = {
        # Step 2 quick wins: accel path + kernels + deterministic decode cache.
        "KORITH_ACCEL_ENABLED": "1",
        "KORITH_KERNELS": "1",
        "KORITH_KERNEL_BACKEND": "cuda",
        "KORITH_KERNEL_VERIFY": "0",
        "KORITH_DECODE_CACHE_ENABLED": "1",
        "KORITH_DECODE_CACHE_REQUIRE_DETERMINISTIC": "1",
        "KORITH_DECODE_SCHEDULER_ENABLED": "1",
        "KORITH_DECODE_ADAPTIVE_BUDGET_ENABLED": "1",
        "KORITH_DECODE_SHAPE_AWARE_SCHEDULER": "1",
        "KORITH_VLLM_EXECUTOR_ALL_LANES": "1",
        "KORITH_SPEC_MISS_ONLY": "1",
    }
    for key, value in defaults.items():
        if not str(os.environ.get(key, "")).strip():
            os.environ[key] = value
    return True


def _load_stdlib_queue_module():
    try:
        import queue as candidate  # type: ignore
        if hasattr(candidate, "Queue"):
            return candidate
    except Exception:
        pass
    stdlib_path = Path(sysconfig.get_path("stdlib")) / "queue.py"
    spec = importlib.util.spec_from_file_location("_korith_stdlib_queue", stdlib_path)
    if not spec or not spec.loader:
        raise RuntimeError("failed to load stdlib queue module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = _load_stdlib_queue_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bucket_ceil(value: int, bucket: int) -> int:
    if bucket <= 0:
        return max(0, int(value))
    v = max(0, int(value))
    return ((v + bucket - 1) // bucket) * bucket


def _backend_is_local_runtime(backend_id: str) -> bool:
    return str(backend_id or "").strip().lower() in ("korith_local", "korith_cuda")


def _looks_like_oom(text: str) -> bool:
    msg = str(text or "").strip().lower()
    if not msg:
        return False
    patterns = (
        "out of memory",
        "cuda out of memory",
        "cudamalloc failed",
        "insufficient memory",
        "cannot allocate memory",
        "std::bad_alloc",
    )
    return any(p in msg for p in patterns)


def _parse_worker_lanes(raw: str) -> list[str]:
    values = []
    for part in str(raw or "").split(","):
        token = str(part or "").strip().upper()
        if not token:
            continue
        if token in ("ALL", "*"):
            return ["ALL"]
        if token == "HIT":
            values.extend(["HIT", "SPEC_HIT"])
            continue
        if token == "MISS":
            values.extend(["MISS", "SPEC_MISS"])
            continue
        if token in ("SPEC", "SPEC_ONLY"):
            values.extend(["SPEC_HIT", "SPEC_MISS"])
            continue
        if token in ("HIT_ONLY", "HOT"):
            values.extend(["HIT", "SPEC_HIT"])
            continue
        if token in ("MISS_ONLY", "COLD"):
            values.extend(["MISS", "SPEC_MISS"])
            continue
        if token in ("SPEC_HIT", "SPEC_MISS"):
            values.append(token)
    out: list[str] = []
    seen = set()
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def enforce_metrics_invariants(engine_metrics: Dict[str, Any], *, no_decode_cache_hit: bool = False) -> None:
    if not isinstance(engine_metrics, dict):
        return

    spec = engine_metrics.setdefault("spec", {})
    perf = engine_metrics.setdefault("perf", {})
    kernels = engine_metrics.setdefault("kernels", {})
    if not isinstance(spec, dict) or not isinstance(perf, dict) or not isinstance(kernels, dict):
        return

    cache_only = bool(spec.get("cache_only", False))
    cache_hit = bool(spec.get("cache_hit", False))
    tokens_out = int(perf.get("tokens_out", 0) or 0)

    baseline_total_ms = float(spec.get("baseline_total_ms", 0.0) or 0.0)
    if not (baseline_total_ms > 0.0):
        baseline_total_ms = 0.0

    # Cache-only no-decode results should report bounded/non-negative savings only.
    if cache_only and tokens_out <= 1:
        net_saved_ms = float(spec.get("net_saved_ms", 0.0) or 0.0)
        saved_ms = float(spec.get("saved_ms", net_saved_ms) or net_saved_ms)
        bounded_net = min(max(0.0, net_saved_ms), baseline_total_ms)
        bounded_saved = min(max(0.0, saved_ms), baseline_total_ms)
        spec["net_saved_ms"] = float(bounded_net)
        spec["saved_ms"] = float(bounded_saved)

    # Never attribute kernel savings when the response came from cache without decode.
    if cache_hit and no_decode_cache_hit:
        perf["tokens_out"] = 0
        perf["decode_ms"] = 0.0
        kernels["enabled"] = False
        kernels["kernels_applied"] = False
        kernels["ms_saved"] = 0.0

    if not bool(kernels.get("kernels_applied", False)):
        kernels["ms_saved"] = 0.0


class ClusterWorker:
    def __init__(
        self,
        worker_id: str,
        gpu_id: int,
        ledger: LedgerStore,
        artifacts: ArtifactStoreAdapter,
        queue_backend: QueueBase,
        registry: WorkerRegistry,
        restore_store: RestoreStore,
        adapter_registry: AdapterRegistry,
        snapshot_index: Optional[SnapshotIndex] = None,
        node_registry: Optional[NodeRegistry] = None,
        node_id: str = "",
        host: str = "localhost",
        amf_coordinator_client = None,
    ) -> None:
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.ledger = ledger
        self.artifacts = artifacts
        self.queue = queue_backend
        self.registry = registry
        self.restore_store = restore_store
        self.adapter_registry = adapter_registry
        self.snapshot_index = snapshot_index
        self.node_registry = node_registry
        self.node_id = node_id
        self.host = host
        self._amf_coordinator_client = amf_coordinator_client
        self._orchestrator: Optional[Any] = None
        self._amf_coordinator_entries: Dict[str, Dict[str, Any]] = {}
        self._amf_coordinator_heartbeat_s = max(
            1.0,
            float(os.environ.get("KORITH_AMF_COORDINATOR_HEARTBEAT_S", "10") or 10),
        )
        self._amf_coordinator_last_heartbeat = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._inflight = 0
        self._q_spec_hit: queue.Queue = queue.Queue()
        self._q_hit: queue.Queue = queue.Queue()
        self._q_spec_miss: queue.Queue = queue.Queue()
        self._q_miss: queue.Queue = queue.Queue()
        self._spec_priority = max(1, int(os.environ.get("KORITH_SPEC_PRIORITY", "4")))
        self._hit_priority = max(1, int(os.environ.get("KORITH_HIT_PRIORITY", "3")))
        self._miss_priority = max(1, int(os.environ.get("KORITH_MISS_PRIORITY", "1")))
        self._queue_budget = {
            "SPEC_HIT": self._spec_priority,
            "HIT": self._hit_priority,
            "SPEC_MISS": max(1, self._spec_priority // 2),
            "MISS": self._miss_priority,
        }
        self._decode_scheduler_enabled = os.environ.get("KORITH_DECODE_SCHEDULER_ENABLED", "0").strip().lower() in _TRUTHY
        self._decode_adaptive_budget_enabled = os.environ.get("KORITH_DECODE_ADAPTIVE_BUDGET_ENABLED", "0").strip().lower() in _TRUTHY
        self._decode_shape_aware_scheduler = os.environ.get("KORITH_DECODE_SHAPE_AWARE_SCHEDULER", "1").strip().lower() in _TRUTHY
        self._decode_budget_depth_weight = max(0.0, float(os.environ.get("KORITH_DECODE_BUDGET_DEPTH_WEIGHT", "0.35") or 0.35))
        self._lane_decode_ema_ms: Dict[str, float] = {}
        self._shape_decode_ema_ms: Dict[str, float] = {}
        self._avg_queue_latency_ms = 0.0
        self._negative_roi_max = max(1, int(os.environ.get("KORITH_NEGATIVE_ROI_MAX", "3")))
        self._negative_roi_cooldown_ms = max(1000, int(os.environ.get("KORITH_NEGATIVE_ROI_COOLDOWN_MS", "30000")))
        self._decode_opt_profile = apply_decode_opt_profile_env_defaults()
        self._accel_enabled = os.environ.get("KORITH_ACCEL_ENABLED", "0") == "1"
        self._engine_cfg = {
            "accel_enabled": self._accel_enabled,
            "cuda_device": int(os.environ.get("KORITH_CUDA_DEVICE", str(self.gpu_id))),
            "cuda_dtype": os.environ.get("KORITH_CUDA_DTYPE", "fp16"),
            "kv_layout_version": os.environ.get("KORITH_KV_LAYOUT_VERSION", "v1"),
        }
        self._spec_enabled = os.environ.get("KORITH_SPEC_ENABLED", "0") == "1"
        self._spec_cfg = {
            "enabled": self._spec_enabled,
            "k": max(1, int(os.environ.get("KORITH_SPEC_K", "6"))),
            "min_accept": max(0.0, min(1.0, float(os.environ.get("KORITH_SPEC_MIN_ACCEPT", "0.55")))),
            "disable_after_n": max(1, int(os.environ.get("KORITH_SPEC_DISABLE_AFTER_N", "50"))),
        }
        self._spec_min_output_tokens = max(1, int(os.environ.get("KORITH_SPEC_MIN_OUTPUT_TOKENS", "128")))
        self._spec_cache_only = os.environ.get("KORITH_SPEC_CACHE_ONLY", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._spec_require_distinct_draft = os.environ.get("KORITH_SPEC_REQUIRE_DISTINCT_DRAFT", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._spec_auto_cache_only = os.environ.get("KORITH_SPEC_AUTO_CACHE_ONLY", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._spec_max_draft_size_ratio = max(0.1, float(os.environ.get("KORITH_SPEC_MAX_DRAFT_SIZE_RATIO", "0.75")))
        self._spec_accept_ema = 0.0
        self._spec_roi_ema = 0.0
        self._spec_min_roi = max(0.0, float(os.environ.get("KORITH_SPEC_MIN_ROI", "1.0")))
        self._spec_samples = 0
        self._spec_disabled_until = 0.0
        self._spec_bad_max = max(1, int(os.environ.get("KORITH_SPEC_BAD_MAX", "5")))
        self._spec_cooldown_ms = max(1000, int(os.environ.get("KORITH_SPEC_COOLDOWN_MS", "60000")))
        self._decode_governor_enabled = os.environ.get("KORITH_DECODE_GOVERNOR_ENABLED", "1").strip().lower() in _TRUTHY
        self._decode_governor_probe_every = max(
            1,
            int(os.environ.get("KORITH_DECODE_GOVERNOR_PROBE_EVERY", "12") or 12),
        )
        self._spec_adaptive_k_enabled = os.environ.get("KORITH_SPEC_ADAPTIVE_K_ENABLED", "1").strip().lower() in _TRUTHY
        self._spec_k_min = max(1, int(os.environ.get("KORITH_SPEC_K_MIN", "2") or 2))
        self._spec_k_max = max(
            self._spec_k_min,
            int(os.environ.get("KORITH_SPEC_K_MAX", str(max(self._spec_k_min, int(self._spec_cfg.get("k", 6)) * 2))) or max(self._spec_k_min, int(self._spec_cfg.get("k", 6)) * 2)),
        )
        self._spec_adaptive_k_up_accept = max(
            0.0,
            min(1.0, float(os.environ.get("KORITH_SPEC_K_UP_ACCEPT", "0.9") or 0.9)),
        )
        self._spec_adaptive_k_down_accept = max(
            0.0,
            min(1.0, float(os.environ.get("KORITH_SPEC_K_DOWN_ACCEPT", "0.65") or 0.65)),
        )
        self._spec_adaptive_k_up_roi = max(0.0, float(os.environ.get("KORITH_SPEC_K_UP_ROI", "1.1") or 1.1))
        self._spec_adaptive_k_down_roi = max(0.0, float(os.environ.get("KORITH_SPEC_K_DOWN_ROI", "0.9") or 0.9))
        self._decode_governor_spec_state: Dict[str, Dict[str, Any]] = {}
        self._amf_seen_by_org: Dict[str, float] = {}
        self._amf_hits_by_org: Dict[str, float] = {}
        self._amf_cache_entries = 0
        self._amf_cache_bytes = 0
        self._amf_warm_ratio = 0.0
        self._amf_prewarm_complete = False
        self._amf_hit_rate = 0.0
        self._amf_ready = False
        self._tenant_saved_ms: Dict[str, float] = {}
        self._vllm_amf_infer_by_lane = os.environ.get("KORITH_VLLM_AMF_INFER_BY_LANE", "1").strip().lower() in _TRUTHY
        self._vllm_amf_miss_ema: Dict[str, float] = {}
        self._kernel_backend = resolve_kernel_backend()
        self._kernel_ctx = self._kernel_backend.context()
        self._kernel_session_disabled = False
        self._decode_calibration_mode = str(
            os.environ.get("KORITH_DECODE_CALIBRATION_MODE", "rolling") or "rolling"
        ).strip().lower()
        if self._decode_calibration_mode not in ("off", "rolling", "shadow", "strict"):
            self._decode_calibration_mode = "rolling"
        self._kernel_calibrate = os.environ.get("KORITH_KERNELS_CALIBRATE", "0").strip().lower() in _TRUTHY
        self._kernel_calibrate_every = max(1, int(os.environ.get("KORITH_KERNELS_CALIBRATE_EVERY", "1") or 1))
        if self._decode_calibration_mode in ("shadow", "strict"):
            self._kernel_calibrate = True
            self._kernel_calibrate_every = 1
        self._kernel_calibrate_counter = 0
        self._kernel_gpu_arch = str(os.environ.get("KORITH_GPU_ARCH", "unknown") or "unknown")
        self._kernel_decode_baseline_ema: Dict[str, float] = {}
        self._kernel_policy_enabled = os.environ.get("KORITH_KERNEL_POLICY_ENABLED", "1").strip().lower() in _TRUTHY
        self._kernel_policy_min_saved_ms = max(
            0.0,
            float(os.environ.get("KORITH_KERNEL_POLICY_MIN_SAVED_MS", "2.5") or 2.5),
        )
        self._kernel_policy_bad_streak_max = max(
            1,
            int(os.environ.get("KORITH_KERNEL_POLICY_BAD_STREAK_MAX", "3") or 3),
        )
        self._kernel_policy_cooldown_ms = max(
            1000,
            int(os.environ.get("KORITH_KERNEL_POLICY_COOLDOWN_MS", "60000") or 60000),
        )
        self._kernel_policy_probe_every = max(
            1,
            int(os.environ.get("KORITH_KERNEL_POLICY_PROBE_EVERY", "8") or 8),
        )
        self._kernel_policy_state: Dict[str, Dict[str, Any]] = {}
        self._vllm_oom_fallback_enabled = _env_truthy("KORITH_VLLM_OOM_FALLBACK_ENABLED", default=True)
        self._vllm_oom_fallback_all_lanes = _env_truthy("KORITH_VLLM_OOM_FALLBACK_ALL_LANES", default=False)
        self._vllm_priority_sched_enabled = _env_truthy("KORITH_VLLM_PRIORITY_SCHED", default=False)
        self._vllm_runtime_contract_enabled = _env_truthy("KORITH_VLLM_RUNTIME_CONTRACT_ENABLED", default=True)
        self._vllm_priority_lane_base = {
            "SPEC_HIT": int(os.environ.get("KORITH_VLLM_PRIORITY_SPEC_HIT", "0") or 0),
            "HIT": int(os.environ.get("KORITH_VLLM_PRIORITY_HIT", "1") or 1),
            "SPEC_MISS": int(os.environ.get("KORITH_VLLM_PRIORITY_SPEC_MISS", "2") or 2),
            "MISS": int(os.environ.get("KORITH_VLLM_PRIORITY_MISS", "3") or 3),
        }
        self._vllm_decode_budget_min_tokens = max(
            16,
            int(os.environ.get("KORITH_VLLM_DECODE_BUDGET_MIN_TOKENS", "96") or 96),
        )
        self._vllm_decode_budget_ratio_default = max(
            0.1,
            min(2.0, float(os.environ.get("KORITH_VLLM_DECODE_BUDGET_RATIO", "1.0") or 1.0)),
        )
        self._vllm_decode_budget_ratio_by_lane = {
            "SPEC_HIT": max(
                0.1,
                min(2.0, float(os.environ.get("KORITH_VLLM_DECODE_BUDGET_RATIO_SPEC_HIT", str(self._vllm_decode_budget_ratio_default)) or self._vllm_decode_budget_ratio_default)),
            ),
            "HIT": max(
                0.1,
                min(2.0, float(os.environ.get("KORITH_VLLM_DECODE_BUDGET_RATIO_HIT", str(self._vllm_decode_budget_ratio_default)) or self._vllm_decode_budget_ratio_default)),
            ),
            "SPEC_MISS": max(
                0.1,
                min(2.0, float(os.environ.get("KORITH_VLLM_DECODE_BUDGET_RATIO_SPEC_MISS", str(self._vllm_decode_budget_ratio_default)) or self._vllm_decode_budget_ratio_default)),
            ),
            "MISS": max(
                0.1,
                min(2.0, float(os.environ.get("KORITH_VLLM_DECODE_BUDGET_RATIO_MISS", str(self._vllm_decode_budget_ratio_default)) or self._vllm_decode_budget_ratio_default)),
            ),
        }
        self._worker_lane_role = str(os.environ.get("KORITH_WORKER_LANE_ROLE", "all") or "all").strip().lower()
        self._worker_lanes = _parse_worker_lanes(os.environ.get("KORITH_WORKER_LANES", ""))
        if not self._worker_lanes and self._worker_lane_role in ("hit", "hit_only", "hot"):
            self._worker_lanes = ["HIT", "SPEC_HIT"]
        elif not self._worker_lanes and self._worker_lane_role in ("miss", "miss_only", "cold"):
            self._worker_lanes = ["MISS", "SPEC_MISS"]
        cache_root_default = self.restore_store.db_path.parent / "snapshot_cache"
        self._snapshot_cache_root = Path(
            os.environ.get("KORITH_SNAPSHOT_CACHE_DIR", str(cache_root_default))
        ).expanduser().resolve()
        self._snapshot_vram_dir = Path(
            os.environ.get("KORITH_SNAPSHOT_VRAM_DIR", str(self._snapshot_cache_root / "vram"))
        ).expanduser().resolve()
        self._snapshot_ram_dir = Path(
            os.environ.get("KORITH_SNAPSHOT_RAM_DIR", str(self._snapshot_cache_root / "ram"))
        ).expanduser().resolve()
        self._snapshot_nvme_dir = Path(
            os.environ.get("KORITH_SNAPSHOT_NVME_DIR", str(self._snapshot_cache_root / "nvme"))
        ).expanduser().resolve()
        self._snapshot_vram_max_bytes = max(
            0,
            int(os.environ.get("KORITH_SNAPSHOT_TIER_VRAM_MAX_BYTES", str(256 * 1024 * 1024)) or (256 * 1024 * 1024)),
        )
        self._snapshot_ram_max_bytes = max(
            self._snapshot_vram_max_bytes,
            int(os.environ.get("KORITH_SNAPSHOT_TIER_RAM_MAX_BYTES", str(2 * 1024 * 1024 * 1024)) or (2 * 1024 * 1024 * 1024)),
        )
        self._snapshot_vram_cache_max_bytes = max(
            0,
            int(os.environ.get("KORITH_SNAPSHOT_VRAM_CACHE_MAX_BYTES", "0") or 0),
        )
        self._snapshot_ram_cache_max_bytes = max(
            0,
            int(os.environ.get("KORITH_SNAPSHOT_RAM_CACHE_MAX_BYTES", "0") or 0),
        )
        self._snapshot_nvme_cache_max_bytes = max(
            0,
            int(os.environ.get("KORITH_SNAPSHOT_NVME_CACHE_MAX_BYTES", "0") or 0),
        )
        self._snapshot_promote_on_restore = os.environ.get("KORITH_SNAPSHOT_PROMOTE_ON_RESTORE", "1").strip().lower() in _TRUTHY
        self._decode_cache_enabled = os.environ.get("KORITH_DECODE_CACHE_ENABLED", "0").strip().lower() in _TRUTHY
        self._decode_cache_require_deterministic = (
            os.environ.get("KORITH_DECODE_CACHE_REQUIRE_DETERMINISTIC", "1").strip().lower() in _TRUTHY
        )
        self._decode_cache_store: Optional[DecodeCacheStore] = None
        if self._decode_cache_enabled:
            decode_cache_default = self.restore_store.db_path.parent / "decode_cache.sqlite"
            decode_cache_db = Path(
                os.environ.get("KORITH_DECODE_CACHE_DB_PATH", str(decode_cache_default))
            ).expanduser().resolve()
            try:
                self._decode_cache_store = DecodeCacheStore(decode_cache_db)
            except Exception:
                self._decode_cache_store = None

    def _queue_front_shape_key(self, q: queue.Queue) -> str:
        try:
            with q.mutex:
                front = q.queue[0] if q.queue else None
        except Exception:
            front = None
        payload = getattr(front, "payload", None)
        if not isinstance(payload, dict):
            return ""
        job = payload.get("job", {})
        if not isinstance(job, dict):
            return ""
        routing = job.get("routing_decision", {})
        if isinstance(routing, dict):
            shape_key = str(routing.get("shape_key", "") or "").strip()
            if shape_key:
                return shape_key
        return str(self._kernel_policy_shape_key(job) or "").strip()

    def _queue_front_replay_score(self, lane: str, q: queue.Queue) -> float:
        score = 0.0
        lane_up = str(lane).upper()
        if lane_up in ("HIT", "SPEC_HIT"):
            score += 0.25
        try:
            with q.mutex:
                front = q.queue[0] if q.queue else None
        except Exception:
            front = None
        payload = getattr(front, "payload", None)
        if not isinstance(payload, dict):
            return float(score)
        job = payload.get("job", {})
        if not isinstance(job, dict):
            return float(score)
        policy = job.get("policy", {})
        if isinstance(policy, dict):
            contract = policy.get("_vllm_contract", {})
            if isinstance(contract, dict):
                replay_state = str(contract.get("replay_state", "") or "").strip().lower()
                replay_local = bool(contract.get("replay_local", False))
                if replay_state == "restore":
                    score += 0.3
                elif replay_state == "hit":
                    score += 0.2
                if replay_local:
                    score += 0.2
        routing = job.get("routing_decision", {})
        if isinstance(routing, dict):
            if bool(routing.get("replay_local", False)):
                score += 0.2
            transfer_requested = bool(routing.get("transfer_requested", False))
            if transfer_requested:
                score -= 0.15
            tier = str(routing.get("snapshot_tier", "") or "").strip().lower()
            if tier in ("vram", "ram", "nvme", "local"):
                score += 0.1
        return float(score)

    def _reset_queue_budget(self, queue_map: Optional[Dict[str, queue.Queue]] = None) -> None:
        base_budget = {
            "SPEC_HIT": self._spec_priority,
            "HIT": self._hit_priority,
            "SPEC_MISS": max(1, self._spec_priority // 2),
            "MISS": self._miss_priority,
        }
        if not self._decode_adaptive_budget_enabled or not isinstance(queue_map, dict):
            self._queue_budget.update(base_budget)
            return

        active_lanes = []
        for lane, q in queue_map.items():
            if lane not in base_budget:
                continue
            try:
                depth = int(q.qsize() or 0)
            except Exception:
                depth = 0
            if depth > 0:
                active_lanes.append((lane, depth))

        if not active_lanes:
            self._queue_budget.update(base_budget)
            return

        score_map: Dict[str, float] = {}
        for lane, depth in active_lanes:
            lane_decode_ema = float(self._lane_decode_ema_ms.get(lane, 1.0) or 1.0)
            effective_decode_ema = lane_decode_ema
            if bool(getattr(self, "_decode_shape_aware_scheduler", True)):
                shape_key = self._queue_front_shape_key(queue_map[lane])
                if shape_key:
                    shape_decode_ema = float(self._shape_decode_ema_ms.get(shape_key, lane_decode_ema) or lane_decode_ema)
                    effective_decode_ema = min(lane_decode_ema, shape_decode_ema)
            depth_factor = 1.0 + (float(getattr(self, "_decode_budget_depth_weight", 0.35) or 0.35) * math.log1p(max(0, depth)))
            replay_factor = 1.0 + max(0.0, self._queue_front_replay_score(lane, queue_map[lane]))
            score = ((self._lane_priority_weight(lane) * depth_factor) * replay_factor) / max(1.0, effective_decode_ema)
            score_map[lane] = max(0.001, float(score))

        total_budget = max(len(active_lanes), sum(base_budget.get(lane, 0) for lane, _ in active_lanes))
        budgets = {lane: 0 for lane in base_budget}
        remaining = max(0, int(total_budget) - len(active_lanes))
        for lane, _ in active_lanes:
            budgets[lane] = 1

        score_sum = sum(score_map.values())
        fractions = []
        if remaining > 0 and score_sum > 0.0:
            for lane, _ in active_lanes:
                exact = (remaining * score_map[lane]) / score_sum
                whole = int(math.floor(exact))
                budgets[lane] += whole
                fractions.append((exact - whole, lane))
            granted = sum(budgets[lane] for lane, _ in active_lanes)
            rem = max(0, int(total_budget) - granted)
            for _, lane in sorted(fractions, reverse=True):
                if rem <= 0:
                    break
                budgets[lane] += 1
                rem -= 1

        self._queue_budget.update(budgets)

    def _lane_priority_weight(self, lane: str) -> float:
        lane_up = str(lane).upper()
        if lane_up == "SPEC_HIT":
            return float(self._spec_priority)
        if lane_up == "HIT":
            return float(self._hit_priority)
        if lane_up == "SPEC_MISS":
            return float(max(1, self._spec_priority // 2))
        return float(self._miss_priority)

    def _select_lane_to_run(self, queue_order: tuple[str, ...], queue_map: Dict[str, queue.Queue]) -> Optional[str]:
        queue_idx = {lane: idx for idx, lane in enumerate(queue_order)}
        candidates = []
        for lane in queue_order:
            q = queue_map[lane]
            if q.empty():
                continue
            if self._queue_budget[lane] <= 0:
                continue
            candidates.append(lane)
        if not candidates:
            return None
        if not self._decode_scheduler_enabled:
            return candidates[0]

        ranked = []
        for lane in candidates:
            budget = float(max(1, int(self._queue_budget.get(lane, 0) or 0)))
            decode_ema = float(self._lane_decode_ema_ms.get(lane, 1.0) or 1.0)
            if bool(getattr(self, "_decode_shape_aware_scheduler", True)):
                shape_key = self._queue_front_shape_key(queue_map[lane])
                if shape_key:
                    shape_decode_ema = float(self._shape_decode_ema_ms.get(shape_key, decode_ema) or decode_ema)
                    decode_ema = min(decode_ema, shape_decode_ema)
            try:
                depth = int(queue_map[lane].qsize() or 0)
            except Exception:
                depth = 0
            depth_bonus = 1.0 + (0.1 * math.log1p(max(0, depth)))
            replay_bonus = 1.0 + max(0.0, self._queue_front_replay_score(lane, queue_map[lane]))
            score = (((self._lane_priority_weight(lane) * budget) * depth_bonus) * replay_bonus) / max(1.0, decode_ema)
            ranked.append((score, -queue_idx.get(lane, 0), lane))
        ranked.sort(reverse=True)
        return ranked[0][2]

    def _result_indicates_local_oom(self, result: Dict[str, Any], artifacts: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        try:
            if int(result.get("exit_code", 0) or 0) == 137:
                return True
        except Exception:
            pass

        candidates: list[str] = []
        for key in ("error", "stderr", "output_text"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                candidates.append(val)
        engine_errors = result.get("engine_errors", [])
        if isinstance(engine_errors, list):
            for item in engine_errors:
                if isinstance(item, str) and item.strip():
                    candidates.append(item)
        for chunk in candidates:
            if _looks_like_oom(chunk):
                return True

        log_path = artifacts.get("log")
        if log_path:
            try:
                text = Path(str(log_path)).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            if text:
                tail = text[-32768:]
                if _looks_like_oom(tail):
                    return True
        return False

    def _resolve_vllm_fallback_target(self, job: Dict[str, Any], lane: str) -> Optional[Dict[str, Any]]:
        lane_up = str(lane).upper()
        endpoint = str(os.environ.get("KORITH_VLLM_ENDPOINT", "")).strip()
        if lane_up in ("MISS", "SPEC_MISS"):
            endpoint = str(os.environ.get("KORITH_VLLM_MISS_ENDPOINT", "")).strip() or endpoint
        elif lane_up in ("HIT", "SPEC_HIT"):
            endpoint = str(os.environ.get("KORITH_VLLM_HIT_ENDPOINT", "")).strip() or endpoint
        if lane_up.startswith("SPEC_"):
            endpoint = str(os.environ.get("KORITH_VLLM_SPEC_ENDPOINT", "")).strip() or endpoint
        if not endpoint:
            return None
        model_from_env = str(os.environ.get("KORITH_VLLM_MODEL_ID", "")).strip()
        if lane_up in ("MISS", "SPEC_MISS"):
            model_from_env = str(os.environ.get("KORITH_VLLM_MISS_MODEL_ID", "")).strip() or model_from_env
        elif lane_up in ("HIT", "SPEC_HIT"):
            model_from_env = str(os.environ.get("KORITH_VLLM_HIT_MODEL_ID", "")).strip() or model_from_env
        if lane_up.startswith("SPEC_"):
            model_from_env = str(os.environ.get("KORITH_VLLM_SPEC_MODEL_ID", "")).strip() or model_from_env
        model_exec = job.get("execution_model", {}) if isinstance(job.get("execution_model"), dict) else {}
        model_req = job.get("model", {}) if isinstance(job.get("model"), dict) else {}
        model_id = model_from_env or str(model_exec.get("model_id", "")).strip() or str(model_req.get("model_id", "")).strip()
        if not model_id:
            return None
        backend_id = str(os.environ.get("KORITH_VLLM_BACKEND_ID", "vllm")).strip() or "vllm"
        return {
            "backend_id": backend_id,
            "model": {"model_id": model_id, "endpoint": endpoint},
        }

    def _try_vllm_oom_fallback(
        self,
        *,
        job: Dict[str, Any],
        lane: str,
        result: Dict[str, Any],
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict[str, Any],
        policy: Dict[str, Any],
        artifacts: Dict[str, Any],
        mf_snapshot_in: Optional[str],
        events_path: Path,
        request_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._vllm_oom_fallback_enabled:
            return None
        lane_up = str(lane).upper()
        if (lane_up not in ("MISS", "SPEC_MISS")) and not self._vllm_oom_fallback_all_lanes:
            return None
        current_backend = str(job.get("execution_backend_id", "") or job.get("backend_id", "")).strip().lower()
        if not _backend_is_local_runtime(current_backend):
            return None
        if not isinstance(result, dict):
            return None
        try:
            if int(result.get("exit_code", 0) or 0) == 0:
                return None
        except Exception:
            return None
        if not self._result_indicates_local_oom(result, artifacts):
            return None

        fallback_target = self._resolve_vllm_fallback_target(job, lane)
        if not isinstance(fallback_target, dict):
            return None
        self._append_event(
            events_path,
            "BACKEND_FALLBACK",
            {
                "job_id": job.get("job_id", ""),
                "request_id": request_id,
                "from_backend_id": current_backend,
                "to_backend_id": str(fallback_target.get("backend_id", "vllm")),
                "reason": "local_oom",
            },
        )
        try:
            fallback_adapter = self.adapter_registry.get_adapter(
                {
                    "backend_id": str(fallback_target["backend_id"]),
                    "model": dict(fallback_target["model"]),
                }
            )
        except Exception:
            return None

        fallback_policy = dict(policy)
        fallback_policy["allow_spec"] = False
        fallback_caps = fallback_adapter.get_capabilities()
        if not (fallback_caps.kv_replay and fallback_caps.deterministic_seeding):
            fallback_policy["allow_amf_reuse"] = False
        fallback_result = fallback_adapter.run_baseline(
            prompt=prompt,
            max_tokens=max_tokens,
            deterministic_cfg=deterministic_cfg,
            policy=fallback_policy,
            artifacts={k: str(v) for k, v in artifacts.items()},
            mf_snapshot_in=mf_snapshot_in,
        )
        if not isinstance(fallback_result, dict):
            return None

        job["execution_backend_id"] = str(fallback_target["backend_id"])
        job["execution_model"] = dict(fallback_target["model"])
        job["execution_backend_version"] = str(getattr(fallback_adapter, "backend_version", "v1"))
        try:
            job["execution_fingerprint"] = dict(fallback_adapter.get_fingerprint() or {})
        except Exception:
            pass
        routing_decision = job.get("routing_decision", {})
        if isinstance(routing_decision, dict):
            routing_decision["execution_backend_id"] = str(fallback_target["backend_id"])
            routing_decision["execution_reason"] = "worker_oom_fallback"
            routing_decision["fallback_from_backend_id"] = current_backend
            routing_decision["fallback_reason"] = "local_oom"
        self._append_event(
            events_path,
            "BACKEND_FALLBACK_SUCCESS",
            {
                "job_id": job.get("job_id", ""),
                "request_id": request_id,
                "from_backend_id": current_backend,
                "to_backend_id": str(fallback_target.get("backend_id", "vllm")),
            },
        )
        GLOBAL_METRICS.inc("backend_fallback_total", 1.0, labels={"org_id": str(job.get("org_id", "default"))})
        return {"adapter": fallback_adapter, "caps": fallback_caps, "result": fallback_result}

    def start(self) -> None:
        self.registry.register(
            self.worker_id,
            host=self.host,
            gpu_id=self.gpu_id,
            node_id=self.node_id,
            capabilities=self._worker_capabilities(),
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _worker_capabilities(self) -> Dict[str, Any]:
        worker_caps: Dict[str, Any] = {
            "gpu_id": self.gpu_id,
            "node_id": self.node_id,
            "lane_role": self._worker_lane_role,
            "amf_ready": bool(self._amf_ready),
            "amf_cache_entries": int(self._amf_cache_entries),
            "amf_cache_bytes": int(self._amf_cache_bytes),
            "amf_hit_rate": float(self._amf_hit_rate),
            "amf_warm_ratio": float(self._amf_warm_ratio),
            "amf_prewarm_complete": bool(self._amf_prewarm_complete),
        }
        if self._worker_lanes:
            worker_caps["lanes"] = list(self._worker_lanes)
        return worker_caps

    def _coordinator_enabled(self) -> bool:
        client = self._amf_coordinator_client
        return bool(client is not None and getattr(client, "enabled", False))

    def _coordinator_register_entry(
        self,
        *,
        tenant_id: str,
        amf_lookup_hash: str,
        worker_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        if not self._coordinator_enabled():
            return
        prefix_hash = str(amf_lookup_hash or "").strip().lower()
        if not prefix_hash:
            return
        tenant = str(tenant_id or "default")
        cache_key = f"{tenant}:{prefix_hash}"
        row = {
            "tenant_id": tenant,
            "hash": prefix_hash,
            "worker_id": str(worker_id or self.worker_id),
            "metadata": dict(metadata or {}),
        }
        self._amf_coordinator_entries[cache_key] = row
        try:
            self._amf_coordinator_client.register(
                prefix_hash=prefix_hash,
                tenant_id=tenant,
                node_id=self.node_id,
                worker_id=self.worker_id,
                metadata=row["metadata"],
            )
        except Exception:
            # Fail open: request execution must continue without coordinator.
            return

    def _coordinator_heartbeat(self, force: bool = False) -> None:
        if not self._coordinator_enabled():
            return
        now = time.time()
        if not force and (now - self._amf_coordinator_last_heartbeat) < self._amf_coordinator_heartbeat_s:
            return
        entries = list(self._amf_coordinator_entries.values())
        try:
            self._amf_coordinator_client.heartbeat(
                node_id=self.node_id,
                entries=entries,
            )
            self._amf_coordinator_last_heartbeat = now
        except Exception:
            return

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._coordinator_heartbeat()
            item = None
            queue_order = ("SPEC_HIT", "HIT", "SPEC_MISS", "MISS")
            queue_map = {
                "SPEC_HIT": self._q_spec_hit,
                "HIT": self._q_hit,
                "SPEC_MISS": self._q_spec_miss,
                "MISS": self._q_miss,
            }
            selected_lane = self._select_lane_to_run(queue_order, queue_map)
            if selected_lane is not None:
                item = queue_map[selected_lane].get()
                self._queue_budget[selected_lane] -= 1
            if item is None:
                has_pending = any(not q.empty() for q in queue_map.values())
                if has_pending:
                    # Prevent starvation when one lane's budget reaches zero while
                    # other lanes remain non-empty but unavailable.
                    self._reset_queue_budget(queue_map)
                    continue
                if all(v <= 0 for v in self._queue_budget.values()):
                    self._reset_queue_budget(queue_map)
                fetched = self.queue.dequeue(self.worker_id, timeout_s=0.5)
                if fetched:
                    lane = str(fetched.payload.get("lane", "MISS")).upper()
                    if lane == "SPEC_HIT":
                        self._q_spec_hit.put(fetched)
                    elif lane == "HIT":
                        self._q_hit.put(fetched)
                    elif lane == "SPEC_MISS":
                        self._q_spec_miss.put(fetched)
                    else:
                        self._q_miss.put(fetched)
                    continue
                self.registry.heartbeat(self.worker_id, self._inflight, capabilities=self._worker_capabilities())
                continue
            self._inflight += 1
            self.registry.heartbeat(self.worker_id, self._inflight, capabilities=self._worker_capabilities())
            try:
                self._process(item.payload["job"], item.payload.get("lane", "MISS"), item.enqueued_at, item.payload)
                self.queue.ack(item.job_id)
            except Exception:
                self.queue.retry(item.job_id, delay_s=1.0)
                GLOBAL_METRICS.inc("replay_failures_total", 1.0)
            self._inflight -= 1
            self.registry.heartbeat(self.worker_id, self._inflight, capabilities=self._worker_capabilities())

    def enable_orchestrator(self, orchestrator: Any) -> None:
        """Wire the AI reasoning orchestrator into the request path. Fail-closed: any
        error from the orchestrator falls back to existing AMF behavior transparently."""
        self._orchestrator = orchestrator

    def _process(self, job: Dict[str, Any], lane: str, enqueued_at: float, payload: Optional[Dict[str, Any]] = None) -> None:
        job_id = job["job_id"]
        org_id = job.get("org_id", "default")
        tenant_id = str(job.get("tenant_id", org_id) or org_id)
        request_id = job.get("request_id", "")
        session_id = job.get("session_id", "")
        artifacts = self.artifacts.init_job(job_id, org_id=org_id)
        events_path = artifacts["events"]
        platform_log = artifacts["job_dir"] / "platform.log"

        queue_latency_ms = max(0.0, (time.time() - enqueued_at) * 1000.0)
        self._append_event(events_path, "WORKER_START", {
            "job_id": job_id,
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "lane": lane,
            "org_id": org_id,
            "tenant_id": tenant_id,
            "request_id": request_id,
            "queue_latency_ms": queue_latency_ms,
        })
        emit_log(
            "worker",
            "WORKER_START",
            {
                "job_id": job_id,
                "run_id": "",
                "worker_id": self.worker_id,
                "session_id": session_id,
                "org_id": org_id,
                "tenant_id": tenant_id,
                "request_id": request_id,
                "latency_ms": queue_latency_ms,
            },
            artifact_log=platform_log,
        )
        self.ledger.update_status(job_id, "RUNNING")

        # AI reasoning orchestrator — fail-closed, zero impact on existing path
        _orch_outcome = None
        if self._orchestrator is not None:
            try:
                _orch_outcome = self._orchestrator.on_request(
                    prefix_hash=str(job.get("prefix_hash", "") or ""),
                    tenant_id=tenant_id,
                    node_id=str(self.gpu_id),
                    worker_id=self.worker_id,
                )
            except Exception:
                _orch_outcome = None

        adapter_job = dict(job)
        if job.get("execution_backend_id"):
            adapter_job["backend_id"] = job.get("execution_backend_id")
        if isinstance(job.get("execution_model"), dict) and job.get("execution_model"):
            adapter_job["model"] = job.get("execution_model")
        adapter = self.adapter_registry.get_adapter(adapter_job)
        caps = adapter.get_capabilities()
        deterministic_cfg = job.get("deterministic_cfg", {})
        max_tokens = int(deterministic_cfg.get("max_tokens", 256) or 256)
        prompt = job.get("prompt_rendered", "")
        policy = dict(job.get("policy", {}))
        policy["_lane"] = str(lane).upper()
        policy["_execution_backend_id"] = str(job.get("execution_backend_id", "") or job.get("backend_id", ""))
        policy["_tenant_id"] = tenant_id
        job_spec_cfg = dict(job.get("spec_cfg", {}) or {})
        if payload and bool(payload.get("force_miss", False)):
            policy["allow_amf_reuse"] = False
            self._append_event(events_path, "AMF_BLOCK", {"job_id": job_id, "reason": "forced_miss", "request_id": request_id})
        if not (caps.kv_replay and caps.deterministic_seeding):
            policy["allow_amf_reuse"] = False
        spec_requested = bool(policy.get("allow_spec", False))
        lane_is_spec = str(lane).upper() in ("SPEC_HIT", "SPEC_MISS")
        if lane_is_spec:
            spec_requested = True
        spec_ignore_governance = os.environ.get("KORITH_SPEC_IGNORE_GOVERNANCE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        fingerprint_hash = job.get("fingerprint_hash", "")
        gov = self.ledger.get_replay_governance(fingerprint_hash) if fingerprint_hash else None
        now = time.time()
        if gov and int(gov.get("replay_disabled", 0)):
            policy["allow_amf_reuse"] = False
            self._append_event(events_path, "AMF_BLOCK", {"job_id": job_id, "reason": "governance_disabled", "request_id": request_id})
        elif gov and float(gov.get("cooldown_until", 0.0) or 0.0) > now:
            policy["allow_amf_reuse"] = False
            self._append_event(events_path, "AMF_BLOCK", {"job_id": job_id, "reason": "negative_roi_cooldown", "request_id": request_id})

        spec_gov = self.ledger.get_spec_governance(fingerprint_hash) if fingerprint_hash else None
        if spec_requested and spec_gov and not spec_ignore_governance:
            spec_disabled = int(spec_gov.get("spec_disabled", 0) or 0) != 0
            cooldown_until = float(spec_gov.get("cooldown_until", 0.0) or 0.0)
            if spec_disabled:
                spec_requested = False
                self._append_event(events_path, "SPEC_DISABLE", {"job_id": job_id, "request_id": request_id, "reason": "governance_disabled"})
            elif cooldown_until > now:
                spec_requested = False
                self._append_event(events_path, "SPEC_DISABLE", {"job_id": job_id, "request_id": request_id, "reason": "governance_cooldown"})

        mf_snapshot_in = self._resolve_snapshot_input_path(
            fingerprint_hash=fingerprint_hash,
            org_id=org_id,
            candidate_path=self.restore_store.get(fingerprint_hash),
        )
        if mf_snapshot_in and self._snapshot_promote_on_restore:
            mf_snapshot_in = self._maybe_promote_snapshot_input(
                fingerprint_hash=fingerprint_hash,
                org_id=org_id,
                snapshot_path=mf_snapshot_in,
                events_path=events_path,
                job_id=job_id,
                request_id=request_id,
            )
        transfer_req = (payload or {}).get("snapshot_transfer", {}) if payload else {}
        if not mf_snapshot_in and transfer_req and transfer_req.get("requested") and self.node_registry is not None:
            remote_node_id = str(transfer_req.get("from_node_id") or "")
            remote = self.node_registry.get_node(remote_node_id) if remote_node_id else None
            if remote:
                transfer_started = time.time()
                try:
                    blob, meta = fetch_snapshot_bytes(
                        node_host=str(remote.get("host", "127.0.0.1")),
                        node_port=int(remote.get("router_port", 0) or 0),
                        fingerprint_hash=fingerprint_hash,
                        org_id=org_id,
                    )
                    local_snap, bytes_written, snapshot_tier = self._cache_snapshot_from_blob(
                        fingerprint_hash=fingerprint_hash,
                        blob=blob,
                    )
                    mf_snapshot_in = str(local_snap)
                    self.restore_store.set(fingerprint_hash, mf_snapshot_in)
                    if self.snapshot_index is not None:
                        self.snapshot_index.upsert_location(
                            fingerprint_hash=fingerprint_hash,
                            snapshot_id=job_id,
                            org_id=org_id,
                            node_id=self.node_id,
                            worker_id=self.worker_id,
                            snapshot_path=str(local_snap),
                            size_bytes=bytes_written,
                            created_at=utc_now(),
                            last_used_at=utc_now(),
                        )
                    self._append_event(
                        events_path,
                        "SNAPSHOT_TRANSFER",
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "from_node_id": remote_node_id,
                            "bytes": bytes_written,
                            "snapshot_tier": snapshot_tier,
                            "duration_ms": (time.time() - transfer_started) * 1000.0,
                            "meta": meta,
                        },
                    )
                    emit_log(
                        "worker",
                        "SNAPSHOT_TRANSFER",
                        {
                            "job_id": job_id,
                            "run_id": "",
                            "worker_id": self.worker_id,
                            "session_id": session_id,
                            "org_id": org_id,
                            "request_id": request_id,
                            "latency_ms": (time.time() - transfer_started) * 1000.0,
                            "from_node_id": remote_node_id,
                            "bytes": bytes_written,
                        },
                        artifact_log=platform_log,
                    )
                    GLOBAL_METRICS.inc("snapshot_transfer_total", 1.0, labels={"org_id": org_id})
                    GLOBAL_METRICS.inc("snapshot_transfer_bytes_total", float(bytes_written), labels={"org_id": org_id})
                except Exception as exc:
                    self._append_event(
                        events_path,
                        "SNAPSHOT_TRANSFER_FAILED",
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "from_node_id": remote_node_id,
                            "error": str(exc),
                        },
                    )
                    GLOBAL_METRICS.inc("snapshot_transfer_failures_total", 1.0, labels={"org_id": org_id})
                    emit_log(
                        "worker",
                        "SNAPSHOT_TRANSFER_FAILED",
                        {
                            "job_id": job_id,
                            "run_id": "",
                            "worker_id": self.worker_id,
                            "session_id": session_id,
                            "org_id": org_id,
                            "request_id": request_id,
                            "latency_ms": 0.0,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "replay_related": True,
                        },
                        artifact_log=platform_log,
                    )
                    policy["allow_amf_reuse"] = False
        if mf_snapshot_in:
            self._append_event(events_path, "MF_RESTORE", {"job_id": job_id, "snapshot_path": mf_snapshot_in, "request_id": request_id})

        if mf_snapshot_in and caps.kv_replay and caps.deterministic_seeding:
            if self._is_kv_transfer_snapshot(mf_snapshot_in):
                self._append_event(
                    events_path,
                    "MF_RESTORE_NATIVE",
                    {
                        "job_id": job_id,
                        "request_id": request_id,
                        "snapshot_path": mf_snapshot_in,
                        "format": "kv_transfer_params",
                    },
                )
            else:
                model_hash = str(job.get("fingerprint", {}).get("model_hash", ""))
                tokenizer_hash = str(job.get("fingerprint", {}).get("tokenizer_hash", ""))
                kv_layout = str(self._engine_cfg.get("kv_layout_version", "v1"))
                n_ctx = int(job.get("deterministic_cfg", {}).get("n_ctx", 0) or 0)
                ok, reason, _ = validate_snapshot_metadata(
                    mf_snapshot_in,
                    fingerprint_hash=fingerprint_hash,
                    model_hash=model_hash,
                    tokenizer_hash=tokenizer_hash,
                    kv_layout_version=kv_layout,
                    n_ctx=n_ctx,
                )
                if not ok:
                    policy["allow_amf_reuse"] = False
                    mf_snapshot_in = None
                    self._append_event(
                        events_path,
                        "CORRUPTION_DETECTED",
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "reason": reason,
                            "replay_related": True,
                        },
                    )
                    GLOBAL_METRICS.inc("corruption_detected_events", 1.0, labels={"org_id": org_id})

        started_at = utc_now()
        run_mode = "baseline"
        kernel_requested = bool(self._kernel_ctx.enabled) and not self._kernel_session_disabled
        kernel_available = bool(getattr(self._kernel_ctx, "available", False))
        kernel_executed = False
        kernel_verify = bool(self._kernel_ctx.verify)
        kernel_fallback = False
        kernel_verify_ok = True
        kernel_ms_saved = 0.0
        no_decode_cache_hit = False
        shadow_decode_baseline_ms = 0.0
        shadow_tokens_out = 0
        decode_cache_hit = False
        decode_cache_checked = False
        decode_cache_saved_est_ms = 0.0
        kernel_policy_shape_key = self._kernel_policy_shape_key(job)
        spec_policy_shape_key = kernel_policy_shape_key
        routing_decision = job.get("routing_decision", {})
        request_shape_key = ""
        if isinstance(routing_decision, dict):
            request_shape_key = str(routing_decision.get("shape_key", "") or "").strip()
        if not request_shape_key:
            request_shape_key = str(kernel_policy_shape_key or "").strip()
        spec_policy_reason = "policy_not_checked"
        kernel_policy_reason = "kernel_disabled"
        request_engine_cfg = dict(self._engine_cfg)
        lane_up = str(lane).upper()
        vllm_priority = self._vllm_priority_for_request(lane=lane_up, shape_key=request_shape_key)
        if vllm_priority is not None:
            policy["_vllm_priority"] = int(vllm_priority)
        policy["_shape_key"] = str(request_shape_key)
        vllm_contract = self._build_vllm_runtime_contract(
            lane=lane_up,
            shape_key=request_shape_key,
            max_tokens=max_tokens,
            queue_latency_ms=queue_latency_ms,
            has_snapshot=bool(mf_snapshot_in),
            vllm_priority=vllm_priority,
            routing_decision=routing_decision if isinstance(routing_decision, dict) else {},
            spec_requested=bool(spec_requested),
            spec_cfg=job_spec_cfg,
        )
        if vllm_contract:
            policy["_vllm_contract"] = dict(vllm_contract)

        def _lane_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in _TRUTHY

        if lane_up in ("HIT", "SPEC_HIT"):
            kernel_requested = kernel_requested and _lane_bool("KORITH_KERNELS_HIT", True)
        if lane_up in ("MISS", "SPEC_MISS"):
            kernel_requested = kernel_requested and _lane_bool("KORITH_KERNELS_MISS", True)
        if lane_up.startswith("SPEC_"):
            kernel_requested = kernel_requested and _lane_bool("KORITH_KERNELS_SPEC", True)

        kernel_backend_override = ""
        if lane_up in ("HIT", "SPEC_HIT"):
            kernel_backend_override = str(os.environ.get("KORITH_KERNEL_BACKEND_HIT", "")).strip()
        elif lane_up in ("MISS", "SPEC_MISS"):
            kernel_backend_override = str(os.environ.get("KORITH_KERNEL_BACKEND_MISS", "")).strip()
        if lane_up.startswith("SPEC_"):
            kernel_backend_override = str(os.environ.get("KORITH_KERNEL_BACKEND_SPEC", "")).strip() or kernel_backend_override
        if kernel_backend_override:
            request_engine_cfg["kernel_backend"] = kernel_backend_override
        if kernel_requested:
            kernel_allowed, kernel_policy_reason = self._kernel_policy_allow(shape_key=kernel_policy_shape_key, lane=lane)
            if kernel_allowed:
                request_engine_cfg["kernels_enabled"] = "1"
                request_engine_cfg["kernel_backend"] = str(
                    request_engine_cfg.get("kernel_backend", self._kernel_ctx.backend)
                )
                request_engine_cfg["kernel_verify"] = "1" if bool(self._kernel_ctx.verify) else "0"
            else:
                kernel_requested = False
                kernel_available = False
                kernel_verify = False
                request_engine_cfg["kernels_enabled"] = "0"
                request_engine_cfg["kernel_backend"] = "none"
                request_engine_cfg["kernel_verify"] = "0"
                self._append_event(
                    events_path,
                    "KERNEL_POLICY_SKIP",
                    {"job_id": job_id, "request_id": request_id, "reason": kernel_policy_reason},
                )
        else:
            request_engine_cfg["kernels_enabled"] = "0"
            request_engine_cfg["kernel_backend"] = "none"
            request_engine_cfg["kernel_verify"] = "0"
        try:
            if policy.get("allow_amf_reuse", True) and caps.kv_replay and caps.deterministic_seeding:
                self._append_event(events_path, "AMF_LOOKUP", {"job_id": job_id, "request_id": request_id})
            else:
                reason = "policy_blocked" if not policy.get("allow_amf_reuse", True) else "unavailable"
                self._append_event(events_path, "AMF_BLOCK", {"job_id": job_id, "reason": reason, "request_id": request_id})
            result = None
            cache_key = self._decode_cache_key(job)
            if cache_key and self._decode_cache_store is not None and self._decode_cache_eligible(job):
                decode_cache_checked = True
                cache_t0 = time.time()
                cached = self._decode_cache_store.get(**cache_key)
                if isinstance(cached, dict):
                    decode_cache_hit = True
                    no_decode_cache_hit = True
                    elapsed_ms = max(0.0, (time.time() - cache_t0) * 1000.0)
                    cached_output = str(cached.get("output_text", "") or "")
                    cached_tokens = max(0, int(cached.get("tokens_out", 0) or 0))
                    decode_cache_saved_est_ms = max(0.0, float(cached.get("decode_ms", 0.0) or 0.0))
                    Path(artifacts["output"]).write_text(cached_output, encoding="utf-8")
                    with Path(artifacts["log"]).open("a", encoding="utf-8") as f:
                        f.write("[DECODE_CACHE] hit=1\n")
                    self._append_event(
                        events_path,
                        "DECODE_CACHE_HIT",
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "decode_ms_saved_est": decode_cache_saved_est_ms,
                        },
                    )
                    result = {
                        "exit_code": 0,
                        "output_text": cached_output,
                        "total_ms": elapsed_ms,
                        "engine_metrics": {
                            "amf": {"supported": False, "decision": "unavailable"},
                            "mf": {"supported": False},
                            "perf": {
                                "tokens_out": cached_tokens,
                                "total_ms": elapsed_ms,
                                "prefill_ms": 0.0,
                                "decode_ms": 0.0,
                                "avg_tps": 0.0,
                            },
                            "spec": {"enabled": False},
                            "kernels": {"enabled": False, "kernels_applied": False, "ms_saved": 0.0},
                            "health": {
                                "decode_cache": {
                                    "hit": True,
                                    "saved_decode_ms_est": decode_cache_saved_est_ms,
                                }
                            },
                        },
                    }
                else:
                    self._append_event(events_path, "DECODE_CACHE_MISS", {"job_id": job_id, "request_id": request_id})
            spec_capable = bool(getattr(caps, "draft_supported", False) and (getattr(caps, "verify_tokens", False) or getattr(caps, "logits_access", False)))
            spec_block_reason = None
            spec_disable_reason = None
            spec_draft_init_reason = None
            effective_spec_cfg = dict(self._spec_cfg)
            if "k" in job_spec_cfg:
                effective_spec_cfg["k"] = max(1, int(job_spec_cfg.get("k") or effective_spec_cfg["k"]))
            if "min_accept" in job_spec_cfg:
                effective_spec_cfg["min_accept"] = max(0.0, min(1.0, float(job_spec_cfg.get("min_accept") or effective_spec_cfg["min_accept"])))
            if "disable_after_n" in job_spec_cfg:
                effective_spec_cfg["disable_after_n"] = max(1, int(job_spec_cfg.get("disable_after_n") or effective_spec_cfg["disable_after_n"]))
            if "cache_only" in job_spec_cfg:
                effective_spec_cfg["cache_only"] = bool(job_spec_cfg.get("cache_only"))
            if "draft_model" in job_spec_cfg:
                effective_spec_cfg["draft_model"] = job_spec_cfg.get("draft_model")
            if spec_requested:
                spec_allowed, spec_policy_reason = self._decode_governor_spec_allow(
                    shape_key=spec_policy_shape_key,
                    lane=lane,
                )
                effective_spec_cfg["k"] = self._decode_governor_spec_k(
                    shape_key=spec_policy_shape_key,
                    lane=lane,
                    requested_k=max(1, int(effective_spec_cfg.get("k", self._spec_cfg.get("k", 1)) or 1)),
                    max_tokens=max_tokens,
                )
                if not spec_allowed:
                    spec_block_reason = spec_policy_reason
                if not self._spec_enabled:
                    spec_block_reason = "disabled_by_config"
                elif time.time() < self._spec_disabled_until:
                    spec_block_reason = "cooldown_active"
                elif not spec_capable:
                    spec_block_reason = "capability_missing"
                elif max_tokens < self._spec_min_output_tokens:
                    spec_block_reason = "short_output"
                elif self._spec_require_distinct_draft and not self._spec_cache_only and not self._spec_auto_cache_only:
                    verify_model = str(job.get("model", {}).get("model_path", "")).strip()
                    draft_model = str(
                        effective_spec_cfg.get("draft_model")
                        or os.environ.get("KORITH_DRAFT_MODEL_PATH", "")
                        or os.environ.get("KORITH_DRAFT_MODEL", "")
                    ).strip()
                    if not draft_model:
                        spec_block_reason = "draft_model_missing"
                    elif verify_model and Path(draft_model).resolve() == Path(verify_model).resolve():
                        spec_block_reason = "draft_model_same_as_verify"
                    else:
                        # Keep speculative profitable by requiring a meaningfully smaller draft.
                        try:
                            if verify_model:
                                v_size = Path(verify_model).stat().st_size
                                d_size = Path(draft_model).stat().st_size
                                if v_size > 0:
                                    ratio = float(d_size) / float(v_size)
                                    if ratio > self._spec_max_draft_size_ratio:
                                        spec_block_reason = "draft_model_too_large"
                        except Exception:
                            pass
                if spec_block_reason is not None:
                    spec_disable_reason = spec_block_reason
                    self._append_event(
                        events_path,
                        "SPEC_DISABLE",
                        {"job_id": job_id, "request_id": request_id, "reason": spec_block_reason},
                    )
                else:
                    run_mode = "speculative"
                    if bool(getattr(self, "_vllm_runtime_contract_enabled", True)):
                        # Refresh contract with final per-request spec settings after governor tuning.
                        policy["_vllm_contract"] = self._build_vllm_runtime_contract(
                            lane=lane_up,
                            shape_key=request_shape_key,
                            max_tokens=max_tokens,
                            queue_latency_ms=queue_latency_ms,
                            has_snapshot=bool(mf_snapshot_in),
                            vllm_priority=vllm_priority,
                            routing_decision=routing_decision if isinstance(routing_decision, dict) else {},
                            spec_requested=True,
                            spec_cfg=effective_spec_cfg,
                        )
                    self._append_event(
                        events_path,
                        "SPEC_ENABLE",
                        {"job_id": job_id, "request_id": request_id, "k": effective_spec_cfg["k"]},
                    )
                    spec_cfg = dict(effective_spec_cfg)
                    if "cache_only" not in spec_cfg:
                        spec_cfg["cache_only"] = bool(self._spec_cache_only)
                    result = adapter.run_speculative(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        deterministic_cfg=deterministic_cfg,
                        policy=policy,
                        artifacts={k: str(v) for k, v in artifacts.items()},
                        mf_snapshot_in=mf_snapshot_in,
                        spec_cfg=spec_cfg,
                        engine_cfg=request_engine_cfg,
                    )
                    if isinstance(result, dict):
                        em = result.get("engine_metrics", {})
                        sm = em.get("spec", {}) if isinstance(em, dict) else {}
                        pm = em.get("perf", {}) if isinstance(em, dict) else {}
                        if isinstance(sm, dict) and bool(sm.get("cache_hit", False)) and isinstance(pm, dict):
                            decode_ms = float(pm.get("decode_ms", 0.0) or 0.0)
                            tokens_out = int(pm.get("tokens_out", 0) or 0)
                            no_decode_cache_hit = (decode_ms <= 0.0 and tokens_out <= 1)
                    # If speculative was requested but adapter/engine disabled it, immediately
                    # fall back to accel/baseline for this request.
                    if isinstance(result, dict):
                        em = result.get("engine_metrics", {})
                        sm = em.get("spec", {}) if isinstance(em, dict) else {}
                        if isinstance(sm, dict) and not bool(sm.get("enabled", False)):
                            disable_reason = str(sm.get("disable_reason", "") or "engine_spec_unavailable")
                            draft_init_reason = str(sm.get("draft_init_reason", "") or "")
                            spec_disable_reason = disable_reason
                            if draft_init_reason:
                                spec_draft_init_reason = draft_init_reason
                            self._append_event(
                                events_path,
                                "SPEC_DISABLE",
                                {
                                    "job_id": job_id,
                                    "request_id": request_id,
                                    "reason": disable_reason,
                                    "draft_init_reason": draft_init_reason,
                                },
                            )
                            result = None
                    if isinstance(result, dict) and result.get("_spec_cache_only_miss"):
                        self._append_event(
                            events_path,
                            "SPEC_DISABLE",
                            {"job_id": job_id, "request_id": request_id, "reason": "cache_miss"},
                        )
                        result = None

            if result is None and self._accel_enabled:
                run_mode = "accel"
                result = adapter.run_accel(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    deterministic_cfg=deterministic_cfg,
                    policy=policy,
                    artifacts={k: str(v) for k, v in artifacts.items()},
                    mf_snapshot_in=mf_snapshot_in,
                    engine_cfg=request_engine_cfg,
                )

            if result is None:
                run_mode = "baseline"
                result = adapter.run_baseline(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    deterministic_cfg=deterministic_cfg,
                    policy=policy,
                    artifacts={k: str(v) for k, v in artifacts.items()},
                    mf_snapshot_in=mf_snapshot_in,
                )
                if kernel_requested:
                    kernel_fallback = True
                    self._kernel_session_disabled = True
                    GLOBAL_METRICS.inc("kernels.fallback_total", 1.0, labels={"org_id": org_id})
                    self._append_event(events_path, "KERNEL_FALLBACK", {"job_id": job_id, "request_id": request_id, "reason": "baseline_path"})

            fallback = self._try_vllm_oom_fallback(
                job=job,
                lane=lane,
                result=result if isinstance(result, dict) else {},
                prompt=prompt,
                max_tokens=max_tokens,
                deterministic_cfg=deterministic_cfg,
                policy=policy,
                artifacts=artifacts,
                mf_snapshot_in=mf_snapshot_in,
                events_path=events_path,
                request_id=request_id,
            )
            if isinstance(fallback, dict):
                adapter = fallback["adapter"]
                caps = fallback["caps"]
                result = fallback["result"]
                run_mode = "baseline"
                cache_key = self._decode_cache_key(job)
                kernel_requested = False
                kernel_available = False
                kernel_executed = False
                kernel_verify = False
                kernel_fallback = False
                kernel_verify_ok = True
                kernel_ms_saved = 0.0
                no_decode_cache_hit = False
                kernel_policy_reason = "backend_fallback"

            if kernel_requested and kernel_available and run_mode in ("accel", "speculative") and not no_decode_cache_hit:
                kernel_executed = True
                if self._kernel_calibrate and run_mode == "accel":
                    self._kernel_calibrate_counter += 1
                    if (self._kernel_calibrate_counter % self._kernel_calibrate_every) == 0:
                        try:
                            shadow_decode_baseline_ms, shadow_tokens_out = self._run_kernel_shadow_baseline(
                                adapter=adapter,
                                prompt=prompt,
                                max_tokens=max_tokens,
                                deterministic_cfg=deterministic_cfg,
                                policy=policy,
                                mf_snapshot_in=mf_snapshot_in,
                                artifacts=artifacts,
                                job_id=job_id,
                            )
                        except Exception as exc:
                            self._append_event(
                                events_path,
                                "KERNEL_CALIBRATION_FAILED",
                                {"job_id": job_id, "request_id": request_id, "error": str(exc)},
                            )
                self._append_event(
                    events_path,
                    "KERNEL_PATH_USED",
                    {
                        "job_id": job_id,
                        "request_id": request_id,
                        "backend": self._kernel_ctx.backend,
                        "verify": kernel_verify,
                    },
                )
                if kernel_verify:
                    try:
                        ref = adapter.run_baseline(
                            prompt=prompt,
                            max_tokens=min(max_tokens, 64),
                            deterministic_cfg=deterministic_cfg,
                            policy=dict(policy),
                            artifacts={k: str(v) for k, v in artifacts.items()},
                            mf_snapshot_in=mf_snapshot_in,
                        )
                        out_accel = str(result.get("output_text", "")).strip()
                        out_ref = str(ref.get("output_text", "")).strip()
                        if out_accel and out_ref and out_accel.split()[:8] != out_ref.split()[:8]:
                            kernel_verify_ok = False
                            kernel_fallback = True
                            self._kernel_session_disabled = True
                            kernel_ms_saved = 0.0
                            result = ref
                            run_mode = "baseline"
                            GLOBAL_METRICS.inc("kernels.verify_failures_total", 1.0, labels={"org_id": org_id})
                            GLOBAL_METRICS.inc("kernels.fallback_total", 1.0, labels={"org_id": org_id})
                            self._append_event(
                                events_path,
                                "KERNEL_FALLBACK",
                                {"job_id": job_id, "request_id": request_id, "reason": "shadow_mismatch"},
                            )
                        else:
                            kernel_ms_saved = 0.0
                    except Exception:
                        kernel_verify_ok = False
                        kernel_fallback = True
                        self._kernel_session_disabled = True
                        GLOBAL_METRICS.inc("kernels.verify_failures_total", 1.0, labels={"org_id": org_id})
                        GLOBAL_METRICS.inc("kernels.fallback_total", 1.0, labels={"org_id": org_id})
                        self._append_event(
                            events_path,
                            "KERNEL_FALLBACK",
                            {"job_id": job_id, "request_id": request_id, "reason": "shadow_error"},
                        )
            if (
                not decode_cache_hit
                and cache_key
                and self._decode_cache_store is not None
                and self._decode_cache_eligible(job)
                and isinstance(result, dict)
                and int(result.get("exit_code", 0)) == 0
            ):
                em = result.get("engine_metrics", {}) if isinstance(result.get("engine_metrics", {}), dict) else {}
                perf = em.get("perf", {}) if isinstance(em.get("perf", {}), dict) else {}
                cached_output = str(result.get("output_text", "") or "")
                tokens_out = int(perf.get("tokens_out", len(cached_output.split())) or len(cached_output.split()))
                decode_ms = float(perf.get("decode_ms", 0.0) or 0.0)
                total_ms = float(perf.get("total_ms", result.get("total_ms", 0.0)) or result.get("total_ms", 0.0))
                self._decode_cache_store.set(
                    **cache_key,
                    output_text=cached_output,
                    tokens_out=tokens_out,
                    decode_ms=decode_ms,
                    total_ms=total_ms,
                    updated_at=time.time(),
                )
                self._append_event(
                    events_path,
                    "DECODE_CACHE_STORE",
                    {"job_id": job_id, "request_id": request_id, "tokens_out": tokens_out},
                )
        except Exception as exc:
            self._append_event(events_path, "ERROR", {"job_id": job_id, "request_id": request_id, **error_fields(exc, replay_related=True)})
            self.ledger.update_status(job_id, "FAILED")
            if mf_snapshot_in:
                self._append_event(events_path, "MF_RESTORE_FAILED", {"job_id": job_id, "reason": "apply_failed", "request_id": request_id})
                GLOBAL_METRICS.inc("mf_restore_failures", 1.0, labels={"org_id": org_id})
            GLOBAL_METRICS.inc("replay_failures_total", 1.0, labels={"org_id": org_id})
            emit_log(
                "worker",
                "ERROR",
                {
                    "job_id": job_id,
                    "run_id": "",
                    "worker_id": self.worker_id,
                    "session_id": session_id,
                    "org_id": org_id,
                    "request_id": request_id,
                    "latency_ms": 0.0,
                    **error_fields(exc, replay_related=True),
                },
                artifact_log=platform_log,
            )
            return
        finished_at = utc_now()

        engine_metrics = result.get("engine_metrics") or {}
        if isinstance(engine_metrics, dict):
            engine_metrics.setdefault("engine", {})
            engine_metrics["engine"]["mode"] = run_mode
            engine_metrics["engine"]["accel_enabled"] = (run_mode in ("accel", "speculative"))
            engine_metrics["engine"]["cuda_device"] = int(self._engine_cfg.get("cuda_device", self.gpu_id))
            engine_metrics["engine"]["kv_layout_version"] = str(self._engine_cfg.get("kv_layout_version", "v1"))
            if spec_disable_reason:
                engine_metrics.setdefault("spec", {})
                engine_metrics["spec"]["disable_reason"] = str(spec_disable_reason)
            if spec_policy_reason:
                engine_metrics.setdefault("spec", {})
                engine_metrics["spec"]["policy_reason"] = str(spec_policy_reason)
                engine_metrics["spec"]["shape_key"] = str(spec_policy_shape_key)
            if spec_draft_init_reason:
                engine_metrics.setdefault("spec", {})
                engine_metrics["spec"]["draft_init_reason"] = str(spec_draft_init_reason)
            engine_metrics.setdefault("kernels", {})
            engine_metrics["kernels"]["enabled"] = bool(kernel_requested and run_mode in ("accel", "speculative"))
            engine_metrics["kernels"]["kernels_applied"] = bool(kernel_executed)
            engine_metrics["kernels"]["backend"] = str(self._kernel_ctx.backend)
            engine_metrics["kernels"]["verify"] = bool(kernel_verify)
            engine_metrics["kernels"]["verify_ok"] = bool(kernel_verify_ok)
            engine_metrics["kernels"]["fallback"] = bool(kernel_fallback)
            engine_metrics["kernels"]["ms_saved"] = float(kernel_ms_saved)
            engine_metrics["kernels"]["policy_reason"] = str(kernel_policy_reason)
            engine_metrics["kernels"]["shape_key"] = str(kernel_policy_shape_key)
            engine_metrics["kernels"]["calibration_mode"] = str(getattr(self, "_decode_calibration_mode", "rolling"))
            if decode_cache_checked:
                health = engine_metrics.setdefault("health", {})
                if isinstance(health, dict):
                    health["decode_cache"] = {
                        "enabled": True,
                        "hit": bool(decode_cache_hit),
                        "saved_decode_ms_est": float(decode_cache_saved_est_ms),
                    }
        if (caps.kv_replay and caps.deterministic_seeding and int(result.get("exit_code", 0)) == 0
                and not isinstance(engine_metrics, dict)):
            engine_metrics = {}
        if caps.kv_replay and caps.deterministic_seeding and int(result.get("exit_code", 0)) == 0 and not engine_metrics:
            self._append_event(
                events_path,
                "ENGINE_METRICS_MISSING",
                {
                    "job_id": job_id,
                    "request_id": request_id,
                    "reason": "engine_binary_missing_authoritative_metrics",
                },
            )
            emit_log(
                "worker",
                "ENGINE_METRICS_MISSING",
                {
                    "job_id": job_id,
                    "run_id": "",
                    "worker_id": self.worker_id,
                    "session_id": session_id,
                    "org_id": org_id,
                    "request_id": request_id,
                    "latency_ms": 0.0,
                    "error_type": "integration",
                    "replay_related": True,
                },
                artifact_log=platform_log,
            )

        if isinstance(engine_metrics, dict):
            enforce_metrics_invariants(engine_metrics, no_decode_cache_hit=no_decode_cache_hit)
            self._update_kernel_timing_estimate(
                job=job,
                run_mode=run_mode,
                engine_metrics=engine_metrics,
                shadow_decode_baseline_ms=shadow_decode_baseline_ms,
                shadow_tokens_out=shadow_tokens_out,
            )
            self._update_kernel_policy_state(
                shape_key=kernel_policy_shape_key,
                engine_metrics=engine_metrics,
            )
            enforce_metrics_invariants(engine_metrics, no_decode_cache_hit=no_decode_cache_hit)
            kernels = engine_metrics.get("kernels", {}) if isinstance(engine_metrics.get("kernels", {}), dict) else {}
            if bool(kernels.get("enabled", False)):
                with platform_log.open("a", encoding="utf-8") as f:
                    f.write(
                        "[KORITH_KERNELS] backend={backend} applied={applied} decode_ms={decode:.3f} "
                        "baseline={baseline:.3f} saved={saved:.3f} comparable={comparable} tag={tag}\n".format(
                            backend=str(kernels.get("backend", "none") or "none"),
                            applied=1 if bool(kernels.get("kernels_applied", False)) else 0,
                            decode=float(kernels.get("decode_ms_actual", 0.0) or 0.0),
                            baseline=float(kernels.get("decode_ms_baseline_est", 0.0) or 0.0),
                            saved=float(kernels.get("ms_saved", 0.0) or 0.0),
                            comparable=1 if bool(kernels.get("comparable", False)) else 0,
                            tag=str(kernels.get("comparable_tag", "") or ""),
                        )
                    )

        metrics = self._compose_metrics(job, result, lane, queue_latency_ms, started_at, finished_at, caps)
        Path(artifacts["metrics"]).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        engine_events_path = result.get("engine_events_path")
        if engine_events_path and Path(engine_events_path).exists():
            with Path(engine_events_path).open("r", encoding="utf-8") as f, Path(events_path).open("a", encoding="utf-8") as out:
                for line in f:
                    out.write(line)
        else:
            self._append_event(events_path, "MF_APPLY", {"job_id": job_id, "supported": False, "request_id": request_id})

        cp = engine_metrics.get("cp") if isinstance(engine_metrics, dict) else None
        if cp:
            self._append_event(events_path, "CP_DECISION", {"job_id": job_id, "request_id": request_id, **cp})
        else:
            self._append_event(events_path, "CP_DECISION", {"job_id": job_id, "request_id": request_id, "reason": "unavailable"})

        self._decode_governor_spec_update(
            shape_key=spec_policy_shape_key,
            spec_metrics=metrics.get("spec", {}) if isinstance(metrics.get("spec", {}), dict) else {},
        )
        spec_metrics = metrics.get("spec", {}) if isinstance(metrics, dict) else {}
        if bool(spec_metrics.get("enabled", False)):
            self._append_event(
                events_path,
                "SPEC_START",
                {"job_id": job_id, "request_id": request_id, "k": int(spec_metrics.get("k", 0) or 0)},
            )
            self._append_event(
                events_path,
                "SPEC_SUMMARY",
                {
                    "job_id": job_id,
                    "request_id": request_id,
                    "k": int(spec_metrics.get("k", 0) or 0),
                    "proposed_tokens": int(spec_metrics.get("proposed_tokens", 0) or 0),
                    "accepted_tokens": int(spec_metrics.get("accepted_tokens", 0) or 0),
                    "acceptance_rate": float(spec_metrics.get("acceptance_rate", 0.0) or 0.0),
                    "speedup_est": float(spec_metrics.get("speedup_est", 0.0) or 0.0),
                },
            )
            proposed_tokens = int(spec_metrics.get("proposed_tokens", 0) or 0)
            accepted_tokens = int(spec_metrics.get("accepted_tokens", 0) or 0)
            acceptance = float(spec_metrics.get("acceptance_rate", 0.0) or 0.0)
            spec_roi = float(spec_metrics.get("roi", 0.0) or 0.0)
            net_saved_ms = float(spec_metrics.get("net_saved_ms", 0.0) or 0.0)
            baseline_total_ms = float(spec_metrics.get("baseline_total_ms", 0.0) or 0.0)
            cache_hit = bool(spec_metrics.get("cache_hit", False))
            has_spec_activity = cache_hit or proposed_tokens > 0

            if has_spec_activity:
                self._append_event(
                    events_path,
                    "SPEC_ACCEPT" if acceptance > 0.0 else "SPEC_REJECT",
                    {
                        "job_id": job_id,
                        "request_id": request_id,
                        "accepted_tokens": accepted_tokens,
                        "proposed_tokens": proposed_tokens,
                    },
                )
                self._spec_samples += 1
                if self._spec_samples == 1:
                    self._spec_accept_ema = acceptance
                    self._spec_roi_ema = spec_roi
                else:
                    self._spec_accept_ema = (0.9 * self._spec_accept_ema) + (0.1 * acceptance)
                    self._spec_roi_ema = (0.9 * self._spec_roi_ema) + (0.1 * spec_roi)
                low_accept_ema = self._spec_accept_ema < float(self._spec_cfg["min_accept"])
                low_roi_ema = self._spec_roi_ema < float(self._spec_min_roi)
                latency_regression = net_saved_ms <= 0.0
                severe_regression = net_saved_ms < -max(50.0, 0.10 * baseline_total_ms)
                if latency_regression:
                    GLOBAL_METRICS.inc("spec_latency_regressions_total", 1.0, labels={"org_id": org_id})
                if severe_regression:
                    self._spec_disabled_until = time.time() + 60.0
                    self._append_event(
                        events_path,
                        "SPEC_DISABLE",
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "reason": "latency_regression_immediate",
                            "net_saved_ms": net_saved_ms,
                            "baseline_total_ms": baseline_total_ms,
                            "cooldown_s": 60,
                        },
                    )
                elif self._spec_samples >= int(self._spec_cfg["disable_after_n"]) and (low_accept_ema or low_roi_ema or latency_regression):
                    self._spec_disabled_until = time.time() + 60.0
                    disable_reason = "low_acceptance" if low_accept_ema else "latency_regression"
                    self._append_event(
                        events_path,
                        "SPEC_DISABLE",
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "reason": disable_reason,
                            "acceptance_ema": self._spec_accept_ema,
                            "roi_ema": self._spec_roi_ema,
                            "cooldown_s": 60,
                        },
                    )
            else:
                self._append_event(
                    events_path,
                    "SPEC_NOOP",
                    {
                        "job_id": job_id,
                        "request_id": request_id,
                        "reason": "no_spec_activity",
                        "accepted_tokens": accepted_tokens,
                        "proposed_tokens": proposed_tokens,
                    },
                )
            self._append_event(events_path, "SPEC_END", {"job_id": job_id, "request_id": request_id})

            # Persist spec governance by fingerprint.
            if fingerprint_hash and has_spec_activity:
                state = self.ledger.get_spec_governance(fingerprint_hash) or {
                    "spec_disabled": 0,
                    "reason": None,
                    "cooldown_until": 0.0,
                    "bad_accept_streak": 0,
                }
                bad_streak = int(state.get("bad_accept_streak", 0) or 0)
                spec_disabled = int(state.get("spec_disabled", 0) or 0)
                reason = state.get("reason")
                cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
                low_accept = acceptance < float(self._spec_cfg["min_accept"])
                low_roi = spec_roi < float(self._spec_min_roi)
                latency_regression = net_saved_ms <= 0.0
                severe_regression = net_saved_ms < -max(50.0, 0.10 * baseline_total_ms)
                if low_accept or low_roi or latency_regression:
                    bad_streak += 1
                    if severe_regression:
                        bad_streak = max(bad_streak, self._spec_bad_max)
                else:
                    bad_streak = 0
                    spec_disabled = 0
                    reason = None
                    cooldown_until = 0.0
                if bad_streak >= self._spec_bad_max:
                    spec_disabled = 1
                    reason = "latency_regression" if (low_roi or latency_regression) and not low_accept else "low_acceptance"
                    cooldown_until = time.time() + (self._spec_cooldown_ms / 1000.0)
                    self._append_event(
                        events_path,
                        "SPEC_DISABLE",
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "reason": reason,
                            "roi": spec_roi,
                            "net_saved_ms": net_saved_ms,
                            "cooldown_ms": self._spec_cooldown_ms,
                        },
                    )
                self.ledger.upsert_spec_governance(
                    fingerprint_hash=fingerprint_hash,
                    org_id=org_id,
                    spec_disabled=spec_disabled,
                    reason=reason,
                    cooldown_until=cooldown_until,
                    bad_accept_streak=bad_streak,
                    updated_at=utc_now(),
                )

        amf_decision = metrics.get("amf", {}).get("decision")
        amf_lookup_hash = str((payload or {}).get("amf_lookup_hash", "") or "")
        if not amf_lookup_hash:
            routing_decision = job.get("routing_decision", {}) if isinstance(job.get("routing_decision", {}), dict) else {}
            amf_lookup_hash = str(routing_decision.get("amf_lookup_hash", "") or "")
        if (
            amf_lookup_hash
            and bool(metrics.get("amf", {}).get("supported", False))
            and amf_decision in ("hit", "miss")
        ):
            amf_metrics = metrics.get("amf", {}) if isinstance(metrics.get("amf", {}), dict) else {}
            self._coordinator_register_entry(
                tenant_id=tenant_id,
                amf_lookup_hash=amf_lookup_hash,
                worker_id=self.worker_id,
                metadata={
                    "cache_entries": int(amf_metrics.get("cache_entries", 0) or 0),
                    "hit_rate": float(amf_metrics.get("hit_rate", self._amf_hit_rate) or self._amf_hit_rate),
                    "warm_ratio": float(amf_metrics.get("warm_ratio", self._amf_warm_ratio) or self._amf_warm_ratio),
                },
            )
        if amf_decision in ("hit", "miss", "blocked"):
            self._append_event(events_path, f"AMF_{amf_decision.upper()}", {"job_id": job_id, "request_id": request_id})
        if _orch_outcome is not None and getattr(_orch_outcome, "ai_augmented", False):
            _orch_prefix = str(job.get("prefix_hash", "") or "")
            try:
                if amf_decision == "hit":
                    self._orchestrator.on_hit(_orch_prefix)
                elif amf_decision == "miss":
                    self._orchestrator.on_miss(_orch_prefix)
            except Exception:
                pass
        if amf_decision == "hit" and self.snapshot_index is not None:
            locations = self.snapshot_index.get_locations(fingerprint_hash=fingerprint_hash, org_id=org_id)
            if locations:
                first = next(
                    (row for row in locations if str(row.get("node_id", "")) in ("", self.node_id)),
                    locations[0],
                )
                self.snapshot_index.mark_used(
                    fingerprint_hash=fingerprint_hash,
                    org_id=org_id,
                    node_id=str(first.get("node_id", self.node_id)),
                    snapshot_id=str(first.get("snapshot_id", job_id)),
                )
        if amf_decision == "blocked":
            GLOBAL_METRICS.inc("replay_failures_total", 1.0, labels={"org_id": org_id})

        self._apply_governance(job, metrics, events_path)

        self._append_event(events_path, "WORKER_END", {
            "job_id": job_id,
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "request_id": request_id,
            "exit_code": result.get("exit_code", 0),
        })

        run_id = str(uuid.uuid4())
        self.ledger.insert_run(
            run_id,
            job_id,
            started_at,
            finished_at,
            int(result.get("exit_code", 0)),
            str(artifacts["metrics"]),
            str(artifacts["events"]),
            str(artifacts["output"]),
            str(artifacts["log"]),
            org_id=org_id,
        )

        if Path(artifacts["mf_snapshot"]).exists():
            snapshot_path_for_ledger = str(artifacts["mf_snapshot"])
            snapshot_path_for_index = str(artifacts["mf_snapshot"])
            snapshot_size = int(Path(artifacts["mf_snapshot"]).stat().st_size)
            snapshot_tier = self._snapshot_tier_for_size(snapshot_size)
            self.ledger.insert_snapshot(
                snapshot_id=job_id,
                job_id=job_id,
                fingerprint_hash=job.get("fingerprint_hash", ""),
                snapshot_path=snapshot_path_for_ledger,
                created_at=finished_at,
                org_id=org_id,
            )
            try:
                write_snapshot_metadata(
                    artifacts["mf_snapshot"],
                    fingerprint_hash=str(job.get("fingerprint_hash", "")),
                    model_hash=str(job.get("fingerprint", {}).get("model_hash", "")),
                    tokenizer_hash=str(job.get("fingerprint", {}).get("tokenizer_hash", "")),
                    backend_id=str(job.get("execution_backend_id", "") or job.get("backend_id", "")),
                    n_ctx=int(job.get("deterministic_cfg", {}).get("n_ctx", 0) or 0),
                    kv_layout_version=str(self._engine_cfg.get("kv_layout_version", "v1")),
                    created_at=finished_at,
                )
            except Exception:
                pass
            try:
                cached_path, cached_size, cached_tier = self._cache_snapshot_from_file(
                    fingerprint_hash=str(job.get("fingerprint_hash", "")),
                    source_path=Path(artifacts["mf_snapshot"]),
                )
                snapshot_path_for_index = str(cached_path)
                snapshot_size = int(cached_size)
                snapshot_tier = cached_tier
            except Exception:
                pass
            self.restore_store.set(str(job.get("fingerprint_hash", "")), snapshot_path_for_index)
            if self.snapshot_index is not None:
                self.snapshot_index.upsert_location(
                    fingerprint_hash=job.get("fingerprint_hash", ""),
                    snapshot_id=job_id,
                    org_id=org_id,
                    node_id=self.node_id,
                    worker_id=self.worker_id,
                    snapshot_path=snapshot_path_for_index,
                    size_bytes=snapshot_size,
                    created_at=finished_at,
                    last_used_at=finished_at,
                )
            self._append_event(
                events_path,
                "MF_SNAPSHOT",
                {
                    "job_id": job_id,
                    "path": str(artifacts["mf_snapshot"]),
                    "cache_path": snapshot_path_for_index,
                    "snapshot_tier": snapshot_tier,
                    "request_id": request_id,
                },
            )

        status = "SUCCEEDED" if int(result.get("exit_code", 0)) == 0 else "FAILED"
        self.ledger.update_status(job_id, status)

        for key in ("metrics", "events", "output", "log", "mf_snapshot"):
            path = artifacts.get(key)
            if not path or not Path(path).exists():
                continue
            try:
                blob = Path(path).read_bytes()
                if hasattr(self.artifacts, "store_artifact"):
                    self.artifacts.store_artifact(job_id, key, blob, org_id=org_id)
            except Exception:
                continue

        self._update_health(metrics, shape_key=request_shape_key)
        emit_log(
            "worker",
            "WORKER_END",
            {
                "job_id": job_id,
                "run_id": run_id,
                "worker_id": self.worker_id,
                "session_id": session_id,
                "org_id": org_id,
                "request_id": request_id,
                "latency_ms": float(metrics.get("perf", {}).get("total_ms", 0.0) or 0.0),
            },
            artifact_log=platform_log,
        )

    def _apply_governance(self, job: Dict[str, Any], metrics: Dict[str, Any], events_path: Path) -> None:
        fingerprint_hash = job.get("fingerprint_hash", "")
        if not fingerprint_hash:
            return
        org_id = job.get("org_id", "default")
        request_id = job.get("request_id", "")
        amf = metrics.get("amf", {})
        decision = str(amf.get("decision", ""))
        roi = float(amf.get("roi", 0.0) or 0.0)
        baseline_prefix_ms = float(amf.get("baseline_prefix_ms", 0.0) or 0.0)
        restore_ms = float(amf.get("restore_ms", 0.0) or 0.0)
        now_epoch = time.time()
        now_ts = utc_now()

        state = self.ledger.get_replay_governance(fingerprint_hash) or {
            "negative_roi_streak": 0,
            "replay_disabled": 0,
            "corruption_detected": 0,
            "restore_guard_disabled": 0,
            "cooldown_until": 0.0,
        }

        replay_disabled = int(state.get("replay_disabled", 0) or 0)
        disabled_reason = state.get("disabled_reason")
        disabled_at = state.get("disabled_at")
        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        negative_roi_streak = int(state.get("negative_roi_streak", 0) or 0)
        corruption_detected = int(state.get("corruption_detected", 0) or 0)
        restore_guard_disabled = int(state.get("restore_guard_disabled", 0) or 0)

        if decision == "hit" and baseline_prefix_ms > 0 and restore_ms > baseline_prefix_ms:
            replay_disabled = 1
            restore_guard_disabled = 1
            disabled_reason = "restore_latency_guard"
            disabled_at = now_ts
            cooldown_until = 0.0
            self._append_event(events_path, "REPLAY_DISABLE", {"job_id": job.get("job_id", ""), "request_id": request_id, "reason": disabled_reason})
            GLOBAL_METRICS.inc("replay_disable_events", 1.0, labels={"org_id": org_id})

        if decision == "hit" and baseline_prefix_ms > 0 and restore_ms > 0:
            if roi < 1.0:
                negative_roi_streak += 1
                GLOBAL_METRICS.inc("negative_roi_streaks", 1.0, labels={"org_id": org_id})
                if negative_roi_streak >= self._negative_roi_max:
                    cooldown_until = now_epoch + (self._negative_roi_cooldown_ms / 1000.0)
                    disabled_reason = "negative_roi_cooldown"
                    disabled_at = now_ts
                    negative_roi_streak = 0
                    self._append_event(
                        events_path,
                        "REPLAY_DISABLE",
                        {
                            "job_id": job.get("job_id", ""),
                            "request_id": request_id,
                            "reason": disabled_reason,
                            "cooldown_ms": self._negative_roi_cooldown_ms,
                        },
                    )
                    GLOBAL_METRICS.inc("replay_disable_events", 1.0, labels={"org_id": org_id})
            else:
                negative_roi_streak = 0

        if self._event_contains(events_path, "CORRUPTION_DETECTED"):
            replay_disabled = 1
            corruption_detected = 1
            disabled_reason = "corruption_detected"
            disabled_at = now_ts
            cooldown_until = 0.0
            self._append_event(events_path, "REPLAY_DISABLE", {"job_id": job.get("job_id", ""), "request_id": request_id, "reason": disabled_reason})
            GLOBAL_METRICS.inc("corruption_detected_events", 1.0, labels={"org_id": org_id})
            GLOBAL_METRICS.inc("replay_disable_events", 1.0, labels={"org_id": org_id})

        self.ledger.upsert_replay_governance(
            fingerprint_hash=fingerprint_hash,
            replay_disabled=replay_disabled,
            disabled_reason=disabled_reason,
            disabled_at=disabled_at,
            cooldown_until=cooldown_until,
            negative_roi_streak=negative_roi_streak,
            corruption_detected=corruption_detected,
            restore_guard_disabled=restore_guard_disabled,
            updated_at=now_ts,
        )

    def _event_contains(self, events_path: Path, event_type: str) -> bool:
        if not events_path.exists():
            return False
        try:
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    if event.get("type") == event_type:
                        return True
        except Exception:
            return False
        return False

    def _snapshot_tier_for_size(self, size_bytes: int) -> str:
        size = max(0, int(size_bytes))
        if size <= self._snapshot_vram_max_bytes:
            return "vram"
        if size <= self._snapshot_ram_max_bytes:
            return "ram"
        return "nvme"

    def _is_kv_transfer_snapshot(self, snapshot_path: str) -> bool:
        path = Path(str(snapshot_path or ""))
        if not path.exists() or not path.is_file():
            return False
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return False
        try:
            payload = json.loads(text)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        kv = payload.get("kv_transfer_params")
        if isinstance(kv, dict) and kv:
            return True
        # Accept raw connector payloads as native KV transfer snapshots.
        if "connector" in payload and ("kv_role" in payload or "cache_id" in payload):
            return True
        return False

    def _snapshot_dir_for_tier(self, tier: str) -> Path:
        if tier == "vram":
            return self._snapshot_vram_dir
        if tier == "ram":
            return self._snapshot_ram_dir
        return self._snapshot_nvme_dir

    def _snapshot_cache_max_bytes_for_tier(self, tier: str) -> int:
        if tier == "vram":
            return self._snapshot_vram_cache_max_bytes
        if tier == "ram":
            return self._snapshot_ram_cache_max_bytes
        return self._snapshot_nvme_cache_max_bytes

    def _snapshot_path_tier(self, path: Path) -> str:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path
        for tier, root in (
            ("vram", self._snapshot_vram_dir),
            ("ram", self._snapshot_ram_dir),
            ("nvme", self._snapshot_nvme_dir),
        ):
            try:
                if resolved.is_relative_to(root):
                    return tier
            except Exception:
                pass
        return "unknown"

    def _snapshot_tier_rank(self, tier: str) -> int:
        if tier == "vram":
            return 0
        if tier == "ram":
            return 1
        if tier == "nvme":
            return 2
        return 3

    def _enforce_snapshot_tier_budget(self, tier: str, protect_path: Optional[Path] = None) -> None:
        max_bytes = self._snapshot_cache_max_bytes_for_tier(tier)
        if max_bytes <= 0:
            return
        tier_dir = self._snapshot_dir_for_tier(tier)
        if not tier_dir.exists():
            return
        files: list[Path] = []
        total = 0
        for path in tier_dir.glob("*.bin"):
            if not path.is_file():
                continue
            try:
                size = int(path.stat().st_size)
            except Exception:
                continue
            total += size
            files.append(path)
        if total <= max_bytes:
            return
        files = sorted(files, key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, p.name))
        for path in files:
            if protect_path is not None:
                try:
                    if path.resolve() == protect_path.resolve():
                        continue
                except Exception:
                    pass
            try:
                size = int(path.stat().st_size)
            except Exception:
                size = 0
            try:
                path.unlink(missing_ok=True)
            except Exception:
                continue
            total -= size
            if total <= max_bytes:
                break

    def _cache_snapshot_from_blob(self, fingerprint_hash: str, blob: bytes) -> tuple[Path, int, str]:
        size = len(blob)
        tier = self._snapshot_tier_for_size(size)
        out_dir = self._snapshot_dir_for_tier(tier)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fingerprint_hash}.bin"
        out_path.write_bytes(blob)
        self._enforce_snapshot_tier_budget(tier, protect_path=out_path)
        return out_path, int(size), tier

    def _cache_snapshot_from_file(self, fingerprint_hash: str, source_path: Path) -> tuple[Path, int, str]:
        size = int(source_path.stat().st_size if source_path.exists() else 0)
        tier = self._snapshot_tier_for_size(size)
        out_dir = self._snapshot_dir_for_tier(tier)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fingerprint_hash}.bin"
        if source_path.resolve() != out_path.resolve():
            shutil.copy2(source_path, out_path)
        source_meta = snapshot_meta_path(source_path)
        target_meta = snapshot_meta_path(out_path)
        if source_meta.exists():
            if source_meta.resolve() != target_meta.resolve():
                shutil.copy2(source_meta, target_meta)
        else:
            # Avoid stale sidecars when the source snapshot has no metadata.
            target_meta.unlink(missing_ok=True)
        self._enforce_snapshot_tier_budget(tier, protect_path=out_path)
        return out_path, int(size), tier

    def _cache_snapshot_location_id(self, fingerprint_hash: str) -> str:
        return f"cache:{str(fingerprint_hash)[:24]}"

    def _maybe_promote_snapshot_input(
        self,
        *,
        fingerprint_hash: str,
        org_id: str,
        snapshot_path: str,
        events_path: Path,
        job_id: str,
        request_id: str,
    ) -> str:
        src = Path(snapshot_path)
        if not src.exists():
            return snapshot_path
        current_tier = self._snapshot_path_tier(src)
        try:
            target_path, size_bytes, target_tier = self._cache_snapshot_from_file(
                fingerprint_hash=fingerprint_hash,
                source_path=src,
            )
        except Exception:
            return snapshot_path
        resolved = str(target_path)
        self.restore_store.set(fingerprint_hash, resolved)
        if self.snapshot_index is not None:
            try:
                self.snapshot_index.upsert_location(
                    fingerprint_hash=fingerprint_hash,
                    snapshot_id=self._cache_snapshot_location_id(fingerprint_hash),
                    org_id=org_id,
                    node_id=self.node_id,
                    worker_id=self.worker_id,
                    snapshot_path=resolved,
                    size_bytes=int(size_bytes),
                    created_at=utc_now(),
                    last_used_at=utc_now(),
                )
            except Exception:
                pass
        if self._snapshot_tier_rank(target_tier) < self._snapshot_tier_rank(current_tier):
            self._append_event(
                events_path,
                "SNAPSHOT_PROMOTE",
                {
                    "job_id": job_id,
                    "request_id": request_id,
                    "from": str(src),
                    "to": resolved,
                    "from_tier": current_tier,
                    "to_tier": target_tier,
                },
            )
        return resolved

    def _resolve_snapshot_input_path(
        self,
        *,
        fingerprint_hash: str,
        org_id: str,
        candidate_path: Optional[str],
    ) -> Optional[str]:
        if candidate_path and Path(candidate_path).exists():
            return candidate_path
        if self.snapshot_index is None:
            return None
        locations = self.snapshot_index.get_locations(fingerprint_hash=fingerprint_hash, org_id=org_id)
        # Prefer same-node paths first for locality.
        for row in locations:
            if row.get("node_id") not in ("", self.node_id):
                continue
            path = Path(str(row.get("snapshot_path", "") or ""))
            if not path.exists():
                continue
            resolved = str(path)
            self.restore_store.set(fingerprint_hash, resolved)
            return resolved
        # Fallback: if node IDs changed across restarts, still reuse any locally
        # accessible snapshot path instead of forcing a miss.
        for row in locations:
            path = Path(str(row.get("snapshot_path", "") or ""))
            if not path.exists():
                continue
            resolved = str(path)
            self.restore_store.set(fingerprint_hash, resolved)
            try:
                self.snapshot_index.upsert_location(
                    fingerprint_hash=fingerprint_hash,
                    snapshot_id=str(row.get("snapshot_id", "")) or str(uuid.uuid4()),
                    org_id=org_id,
                    node_id=self.node_id,
                    worker_id=self.worker_id,
                    snapshot_path=resolved,
                    size_bytes=int(path.stat().st_size),
                    created_at=utc_now(),
                    last_used_at=utc_now(),
                )
            except Exception:
                pass
            return resolved
        return None

    def _kernel_baseline_key(self, *, job: Dict[str, Any], tokens_out: int) -> str:
        det = job.get("deterministic_cfg", {}) if isinstance(job.get("deterministic_cfg", {}), dict) else {}
        model = (
            job.get("execution_model", {})
            if isinstance(job.get("execution_model"), dict) and job.get("execution_model")
            else (job.get("model", {}) if isinstance(job.get("model", {}), dict) else {})
        )
        fp = (
            job.get("execution_fingerprint", {})
            if isinstance(job.get("execution_fingerprint"), dict) and job.get("execution_fingerprint")
            else (job.get("fingerprint", {}) if isinstance(job.get("fingerprint", {}), dict) else {})
        )
        backend_id = str(job.get("execution_backend_id", "") or job.get("backend_id", "") or "")
        model_id = str(model.get("model_id", "") or "")
        model_hash = str(fp.get("model_hash", "") or "")
        n_batch = int(det.get("n_batch", 0) or 0)
        n_ctx_bucket = _bucket_ceil(int(det.get("n_ctx", 0) or 0), 1024)
        prompt_bucket = _bucket_ceil(int(job.get("prompt_tokens", 0) or 0), 64)
        tok_out = max(0, int(tokens_out))
        return "|".join(
            [
                model_hash,
                model_id,
                backend_id,
                str(n_batch),
                str(n_ctx_bucket),
                str(prompt_bucket),
                str(tok_out),
                self._kernel_gpu_arch,
                str(self.gpu_id),
                "nonspec",
            ]
        )

    def _kernel_policy_shape_key(self, job: Dict[str, Any]) -> str:
        det = job.get("deterministic_cfg", {}) if isinstance(job.get("deterministic_cfg", {}), dict) else {}
        model = (
            job.get("execution_model", {})
            if isinstance(job.get("execution_model"), dict) and job.get("execution_model")
            else (job.get("model", {}) if isinstance(job.get("model", {}), dict) else {})
        )
        fp = (
            job.get("execution_fingerprint", {})
            if isinstance(job.get("execution_fingerprint"), dict) and job.get("execution_fingerprint")
            else (job.get("fingerprint", {}) if isinstance(job.get("fingerprint", {}), dict) else {})
        )
        backend_id = str(job.get("execution_backend_id", "") or job.get("backend_id", "") or "")
        model_id = str(model.get("model_id", "") or "")
        model_hash = str(fp.get("model_hash", "") or "")
        n_batch = int(det.get("n_batch", 0) or 0)
        n_ctx_bucket = _bucket_ceil(int(det.get("n_ctx", 0) or 0), 1024)
        prompt_bucket = _bucket_ceil(int(job.get("prompt_tokens", 0) or 0), 128)
        max_tokens_bucket = _bucket_ceil(int(det.get("max_tokens", 0) or 0), 64)
        return "|".join(
            [
                model_hash,
                model_id,
                backend_id,
                str(n_batch),
                str(n_ctx_bucket),
                str(prompt_bucket),
                str(max_tokens_bucket),
                self._kernel_gpu_arch,
                str(self.gpu_id),
            ]
        )

    def _decode_governor_spec_allow(self, *, shape_key: str, lane: str) -> tuple[bool, str]:
        if not shape_key:
            return True, "shape_unknown"
        if not self._decode_governor_enabled:
            return True, "governor_disabled"
        lane_up = str(lane).upper()
        if lane_up not in ("SPEC_HIT", "SPEC_MISS"):
            return True, "non_spec_lane"

        state = self._decode_governor_spec_state.setdefault(shape_key, {})
        requests = int(state.get("requests", 0) or 0) + 1
        state["requests"] = requests
        now = time.time()
        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        if cooldown_until > now:
            if (requests % self._decode_governor_probe_every) == 0:
                state["last_decision"] = "probe"
                return True, "probe"
            state["last_decision"] = "cooldown"
            return False, "governor_cooldown"

        state["last_decision"] = "active"
        return True, "active"

    def _decode_governor_spec_k(self, *, shape_key: str, lane: str, requested_k: int, max_tokens: int) -> int:
        k = max(1, int(requested_k or 1))
        max_tokens_int = max(1, int(max_tokens or 1))
        k = min(k, max_tokens_int)
        if not self._decode_governor_enabled or not bool(getattr(self, "_spec_adaptive_k_enabled", True)):
            return k
        if not shape_key:
            return k
        lane_up = str(lane).upper()
        if lane_up not in ("SPEC_HIT", "SPEC_MISS"):
            return k

        state = self._decode_governor_spec_state.setdefault(shape_key, {})
        min_k = max(1, int(getattr(self, "_spec_k_min", 1) or 1))
        max_k = max(min_k, int(getattr(self, "_spec_k_max", max(min_k, k)) or max(min_k, k)))
        cur = int(state.get("k_current", k) or k)
        cur = min(max_k, max(min_k, cur))
        cur = min(cur, max_tokens_int)
        state["k_current"] = cur
        return cur

    def _decode_governor_spec_update(self, *, shape_key: str, spec_metrics: Dict[str, Any]) -> None:
        if not self._decode_governor_enabled or not shape_key:
            return
        if not isinstance(spec_metrics, dict):
            return
        if not bool(spec_metrics.get("enabled", False)):
            return

        proposed = int(spec_metrics.get("proposed_tokens", 0) or 0)
        accepted = int(spec_metrics.get("accepted_tokens", 0) or 0)
        cache_hit = bool(spec_metrics.get("cache_hit", False))
        has_activity = cache_hit or proposed > 0
        if not has_activity:
            return

        acceptance = float(spec_metrics.get("acceptance_rate", 0.0) or 0.0)
        roi = float(spec_metrics.get("roi", 0.0) or 0.0)
        net_saved_ms = float(spec_metrics.get("net_saved_ms", 0.0) or 0.0)
        state = self._decode_governor_spec_state.setdefault(shape_key, {})
        bad_streak = int(state.get("bad_streak", 0) or 0)

        prev_accept = float(state.get("accept_ema", acceptance) or acceptance)
        prev_roi = float(state.get("roi_ema", roi) or roi)
        state["accept_ema"] = acceptance if "accept_ema" not in state else (0.9 * prev_accept) + (0.1 * acceptance)
        state["roi_ema"] = roi if "roi_ema" not in state else (0.9 * prev_roi) + (0.1 * roi)

        low_accept = acceptance < float(self._spec_cfg["min_accept"])
        low_roi = roi < float(self._spec_min_roi)
        latency_regression = net_saved_ms <= 0.0
        if low_accept or low_roi or latency_regression:
            bad_streak += 1
        else:
            bad_streak = 0

        state["bad_streak"] = bad_streak
        if bad_streak >= self._spec_bad_max:
            state["cooldown_until"] = float(time.time() + (self._spec_cooldown_ms / 1000.0))

        if bool(getattr(self, "_spec_adaptive_k_enabled", True)):
            k_cur = int(
                state.get(
                    "k_current",
                    spec_metrics.get("k", self._spec_cfg.get("k", 1)),
                )
                or 1
            )
            min_k = max(1, int(getattr(self, "_spec_k_min", 1) or 1))
            max_k = max(min_k, int(getattr(self, "_spec_k_max", max(min_k, k_cur)) or max(min_k, k_cur)))
            k_new = k_cur

            down_accept = float(getattr(self, "_spec_adaptive_k_down_accept", 0.65) or 0.65)
            down_roi = float(getattr(self, "_spec_adaptive_k_down_roi", 0.9) or 0.9)
            up_accept = float(getattr(self, "_spec_adaptive_k_up_accept", 0.9) or 0.9)
            up_roi = float(getattr(self, "_spec_adaptive_k_up_roi", 1.1) or 1.1)

            if latency_regression or acceptance < down_accept or roi < down_roi:
                k_new = max(min_k, k_cur - 1)
            elif acceptance >= up_accept and roi >= up_roi and net_saved_ms > 0.0 and proposed > 0:
                k_new = min(max_k, k_cur + 1)

            if k_new != k_cur:
                state["k_current"] = int(k_new)
                state["k_last_change"] = float(time.time())

    def _vllm_decode_budget_tokens(self, *, lane: str, shape_key: str, max_tokens: int) -> int:
        max_tokens = max(1, int(max_tokens or 1))
        lane_up = str(lane).upper()
        ratio = float(
            self._vllm_decode_budget_ratio_by_lane.get(
                lane_up,
                getattr(self, "_vllm_decode_budget_ratio_default", 1.0),
            )
            or getattr(self, "_vllm_decode_budget_ratio_default", 1.0)
        )
        ratio = max(0.1, min(2.0, ratio))

        if shape_key:
            lane_decode_map = getattr(self, "_lane_decode_ema_ms", {})
            shape_decode_map = getattr(self, "_shape_decode_ema_ms", {})
            lane_decode_ema = float(lane_decode_map.get(lane_up, 0.0) or 0.0)
            shape_decode_ema = float(shape_decode_map.get(shape_key, 0.0) or 0.0)
            if lane_decode_ema > 0.0 and shape_decode_ema > 0.0:
                if shape_decode_ema <= (0.8 * lane_decode_ema):
                    ratio *= 1.10
                elif shape_decode_ema >= (1.2 * lane_decode_ema):
                    ratio *= 0.85
                ratio = max(0.1, min(2.0, ratio))

        budget = int(max(1, round(float(max_tokens) * ratio)))
        budget = max(int(getattr(self, "_vllm_decode_budget_min_tokens", 96) or 96), budget)
        return int(max(1, min(max_tokens, budget)))

    def _build_vllm_runtime_contract(
        self,
        *,
        lane: str,
        shape_key: str,
        max_tokens: int,
        queue_latency_ms: float,
        has_snapshot: bool,
        vllm_priority: Optional[int],
        routing_decision: Optional[Dict[str, Any]] = None,
        spec_requested: bool = False,
        spec_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not bool(getattr(self, "_vllm_runtime_contract_enabled", True)):
            return {}
        lane_up = str(lane).upper()
        routing = routing_decision if isinstance(routing_decision, dict) else {}
        replay_local = bool(routing.get("replay_local", False))
        transfer_requested = bool(routing.get("transfer_requested", False))
        snapshot_tier = str(routing.get("snapshot_tier", "") or "").strip()
        prompt_tokens = int(routing.get("prompt_tokens", 0) or 0)
        if lane_up in ("HIT", "SPEC_HIT"):
            replay_state = "hit"
        else:
            replay_state = "miss"
        if has_snapshot:
            replay_state = "restore"
        elif transfer_requested:
            replay_state = "restore"
        merged_spec_cfg = dict(getattr(self, "_spec_cfg", {}) or {})
        if isinstance(spec_cfg, dict):
            merged_spec_cfg.update(spec_cfg)
        try:
            spec_k = max(1, int(merged_spec_cfg.get("k", 1) or 1))
        except Exception:
            spec_k = 1
        try:
            spec_min_accept = float(merged_spec_cfg.get("min_accept", 0.0) or 0.0)
        except Exception:
            spec_min_accept = 0.0
        spec_min_accept = max(0.0, min(1.0, spec_min_accept))
        try:
            spec_disable_after_n = max(1, int(merged_spec_cfg.get("disable_after_n", 1) or 1))
        except Exception:
            spec_disable_after_n = 1
        spec_cache_only = bool(merged_spec_cfg.get("cache_only", False))
        return {
            "lane": lane_up,
            "shape_key": str(shape_key or ""),
            "target_tokens": int(max(1, int(max_tokens or 1))),
            "decode_budget_tokens": int(
                self._vllm_decode_budget_tokens(
                    lane=lane_up,
                    shape_key=str(shape_key or ""),
                    max_tokens=int(max_tokens or 1),
                )
            ),
            "replay_state": replay_state,
            "replay_local": bool(replay_local),
            "snapshot_tier": snapshot_tier,
            "prompt_tokens": int(max(0, prompt_tokens)),
            "queue_latency_ms": float(max(0.0, float(queue_latency_ms or 0.0))),
            "priority": int(vllm_priority) if vllm_priority is not None else -1,
            "spec_enabled": bool(spec_requested),
            "spec_k": int(spec_k if spec_requested else 0),
            "spec_min_accept": float(spec_min_accept if spec_requested else 0.0),
            "spec_disable_after_n": int(spec_disable_after_n if spec_requested else 0),
            "spec_cache_only": bool(spec_cache_only if spec_requested else False),
            "contract_version": 1,
        }

    def _vllm_priority_for_request(self, *, lane: str, shape_key: str) -> Optional[int]:
        if not bool(getattr(self, "_vllm_priority_sched_enabled", False)):
            return None
        lane_up = str(lane).upper()
        base_map = getattr(self, "_vllm_priority_lane_base", {})
        try:
            priority = int(base_map.get(lane_up, base_map.get("MISS", 0)) or 0)
        except Exception:
            priority = 0
        priority = max(0, priority)
        if shape_key:
            lane_decode_ema = float(self._lane_decode_ema_ms.get(lane_up, 0.0) or 0.0)
            shape_decode_ema = float(self._shape_decode_ema_ms.get(shape_key, 0.0) or 0.0)
            if lane_decode_ema > 0.0 and shape_decode_ema > 0.0:
                if shape_decode_ema <= (0.8 * lane_decode_ema):
                    priority = max(0, priority - 1)
                elif shape_decode_ema >= (1.2 * lane_decode_ema):
                    priority = min(100, priority + 1)
        return int(priority)

    def _kernel_policy_allow(self, *, shape_key: str, lane: str) -> tuple[bool, str]:
        if not shape_key:
            return True, "shape_unknown"
        if not self._kernel_policy_enabled:
            return True, "policy_disabled"
        lane_up = str(lane).upper()
        if lane_up.startswith("SPEC_"):
            # Keep speculative behavior deterministic; kernel policy currently tunes
            # only non-spec decode shapes.
            return True, "spec_lane"
        state = self._kernel_policy_state.setdefault(shape_key, {})
        requests = int(state.get("requests", 0.0) or 0)
        requests += 1
        state["requests"] = float(requests)
        now = time.time()
        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        if cooldown_until > now:
            probe_every = max(1, int(self._kernel_policy_probe_every))
            if (requests % probe_every) == 0:
                state["last_decision"] = "probe"
                return True, "probe"
            state["last_decision"] = "cooldown"
            return False, "cooldown"
        state["last_decision"] = "active"
        return True, "active"

    def _update_kernel_policy_state(self, *, shape_key: str, engine_metrics: Dict[str, Any]) -> None:
        if not self._kernel_policy_enabled or not shape_key:
            return
        if not isinstance(engine_metrics, dict):
            return
        kernels = engine_metrics.get("kernels", {}) if isinstance(engine_metrics.get("kernels", {}), dict) else {}
        if not kernels:
            return
        state = self._kernel_policy_state.setdefault(shape_key, {})
        applied = bool(kernels.get("kernels_applied", False))
        if not applied:
            return

        comparable = bool(kernels.get("comparable", False))
        ms_saved = max(0.0, float(kernels.get("ms_saved", 0.0) or 0.0))
        fallback = bool(kernels.get("fallback", False))
        bad_streak = int(state.get("bad_streak", 0.0) or 0)
        now = time.time()

        if fallback:
            bad_streak += 1
        elif comparable and ms_saved >= self._kernel_policy_min_saved_ms:
            bad_streak = 0
            prev_saved = float(state.get("saved_ema", 0.0) or 0.0)
            state["saved_ema"] = ms_saved if prev_saved <= 0.0 else ((0.8 * prev_saved) + (0.2 * ms_saved))
        elif comparable:
            bad_streak += 1

        state["bad_streak"] = float(max(0, bad_streak))
        if bad_streak >= self._kernel_policy_bad_streak_max:
            state["cooldown_until"] = float(now + (self._kernel_policy_cooldown_ms / 1000.0))

    def _decode_cache_key(self, job: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if not self._decode_cache_enabled:
            return None
        org_id = str(job.get("org_id", "default") or "default")
        backend_id = str(job.get("execution_backend_id", "") or job.get("backend_id", "") or "")
        fingerprint_hash = str(job.get("fingerprint_hash", "") or "")
        prompt_hash = str(job.get("prompt_hash", "") or "")
        sampling_hash = str(job.get("sampling_hash", "") or "")
        if not (backend_id and fingerprint_hash and prompt_hash and sampling_hash):
            return None
        return {
            "org_id": org_id,
            "backend_id": backend_id,
            "fingerprint_hash": fingerprint_hash,
            "prompt_hash": prompt_hash,
            "sampling_hash": sampling_hash,
        }

    def _decode_cache_eligible(self, job: Dict[str, Any]) -> bool:
        if not self._decode_cache_enabled:
            return False
        if not self._decode_cache_require_deterministic:
            return True
        det = job.get("deterministic_cfg", {}) if isinstance(job.get("deterministic_cfg", {}), dict) else {}

        def _float(name: str, default: float) -> float:
            try:
                return float(det.get(name, default))
            except Exception:
                return default

        def _int(name: str, default: int) -> int:
            try:
                return int(det.get(name, default))
            except Exception:
                return default

        temperature = _float("temperature", 0.0)
        top_p = _float("top_p", 1.0)
        top_k = _int("top_k", 0)
        seed = det.get("seed", None)
        return (temperature == 0.0) and (top_p >= 1.0) and (top_k in (0, -1)) and (seed is not None)

    def _run_kernel_shadow_baseline(
        self,
        *,
        adapter,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict[str, Any],
        policy: Dict[str, Any],
        mf_snapshot_in: Optional[str],
        artifacts: Dict[str, Any],
        job_id: str,
    ) -> tuple[float, int]:
        shadow_dir = artifacts["job_dir"] / "_kernel_shadow"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_artifacts = {
            "job_dir": str(shadow_dir),
            "engine_metrics": str(shadow_dir / f"{job_id}_engine_metrics.json"),
            "engine_events": str(shadow_dir / f"{job_id}_events.jsonl"),
            "mf_snapshot": str(shadow_dir / f"{job_id}_mf_snapshot.bin"),
            "output": str(shadow_dir / f"{job_id}_output.txt"),
            "log": str(shadow_dir / f"{job_id}_run.log"),
        }
        shadow_engine_cfg = dict(self._engine_cfg)
        shadow_engine_cfg["kernels_enabled"] = "0"
        shadow_engine_cfg["kernel_backend"] = "none"
        shadow_engine_cfg["kernel_verify"] = "0"
        shadow_policy = dict(policy)
        shadow_policy["allow_spec"] = False
        shadow = adapter.run_accel(
            prompt=prompt,
            max_tokens=max_tokens,
            deterministic_cfg=deterministic_cfg,
            policy=shadow_policy,
            artifacts=shadow_artifacts,
            mf_snapshot_in=mf_snapshot_in,
            engine_cfg=shadow_engine_cfg,
        )
        em = shadow.get("engine_metrics", {}) if isinstance(shadow, dict) else {}
        perf = em.get("perf", {}) if isinstance(em, dict) else {}
        decode_ms = float(perf.get("decode_ms", shadow.get("total_ms", 0.0) if isinstance(shadow, dict) else 0.0) or 0.0)
        tokens_out = int(perf.get("tokens_out", 0) or 0)
        return max(0.0, decode_ms), max(0, tokens_out)

    def _update_kernel_timing_estimate(
        self,
        *,
        job: Dict[str, Any],
        run_mode: str,
        engine_metrics: Dict[str, Any],
        shadow_decode_baseline_ms: float = 0.0,
        shadow_tokens_out: int = 0,
    ) -> None:
        if not isinstance(engine_metrics, dict):
            return
        perf = engine_metrics.get("perf", {}) if isinstance(engine_metrics, dict) else {}
        decode_actual = float(perf.get("decode_ms", 0.0) or 0.0)
        tokens_out = int(perf.get("tokens_out", 0) or 0)
        if decode_actual <= 0.0 or tokens_out <= 0:
            return

        spec = engine_metrics.get("spec", {}) if isinstance(engine_metrics.get("spec", {}), dict) else {}
        kernels = engine_metrics.setdefault("kernels", {})
        if not isinstance(kernels, dict):
            return
        kernels_applied = bool(kernels.get("kernels_applied", False))
        spec_enabled = bool(spec.get("enabled", False))
        cache_hit = bool(spec.get("cache_hit", False))
        nonspec_mode = run_mode in ("baseline", "accel") and not spec_enabled and not cache_hit

        kernels["decode_ms_actual"] = float(decode_actual)
        kernels["decode_ms_baseline_est"] = 0.0
        kernels["comparable"] = False
        kernels["comparable_tag"] = "unavailable"
        kernels["baseline_source"] = "none"
        kernels["baseline_key"] = ""
        kernels["comparable_requirements"] = {
            "same_model": False,
            "same_shape": False,
            "same_generation_len": False,
            "same_backend_path": False,
            "same_non_spec_path": bool(nonspec_mode),
        }
        calibration_mode = str(getattr(self, "_decode_calibration_mode", "rolling") or "rolling").strip().lower()
        if calibration_mode not in ("off", "rolling", "shadow", "strict"):
            calibration_mode = "rolling"

        if not nonspec_mode:
            kernels["ms_saved"] = 0.0
            return

        if calibration_mode == "off":
            kernels["ms_saved"] = 0.0
            kernels["comparable_tag"] = "calibration_off"
            return

        key = self._kernel_baseline_key(job=job, tokens_out=tokens_out)
        kernels["baseline_key"] = str(key)
        reqs = kernels.get("comparable_requirements", {})
        if isinstance(reqs, dict):
            reqs["same_model"] = True
            reqs["same_shape"] = True
            reqs["same_generation_len"] = True
            reqs["same_backend_path"] = True
            reqs["same_non_spec_path"] = True
            kernels["comparable_requirements"] = reqs
        if not kernels_applied:
            if calibration_mode == "rolling":
                prev = self._kernel_decode_baseline_ema.get(key)
                self._kernel_decode_baseline_ema[key] = (
                    decode_actual if prev is None else (0.8 * prev) + (0.2 * decode_actual)
                )
                kernels["decode_ms_baseline_est"] = float(self._kernel_decode_baseline_ema[key])
                kernels["comparable_tag"] = "baseline_sample"
                kernels["baseline_source"] = "rolling_sample"
            else:
                kernels["comparable_tag"] = "baseline_sample_ignored"
            kernels["ms_saved"] = 0.0
            return

        if run_mode != "accel":
            kernels["ms_saved"] = 0.0
            kernels["comparable_tag"] = "non_accel_path"
            return

        baseline_est = 0.0
        comparable = False
        comparable_tag = "baseline_missing"
        if shadow_decode_baseline_ms > 0.0 and shadow_tokens_out == tokens_out:
            baseline_est = float(shadow_decode_baseline_ms)
            comparable = True
            comparable_tag = "shadow_baseline"
            kernels["baseline_source"] = "shadow"
            if calibration_mode == "rolling":
                prev = self._kernel_decode_baseline_ema.get(key)
                self._kernel_decode_baseline_ema[key] = (
                    baseline_est if prev is None else (0.8 * prev) + (0.2 * baseline_est)
                )
        elif shadow_decode_baseline_ms > 0.0 and shadow_tokens_out != tokens_out:
            reqs = kernels.get("comparable_requirements", {})
            if isinstance(reqs, dict):
                reqs["same_generation_len"] = False
                kernels["comparable_requirements"] = reqs
            comparable_tag = "shadow_tokens_mismatch"
        elif calibration_mode == "rolling":
            ref = self._kernel_decode_baseline_ema.get(key)
            if ref is not None and ref > 0.0:
                baseline_est = float(ref)
                comparable = True
                comparable_tag = "rolling_calibrated"
                kernels["baseline_source"] = "rolling"
            else:
                comparable_tag = "rolling_baseline_missing"
        elif calibration_mode in ("shadow", "strict"):
            comparable = False
            comparable_tag = "shadow_required"

        kernels["decode_ms_baseline_est"] = float(baseline_est)
        kernels["comparable"] = bool(comparable)
        kernels["comparable_tag"] = comparable_tag
        kernels["ms_saved"] = float(max(0.0, baseline_est - decode_actual) if comparable else 0.0)

    def _compose_metrics(
        self,
        job: Dict[str, Any],
        result: Dict[str, Any],
        lane: str,
        queue_latency_ms: float,
        started_at: str,
        finished_at: str,
        caps=None,
    ) -> Dict[str, Any]:
        engine_metrics = result.get("engine_metrics") or {}
        amf = engine_metrics.get("amf", {})
        mf = engine_metrics.get("mf", {})
        perf = engine_metrics.get("perf", {})

        ids = {
            "job_id": job["job_id"],
            "parent_job_id": job.get("parent_job_id"),
            "created_at": job.get("created_at"),
            "started_at": started_at,
            "finished_at": finished_at,
            "org_id": job.get("org_id", "default"),
            "tenant_id": job.get("tenant_id", job.get("org_id", "default")),
            "request_id": job.get("request_id", ""),
        }
        exec_backend_id = str(job.get("execution_backend_id", "") or job.get("backend_id", ""))
        exec_backend_version = str(job.get("execution_backend_version", "") or job.get("backend_version", "v1"))
        exec_model = job.get("execution_model") if isinstance(job.get("execution_model"), dict) else None
        exec_fingerprint = job.get("execution_fingerprint") if isinstance(job.get("execution_fingerprint"), dict) else None
        model_src = exec_model if exec_model else (job.get("model", {}) if isinstance(job.get("model"), dict) else {})
        fp_src = exec_fingerprint if exec_fingerprint else (job.get("fingerprint", {}) if isinstance(job.get("fingerprint"), dict) else {})
        backend = {
            "backend_id": exec_backend_id,
            "backend_version": exec_backend_version,
        }
        model = {
            "model_id": str(model_src.get("model_id", "")),
            "model_hash": str(fp_src.get("model_hash", "")),
            "tokenizer_hash": str(fp_src.get("tokenizer_hash", "")),
        }
        input_block = {
            "prompt_hash": job["prompt_hash"],
            "sampling_hash": job["sampling_hash"],
            "prompt_tokens": job.get("prompt_tokens", 0),
        }
        scheduling = {
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "gpu_id": self.gpu_id,
            "lane": lane,
            "queue_latency_ms": queue_latency_ms,
        }
        perf_block = {
            "tokens_out": perf.get("tokens_out", 0),
            "total_ms": perf.get("total_ms", result.get("total_ms", 0.0)),
            "prefill_ms": perf.get("prefill_ms", 0.0),
            "decode_ms": perf.get("decode_ms", 0.0),
            "avg_tps": perf.get("avg_tps", 0.0),
        }
        spec_block = {
            "supported": bool(getattr(caps, "draft_supported", False) and (getattr(caps, "verify_tokens", False) or getattr(caps, "logits_access", False))) if caps is not None else False,
            "enabled": bool(engine_metrics.get("spec", {}).get("enabled", False)) if isinstance(engine_metrics, dict) else False,
            "k": int(engine_metrics.get("spec", {}).get("k", 0) or 0) if isinstance(engine_metrics, dict) else 0,
            "proposed_tokens": int(engine_metrics.get("spec", {}).get("proposed_tokens", 0) or 0) if isinstance(engine_metrics, dict) else 0,
            "accepted_tokens": int(engine_metrics.get("spec", {}).get("accepted_tokens", 0) or 0) if isinstance(engine_metrics, dict) else 0,
            "acceptance_rate": float(engine_metrics.get("spec", {}).get("acceptance_rate", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "verify_ms": float(engine_metrics.get("spec", {}).get("verify_ms", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "draft_ms": float(engine_metrics.get("spec", {}).get("draft_ms", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "overhead_ms": float(engine_metrics.get("spec", {}).get("overhead_ms", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "baseline_total_ms": float(engine_metrics.get("spec", {}).get("baseline_total_ms", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "net_saved_ms": float(engine_metrics.get("spec", {}).get("net_saved_ms", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "saved_ms": float(engine_metrics.get("spec", {}).get("saved_ms", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "roi": float(engine_metrics.get("spec", {}).get("roi", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "speedup_est": float(engine_metrics.get("spec", {}).get("speedup_est", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "cache_hit": bool(engine_metrics.get("spec", {}).get("cache_hit", False)) if isinstance(engine_metrics, dict) else False,
            "cache_ms": float(engine_metrics.get("spec", {}).get("cache_ms", 0.0) or 0.0) if isinstance(engine_metrics, dict) else 0.0,
            "cache_only": bool(engine_metrics.get("spec", {}).get("cache_only", False)) if isinstance(engine_metrics, dict) else False,
            "disable_reason": str(engine_metrics.get("spec", {}).get("disable_reason", "")) if isinstance(engine_metrics, dict) else "",
            "policy_reason": str(engine_metrics.get("spec", {}).get("policy_reason", "")) if isinstance(engine_metrics, dict) else "",
            "shape_key": str(engine_metrics.get("spec", {}).get("shape_key", "")) if isinstance(engine_metrics, dict) else "",
        }
        lane_up = str(lane).upper()
        if lane_up.startswith("SPEC_") and not spec_block["enabled"]:
            scheduling["lane"] = lane_up.replace("SPEC_", "", 1)
        engine_block = {
            "mode": str(engine_metrics.get("engine", {}).get("mode", "baseline")) if isinstance(engine_metrics, dict) else "baseline",
            "accel_enabled": bool(engine_metrics.get("engine", {}).get("accel_enabled", False)) if isinstance(engine_metrics, dict) else False,
            "cuda_device": int(engine_metrics.get("engine", {}).get("cuda_device", self.gpu_id) or self.gpu_id) if isinstance(engine_metrics, dict) else self.gpu_id,
            "kv_layout_version": str(engine_metrics.get("engine", {}).get("kv_layout_version", self._engine_cfg.get("kv_layout_version", "v1"))) if isinstance(engine_metrics, dict) else str(self._engine_cfg.get("kv_layout_version", "v1")),
        }
        kernels_src = engine_metrics.get("kernels", {}) if isinstance(engine_metrics.get("kernels", {}), dict) else {}
        comparable_requirements_src = kernels_src.get("comparable_requirements", {})
        comparable_requirements = {}
        if isinstance(comparable_requirements_src, dict):
            comparable_requirements = {
                str(k): bool(v)
                for k, v in comparable_requirements_src.items()
            }
        kernels_block = {
            "enabled": bool(kernels_src.get("enabled", False)),
            "backend": str(kernels_src.get("backend", self._kernel_ctx.backend)),
            "kernels_applied": bool(kernels_src.get("kernels_applied", False)),
            "verify": bool(kernels_src.get("verify", False)),
            "verify_ok": bool(kernels_src.get("verify_ok", True)),
            "fallback": bool(kernels_src.get("fallback", False)),
            "decode_ms_actual": float(kernels_src.get("decode_ms_actual", perf_block["decode_ms"]) or perf_block["decode_ms"]),
            "decode_ms_baseline_est": float(kernels_src.get("decode_ms_baseline_est", 0.0) or 0.0),
            "baseline_key": str(kernels_src.get("baseline_key", "")),
            "baseline_source": str(kernels_src.get("baseline_source", "")),
            "comparable": bool(kernels_src.get("comparable", False)),
            "comparable_tag": str(kernels_src.get("comparable_tag", "")),
            "comparable_requirements": comparable_requirements,
            "policy_reason": str(kernels_src.get("policy_reason", "")),
            "shape_key": str(kernels_src.get("shape_key", "")),
            "calibration_mode": str(kernels_src.get("calibration_mode", "")),
            "ms_saved": float(kernels_src.get("ms_saved", 0.0) or 0.0),
        }
        if (
            bool(getattr(self, "_vllm_amf_infer_by_lane", True))
            and str(exec_backend_id) in ("vllm", "vllm_openai")
            and isinstance(amf, dict)
        ):
            ema_store = getattr(self, "_vllm_amf_miss_ema", None)
            if not isinstance(ema_store, dict):
                ema_store = {}
                self._vllm_amf_miss_ema = ema_store

            org_for_key = str(job.get("org_id", "default") or "default")
            fp_for_key = str(job.get("fingerprint_hash", "") or job.get("prompt_hash", "") or "")
            baseline_key = f"{org_for_key}:{fp_for_key}:{exec_backend_id}"
            total_ms_now = float(perf_block.get("total_ms", 0.0) or 0.0)
            miss_ema = float(ema_store.get(baseline_key, 0.0) or 0.0)

            if lane_up in ("MISS", "SPEC_MISS") and total_ms_now > 0.0:
                miss_ema = total_ms_now if miss_ema <= 0.0 else ((0.8 * miss_ema) + (0.2 * total_ms_now))
                ema_store[baseline_key] = miss_ema

            amf_supported_raw = bool(amf.get("supported", False))
            amf_decision_raw = str(amf.get("decision", "") or "").strip().lower()
            if (not amf_supported_raw or amf_decision_raw in ("", "unavailable")) and lane_up in (
                "HIT",
                "SPEC_HIT",
                "MISS",
                "SPEC_MISS",
            ):
                inferred_decision = "hit" if lane_up in ("HIT", "SPEC_HIT") else "miss"
                amf["supported"] = True
                amf["decision"] = inferred_decision
                amf["estimate_method"] = str(amf.get("estimate_method", "lane_miss_ema"))
                prompt_tokens_est = max(1, int(input_block.get("prompt_tokens", 0) or 0))

                if inferred_decision == "hit" and miss_ema > 0.0:
                    inferred_saved_ms = max(0.0, miss_ema - total_ms_now)
                    inferred_skip = min(1.0, max(0.0, inferred_saved_ms / miss_ema))
                    amf["baseline_prefix_ms"] = float(
                        max(float(amf.get("baseline_prefix_ms", 0.0) or 0.0), inferred_saved_ms)
                    )
                    amf["saved_ms"] = float(max(float(amf.get("saved_ms", 0.0) or 0.0), inferred_saved_ms))
                    amf["skip_ratio"] = float(max(float(amf.get("skip_ratio", 0.0) or 0.0), inferred_skip))
                    inferred_skipped_tokens = int(round(prompt_tokens_est * float(amf.get("skip_ratio", 0.0) or 0.0)))
                    amf["skipped_tokens"] = int(max(int(amf.get("skipped_tokens", 0) or 0), inferred_skipped_tokens))
                    amf["prefix_len"] = int(max(int(amf.get("prefix_len", 0) or 0), int(amf.get("skipped_tokens", 0) or 0)))
                    amf["roi"] = float(
                        max(float(amf.get("roi", 0.0) or 0.0), inferred_saved_ms / max(1.0, total_ms_now))
                    )
                else:
                    if inferred_decision == "miss":
                        amf["skip_ratio"] = 0.0
                        amf["skipped_tokens"] = 0
                        amf["prefix_len"] = 0
                    amf["saved_ms"] = float(max(0.0, float(amf.get("saved_ms", 0.0) or 0.0)))
                    amf["baseline_prefix_ms"] = float(max(0.0, float(amf.get("baseline_prefix_ms", 0.0) or 0.0)))
                    amf["roi"] = float(max(0.0, float(amf.get("roi", 0.0) or 0.0)))
        prefill_saved_ms = max(0.0, float(amf.get("saved_ms", 0.0) or 0.0))
        kernels_credit_allowed = bool(kernels_block.get("kernels_applied", False))
        # Trust kernel savings only when kernels actually ran. If comparability is
        # explicitly known, require it too to avoid non-comparable attribution.
        if "comparable" in kernels_src:
            kernels_credit_allowed = kernels_credit_allowed and bool(kernels_block.get("comparable", False))
        if comparable_requirements:
            kernels_credit_allowed = kernels_credit_allowed and all(bool(v) for v in comparable_requirements.values())
        kernels_saved_ms = max(0.0, float(kernels_block.get("ms_saved", 0.0) or 0.0)) if kernels_credit_allowed else 0.0
        kernels_block["ms_saved"] = float(kernels_saved_ms)
        spec_saved_raw = max(0.0, float(spec_block.get("saved_ms", 0.0) or 0.0))
        spec_saved_ms = max(0.0, spec_saved_raw - kernels_saved_ms)
        savings_block = {
            "prefill_saved_ms": prefill_saved_ms,
            "spec_saved_ms": spec_saved_ms,
            "kernels_saved_ms": kernels_saved_ms,
            "total_saved_ms": prefill_saved_ms + spec_saved_ms + kernels_saved_ms,
        }
        cap_amf_supported = False
        if caps is not None:
            cap_amf_supported = bool(getattr(caps, "kv_replay", False) and getattr(caps, "deterministic_seeding", False))

        amf_supported = cap_amf_supported
        if "supported" in amf:
            amf_supported = bool(amf.get("supported", False))

        mf_supported = cap_amf_supported
        if "supported" in mf:
            mf_supported = bool(mf.get("supported", False))

        snapshot_id = mf.get("snapshot_id", "")
        if not snapshot_id and job.get("job_id"):
            snapshot_id = job["job_id"]

        errors = engine_metrics.get("errors", [])
        if not errors:
            errors = []
        engine_errors = result.get("engine_errors", [])
        if isinstance(engine_errors, list):
            for item in engine_errors:
                if isinstance(item, str) and item.strip():
                    errors.append(item.strip())
        if cap_amf_supported and not engine_metrics:
            errors.append("engine_metrics_missing_for_capable_backend")
        if int(result.get("exit_code", 0)) != 0:
            errors = list(errors) if isinstance(errors, list) else []
            errors.append(f"exit_code:{result.get('exit_code')}")

        health = engine_metrics.get("health", {})
        if not health:
            hit_rate = 1.0 if amf.get("decision") == "hit" else 0.0
            roi = float(amf.get("roi", 0.0) or 0.0)
            pressure = float(mf.get("eviction_pressure", 1.0) or 1.0)
            stability = "ok" if not errors else "degraded"
            health = {
                "hit_rate": hit_rate,
                "roi_ema": roi,
                "pressure": pressure,
                "stability": stability,
                "line": f"[KORITH_HEALTH] hit_rate={hit_rate:.3f} roi_ema={roi:.3f} pressure={pressure:.3f} stability={stability}",
            }

        return {
            "ids": ids,
            "backend": backend,
            "model": model,
            "input": input_block,
            "scheduling": scheduling,
            "amf": {
                "supported": amf_supported,
                "decision": amf.get("decision", "unavailable" if not amf_supported else "miss"),
                "prefix_len": amf.get("prefix_len", 0),
                "skipped_tokens": amf.get("skipped_tokens", 0),
                "skip_ratio": amf.get("skip_ratio", 0.0),
                "restore_ms": amf.get("restore_ms", 0.0),
                "baseline_prefix_ms": amf.get("baseline_prefix_ms", 0.0),
                "saved_ms": amf.get("saved_ms", 0.0),
                "roi": amf.get("roi", 0.0),
            },
            "mf": {
                "supported": mf_supported,
                "min_admit_roi": mf.get("min_admit_roi", 0.0),
                "eviction_pressure": mf.get("eviction_pressure", 1.0),
                "replay_disable_mask": mf.get("replay_disable_mask", 0),
                "cooldown_ms": mf.get("cooldown_ms", 0),
                "snapshot_id": snapshot_id,
            },
            "engine": engine_block,
            "kernels": kernels_block,
            "spec": spec_block,
            "savings": savings_block,
            "perf": perf_block,
            "health": health,
            "errors": errors,
        }

    def _update_health(self, metrics: Dict[str, Any], shape_key: str = "") -> None:
        org_id = metrics.get("ids", {}).get("org_id", "default")
        tenant_id = metrics.get("ids", {}).get("tenant_id", org_id)
        amf = metrics.get("amf", {})
        scheduling = metrics.get("scheduling", {})
        perf = metrics.get("perf", {})
        spec = metrics.get("spec", {})

        GLOBAL_METRICS.inc("jobs_processed_total", 1.0, labels={"org_id": org_id})
        if amf.get("supported"):
            GLOBAL_METRICS.inc("amf_seen_total", 1.0, labels={"org_id": org_id})
            GLOBAL_METRICS.inc("amf_hit_total", 1.0 if amf.get("decision") == "hit" else 0.0, labels={"org_id": org_id})
            self._amf_seen_by_org[tenant_id] = self._amf_seen_by_org.get(tenant_id, 0.0) + 1.0
            if amf.get("decision") == "hit":
                self._amf_hits_by_org[tenant_id] = self._amf_hits_by_org.get(tenant_id, 0.0) + 1.0
            seen = self._amf_seen_by_org.get(tenant_id, 0.0)
            hits = self._amf_hits_by_org.get(tenant_id, 0.0)
            GLOBAL_METRICS.set_gauge("amf_hit_rate", hits / seen if seen > 0 else 0.0, labels={"org_id": org_id})
            misses = max(0.0, seen - hits)
            try:
                cache_entries = int(amf.get("cache_entries", self._amf_cache_entries) or self._amf_cache_entries)
            except Exception:
                cache_entries = self._amf_cache_entries
            cache_bytes = int(amf.get("cache_bytes", 0) or 0)
            print(
                "[AMF_STATS_TENANT] tenant={tenant} hits={hits} misses={misses} entries={entries} bytes={bytes}".format(
                    tenant=str(tenant_id),
                    hits=int(hits),
                    misses=int(misses),
                    entries=int(max(0, cache_entries)),
                    bytes=int(max(0, cache_bytes)),
                )
            )
        total_seen = sum(float(v) for v in self._amf_seen_by_org.values())
        total_hits = sum(float(v) for v in self._amf_hits_by_org.values())
        if total_seen > 0.0:
            self._amf_hit_rate = max(0.0, min(1.0, total_hits / total_seen))
        else:
            self._amf_hit_rate = max(0.0, min(1.0, float(amf.get("hit_rate", 0.0) or 0.0)))
        try:
            cache_entries = int(amf.get("cache_entries", self._amf_cache_entries) or self._amf_cache_entries)
        except Exception:
            cache_entries = self._amf_cache_entries
        self._amf_cache_entries = max(0, cache_entries)
        try:
            cache_bytes = int(amf.get("cache_bytes", self._amf_cache_bytes) or self._amf_cache_bytes)
        except Exception:
            cache_bytes = self._amf_cache_bytes
        self._amf_cache_bytes = max(0, cache_bytes)
        try:
            warm_ratio = float(amf.get("warm_ratio", self._amf_warm_ratio) or self._amf_warm_ratio)
        except Exception:
            warm_ratio = self._amf_warm_ratio
        self._amf_warm_ratio = max(0.0, min(1.0, warm_ratio))
        self._amf_prewarm_complete = bool(amf.get("prewarm_complete", self._amf_prewarm_complete))
        self._amf_ready = bool(self._amf_prewarm_complete or self._amf_hit_rate > 0.5)
        savings = metrics.get("savings", {}) if isinstance(metrics.get("savings", {}), dict) else {}
        saved_ms = float(savings.get("total_saved_ms", 0.0) or 0.0)
        self._tenant_saved_ms[tenant_id] = self._tenant_saved_ms.get(tenant_id, 0.0) + max(0.0, saved_ms)
        GLOBAL_METRICS.set_gauge("amf_warm_ratio", self._amf_warm_ratio, labels={"org_id": org_id})

        skip_ratio = float(amf.get("skip_ratio", 0.0) or 0.0)
        roi = float(amf.get("roi", 0.0) or 0.0)
        queue_latency_ms = float(scheduling.get("queue_latency_ms", 0.0) or 0.0)
        lane = str(scheduling.get("lane", "MISS") or "MISS").upper()
        decode_ms = float(perf.get("decode_ms", 0.0) or 0.0)
        if decode_ms > 0.0:
            prev_decode = float(self._lane_decode_ema_ms.get(lane, decode_ms) or decode_ms)
            self._lane_decode_ema_ms[lane] = (0.9 * prev_decode) + (0.1 * decode_ms)
            if shape_key:
                prev_shape_decode = float(self._shape_decode_ema_ms.get(shape_key, decode_ms) or decode_ms)
                self._shape_decode_ema_ms[shape_key] = (0.9 * prev_shape_decode) + (0.1 * decode_ms)

        self._avg_queue_latency_ms = (0.9 * self._avg_queue_latency_ms) + (0.1 * queue_latency_ms)

        GLOBAL_METRICS.set_gauge("avg_skip_ratio", skip_ratio)
        GLOBAL_METRICS.set_gauge("avg_roi", roi)
        GLOBAL_METRICS.set_gauge("roi_ema", roi)
        GLOBAL_METRICS.set_gauge("queue_latency_ms", queue_latency_ms)
        GLOBAL_METRICS.set_gauge("avg_queue_latency_ms", self._avg_queue_latency_ms)
        GLOBAL_METRICS.set_gauge("worker_health", 1.0, labels={"worker_id": self.worker_id})
        GLOBAL_METRICS.set_gauge("gpu_utilization", 0.0, labels={"gpu_id": str(self.gpu_id)})
        GLOBAL_METRICS.set_gauge("gpu_utilization_percent", 0.0, labels={"gpu_id": str(self.gpu_id)})
        GLOBAL_METRICS.set_gauge("decode_tokens_per_second", float(perf.get("avg_tps", 0.0) or 0.0), labels={"org_id": org_id})
        GLOBAL_METRICS.set_gauge("spec_acceptance_rate", float(spec.get("acceptance_rate", 0.0) or 0.0), labels={"org_id": org_id})
        GLOBAL_METRICS.set_gauge("spec_roi", float(spec.get("roi", 0.0) or 0.0), labels={"org_id": org_id})
        GLOBAL_METRICS.set_gauge("spec_roi_ema", float(self._spec_roi_ema), labels={"org_id": org_id})
        GLOBAL_METRICS.set_gauge("spec_speedup_est", float(spec.get("speedup_est", 0.0) or 0.0), labels={"org_id": org_id})
        GLOBAL_METRICS.set_gauge("org_request_rate", 0.0, labels={"org_id": org_id})
        GLOBAL_METRICS.set_gauge("org_token_rate", 0.0, labels={"org_id": org_id})

    def amf_health(self) -> Dict[str, Any]:
        hit_rate = float(max(0.0, min(1.0, self._amf_hit_rate)))
        warm_ratio = float(max(0.0, min(1.0, self._amf_warm_ratio)))
        ready = bool(self._amf_prewarm_complete or hit_rate > 0.5)
        return {
            "cache_entries": int(max(0, self._amf_cache_entries)),
            "cache_bytes": int(max(0, self._amf_cache_bytes)),
            "hit_rate": hit_rate,
            "warm_ratio": warm_ratio,
            "ready": ready,
        }

    def amf_metrics_snapshot(self) -> Dict[str, Any]:
        tenants: Dict[str, Dict[str, float]] = {}
        for tenant, seen in self._amf_seen_by_org.items():
            hits = float(self._amf_hits_by_org.get(tenant, 0.0) or 0.0)
            tenants[str(tenant)] = {
                "requests": float(seen),
                "hits": float(hits),
                "hit_rate": float(hits / seen) if seen > 0 else 0.0,
                "savings_ms": float(self._tenant_saved_ms.get(tenant, 0.0) or 0.0),
            }
        return {
            "node_id": self.node_id,
            "worker_id": self.worker_id,
            "requests_total": float(sum(self._amf_seen_by_org.values())),
            "hit_rate": float(self._amf_hit_rate),
            "entries": int(self._amf_cache_entries),
            "storage_bytes": int(self._amf_cache_bytes),
            "warmth": float(self._amf_warm_ratio),
            "savings_ms_total": float(sum(self._tenant_saved_ms.values())),
            "tenants": tenants,
        }

    def _append_event(self, events_path: Path, event_type: str, payload: Dict[str, Any]) -> None:
        evt = {"type": event_type, "ts": utc_now(), "payload": payload}
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
