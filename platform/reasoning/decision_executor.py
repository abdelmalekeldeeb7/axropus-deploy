"""
Component 3: Decision Executor
==============================
Translates DecisionVector outputs from the ReasoningModel into actual AMF actions,
enforcing safety overrides that guarantee the reasoning model can ONLY improve
decisions, never worsen them.

Safety rules (applied in order)
---------------------------------
1. ROI gate       — if roi_ema_norm < roi_threshold, override cache_admission=True → False.
                    The model cannot force caching when AMF's ROI signal says no.
2. Hot protection — if hot_entry_ratio > hot_ratio_threshold AND eviction_pressure < pressure_threshold,
                    block model-requested evictions. AMF pressure overrides hot protection.
3. Throttle cap   — if hit_rate >= healthy_hit_rate, cap throttle_back_pressure at
                    throttle_max_when_healthy. The model cannot throttle a healthy cache.

Every decision and every override is logged and returned in ExecutionResult so
Component 4 (RewardCalculator) can score outcomes.

Usage
-----
    executor = DecisionExecutor(
        coordinator_client=AmfCoordinatorClient("http://localhost:8500"),
        spec_controller=SpecDecodeController(),
    )

    result = executor.execute(
        dv=model.decide(tensor),
        context=ExecutionContext(
            tenant_id="t-1", node_id="gpu-0", worker_id="w-1", tensor=tensor
        ),
    )

    # Route decision and throttle are returned for the orchestrator to consume
    print(result.admitted, result.throttle_applied, result.overrides)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics_collector import Slot
from .reasoning_model import DecisionVector, UnifiedDecisionVector

log = logging.getLogger(__name__)

# ── override reason strings (consumed by RewardCalculator) ───────────────────
OVERRIDE_ROI_GATE              = "roi_gate:admission_blocked"
OVERRIDE_HOT_PROTECT           = "hot_protect:eviction_blocked"
OVERRIDE_THROTTLE_CAP          = "throttle_cap:healthy_cache"
OVERRIDE_DEPTH_CAP             = "depth_cap_low_acceptance"
OVERRIDE_ENTROPY_GUARD         = "entropy_guard_high_entropy"
OVERRIDE_KV_COMPRESS_GUARD     = "kv_compression_guard_util_low"
OVERRIDE_RUST_VETO             = "rust_scheduler_veto"
OVERRIDE_CROSS_NODE_LOAD_HIGH  = "cross_node_blocked:local_load_high"
OVERRIDE_REPLICATE_SKIPPED     = "replicate_skipped:no_coordinator_support"


# ── context ──────────────────────────────────────────────────────────────────

@dataclass
class ExecutionContext:
    """AMF runtime context for one executor cycle."""
    tenant_id: str
    node_id: str
    worker_id: str
    tensor: np.ndarray  # (32,) float32 from MetricsCollector.get_metrics_tensor()


# ── result ───────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """
    Outcome of one executor cycle.

    All fields are consumed by the Orchestrator (routing, throttle) and by the
    RewardCalculator (overrides, admitted, eviction_sent, elapsed_ms).
    """
    admitted: bool               # final cache admission decision after safety gating
    eviction_sent: bool          # eviction request was accepted by coordinator
    eviction_blocked: bool       # eviction was blocked by hot-protection override
    pre_warm_queued: int         # number of pre-warm lookups dispatched
    spec_k_applied: Optional[int]  # spec decode k that was set (None = unchanged)
    throttle_applied: float      # effective throttle back-pressure in [0, 1]
    overrides: List[str]         # override reason strings for reward calculator
    elapsed_ms: float            # wall-clock time to execute all actions
    decision: Dict[str, Any]     # original DecisionVector serialised with to_dict()
    # fleet action outcomes
    replication_sent: List[str]  # node_ids replication was dispatched to
    evict_hints_sent: List[str]  # node_ids eviction hints were sent to
    cross_node_restore: Optional[str]  # node_id used for cross-node restore, or None


# ── executor ─────────────────────────────────────────────────────────────────

class DecisionExecutor:
    """
    Applies a DecisionVector to AMF with mandatory safety overrides.

    The reasoning model can only improve AMF decisions, never worsen them.
    Every override is logged and surfaced in ExecutionResult.overrides.

    Parameters
    ----------
    coordinator_client : AmfCoordinatorClient
        Used for evict() and lookup() (pre-warm). Typed loosely for testability.
    spec_controller : SpecDecodeController | None
        If provided, executor updates enabled/k from the decision vector.
    roi_threshold : float
        Minimum roi_ema_norm to permit cache admission. Below this, ROI gate wins.
    hot_ratio_threshold : float
        Hot-entry ratio above which evictions are protected (unless under pressure).
    pressure_threshold : float
        Eviction pressure above which AMF overrides hot protection and allows eviction.
    throttle_max_when_healthy : float
        Maximum throttle the model can apply when cache hit rate is healthy.
    healthy_hit_rate : float
        Hit rate above which the cache is considered "healthy" for throttle capping.
    """

    def __init__(
        self,
        coordinator_client,
        spec_controller=None,
        roi_threshold: float = 0.3,
        hot_ratio_threshold: float = 0.60,
        pressure_threshold: float = 0.50,
        throttle_max_when_healthy: float = 0.10,
        healthy_hit_rate: float = 0.95,
    ) -> None:
        self._coordinator = coordinator_client
        self._spec = spec_controller
        self._roi_threshold = float(roi_threshold)
        self._hot_ratio_threshold = float(hot_ratio_threshold)
        self._pressure_threshold = float(pressure_threshold)
        self._throttle_max_when_healthy = float(throttle_max_when_healthy)
        self._healthy_hit_rate = float(healthy_hit_rate)

    # ── main API ─────────────────────────────────────────────────────────────

    def execute(
        self,
        dv: Any,   # DecisionVector or UnifiedDecisionVector
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Apply the DecisionVector to AMF.

        Extracts key signals from context.tensor, applies safety overrides in
        order, then dispatches coordinator calls for evictions and pre-warm lookups,
        and updates the spec decode controller.

        Parameters
        ----------
        dv : DecisionVector
            Output from ReasoningModel.decide() (or its safe fallback).
        context : ExecutionContext
            Current runtime context including the 32-element metrics tensor.

        Returns
        -------
        ExecutionResult
        """
        t0 = time.perf_counter()
        overrides: List[str] = []
        tensor = context.tensor

        # Extract safety signals from tensor
        roi_ema_norm   = _slot(tensor, Slot.ROI_EMA_NORM)
        hot_ratio      = _slot(tensor, Slot.HOT_ENTRY_RATIO)
        hit_rate       = _slot(tensor, Slot.HIT_RATE)
        evict_pressure = _slot(tensor, Slot.EVICTION_PRESSURE)

        # ── 1. ROI gate on cache admission ───────────────────────────────────
        admitted = dv.cache_admission
        if admitted and roi_ema_norm < self._roi_threshold:
            admitted = False
            overrides.append(OVERRIDE_ROI_GATE)
            log.info(
                "[executor] roi_gate: admission blocked "
                "(roi_ema_norm=%.3f < threshold=%.3f)",
                roi_ema_norm,
                self._roi_threshold,
            )

        # ── 2. Hot prefix protection on evictions ─────────────────────────────
        eviction_blocked = False
        eviction_sent = False

        if dv.eviction_target is not None:
            # Allow eviction if cache is not hot OR AMF pressure demands it
            hot_and_stable = (
                hot_ratio > self._hot_ratio_threshold
                and evict_pressure < self._pressure_threshold
            )
            if hot_and_stable:
                eviction_blocked = True
                overrides.append(OVERRIDE_HOT_PROTECT)
                log.info(
                    "[executor] hot_protect: eviction blocked for %s "
                    "(hot_ratio=%.3f > %.3f, evict_pressure=%.3f < %.3f)",
                    dv.eviction_target,
                    hot_ratio,
                    self._hot_ratio_threshold,
                    evict_pressure,
                    self._pressure_threshold,
                )
            else:
                eviction_sent = bool(
                    self._coordinator.evict(
                        prefix_hash=dv.eviction_target,
                        tenant_id=context.tenant_id,
                    )
                )
                log.info(
                    "[executor] evict: hash=%s sent=%s "
                    "(hot_ratio=%.3f, evict_pressure=%.3f)",
                    dv.eviction_target,
                    eviction_sent,
                    hot_ratio,
                    evict_pressure,
                )

        # ── 3. Throttle cap on healthy caches ─────────────────────────────────
        throttle = dv.throttle_back_pressure
        if hit_rate >= self._healthy_hit_rate and throttle > self._throttle_max_when_healthy:
            throttle = self._throttle_max_when_healthy
            overrides.append(OVERRIDE_THROTTLE_CAP)
            log.info(
                "[executor] throttle_cap: %.3f → %.3f (hit_rate=%.3f >= %.3f)",
                dv.throttle_back_pressure,
                throttle,
                hit_rate,
                self._healthy_hit_rate,
            )

        # ── 4. Decode safety rules (UnifiedDecisionVector only) ──────────────
        # These rules apply when the dv has decode fields. They run before spec
        # decode is committed — each rule can further constrain the decode decision.
        # The reasoning model can ONLY improve decode, never make it more aggressive
        # than the safety bounds allow.
        is_unified = isinstance(dv, UnifiedDecisionVector)
        if is_unified:
            acceptance_ema    = _slot(tensor, Slot.ACCEPTANCE_EMA)
            entropy_ema       = _slot(tensor, Slot.ENTROPY_EMA)
            kv_cache_util     = _slot(tensor, Slot.KV_CACHE_UTILIZATION)
            lane_hold         = _slot(tensor, Slot.LANE_HOLD_REMAINING)

            # 4a. depth_cap: low acceptance → cap spec depth at 3
            if dv.adaptive_depth_override and acceptance_ema < 0.50:
                dv = UnifiedDecisionVector(**{**dv.__dict__, "spec_decode_depth": min(dv.spec_decode_depth, 3)})
                overrides.append(OVERRIDE_DEPTH_CAP)
                log.info(
                    "[executor] depth_cap: spec_decode_depth capped at 3 "
                    "(acceptance_ema=%.3f < 0.50)", acceptance_ema
                )

            # 4b. entropy_guard: high entropy → disable spec decode entirely
            if entropy_ema > 0.70:
                dv = UnifiedDecisionVector(**{**dv.__dict__, "spec_decode_enable": False, "spec_mode": "disabled"})
                overrides.append(OVERRIDE_ENTROPY_GUARD)
                log.info(
                    "[executor] entropy_guard: spec disabled (entropy_ema=%.3f > 0.70)",
                    entropy_ema,
                )

            # 4c. kv_compression_guard: cache not under pressure → block compression
            if dv.kv_compression_enable and kv_cache_util < 0.80:
                dv = UnifiedDecisionVector(**{**dv.__dict__, "kv_compression_enable": False})
                overrides.append(OVERRIDE_KV_COMPRESS_GUARD)
                log.info(
                    "[executor] kv_compression_guard: compression blocked "
                    "(kv_cache_util=%.3f < 0.80)", kv_cache_util,
                )

            # 4d. rust_scheduler_veto: Rust has held a lane > 16 steps → it knows best
            if dv.adaptive_depth_override and lane_hold > 16.0:
                dv = UnifiedDecisionVector(**{**dv.__dict__, "adaptive_depth_override": False})
                overrides.append(OVERRIDE_RUST_VETO)
                log.info(
                    "[executor] rust_scheduler_veto: adaptive_depth_override cleared "
                    "(lane_hold=%.0f > 16 steps)", lane_hold,
                )

        # ── 5. Spec decode settings ───────────────────────────────────────────
        spec_k_applied: Optional[int] = None
        spec_decode_enable = getattr(dv, "spec_decode_enable", True)
        spec_decode_k = getattr(dv, "spec_decode_depth", getattr(dv, "spec_decode_k", 4))
        if self._spec is not None:
            new_k = max(self._spec.MIN_K, min(self._spec.MAX_K, spec_decode_k))
            if spec_decode_enable != self._spec.enabled or new_k != self._spec.current_k:
                self._spec.enabled = spec_decode_enable
                self._spec.current_k = new_k
                spec_k_applied = new_k
                log.debug(
                    "[executor] spec_decode: enabled=%s k=%d",
                    spec_decode_enable,
                    new_k,
                )

        # ── 5. Pre-warm lookups (fire-and-forget) ─────────────────────────────
        pre_warm_queued = 0
        for prefix_hash in dv.pre_warm_predictions:
            try:
                self._coordinator.lookup(
                    prefix_hash=prefix_hash,
                    tenant_id=context.tenant_id,
                )
                pre_warm_queued += 1
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "[executor] pre_warm lookup error for %s: %s", prefix_hash, exc
                )

        # ── 6. Fleet actions (UnifiedDecisionVector only) ─────────────────────
        replication_sent: List[str] = []
        evict_hints_sent: List[str] = []
        cross_node_restore: Optional[str] = None

        if is_unified:
            # 6a. Cross-node restore target
            cross_node_target = getattr(dv, "cross_node_restore_target", None)
            prefer_local      = getattr(dv, "prefer_local_restore", True)
            dynamo_max_load   = _slot(tensor, Slot.DYNAMO_MAX_WORKER_LOAD)

            if cross_node_target and not prefer_local:
                # Safety: if the whole fleet is overloaded, skip cross-node
                if dynamo_max_load > 0.95:
                    overrides.append(OVERRIDE_CROSS_NODE_LOAD_HIGH)
                    log.info(
                        "[executor] cross_node_blocked: fleet max_load=%.2f > 0.95",
                        dynamo_max_load,
                    )
                else:
                    cross_node_restore = cross_node_target
                    log.debug(
                        "[executor] cross_node_restore: target=%s", cross_node_restore
                    )

            # 6b. Replication hints — best-effort; coordinator may not support it
            replicate_to = getattr(dv, "replicate_to_nodes", [])
            for node_id in replicate_to:
                try:
                    self._coordinator.replicate(
                        node_id=node_id,
                        tenant_id=context.tenant_id,
                    )
                    replication_sent.append(node_id)
                    log.debug("[executor] replicate sent to node=%s", node_id)
                except AttributeError:
                    overrides.append(OVERRIDE_REPLICATE_SKIPPED)
                    break  # coordinator doesn't support replication; skip remaining
                except Exception as exc:
                    log.debug("[executor] replicate error for node=%s: %s", node_id, exc)

            # 6c. Eviction hints on remote nodes — suggest freeing cold prefixes
            evict_hints = getattr(dv, "evict_hint_nodes", [])
            for node_id in evict_hints:
                try:
                    self._coordinator.evict_hint(
                        node_id=node_id,
                        tenant_id=context.tenant_id,
                    )
                    evict_hints_sent.append(node_id)
                    log.debug("[executor] evict_hint sent to node=%s", node_id)
                except AttributeError:
                    break  # coordinator doesn't support evict_hint
                except Exception as exc:
                    log.debug("[executor] evict_hint error for node=%s: %s", node_id, exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ExecutionResult(
            admitted=admitted,
            eviction_sent=eviction_sent,
            eviction_blocked=eviction_blocked,
            pre_warm_queued=pre_warm_queued,
            spec_k_applied=spec_k_applied,
            throttle_applied=throttle,
            overrides=overrides,
            elapsed_ms=elapsed_ms,
            decision=dv.to_dict(),
            replication_sent=replication_sent,
            evict_hints_sent=evict_hints_sent,
            cross_node_restore=cross_node_restore,
        )

        log.debug(
            "[executor] cycle done: admitted=%s evict_sent=%s blocked=%s "
            "pre_warm=%d throttle=%.3f overrides=%s elapsed=%.2f ms",
            admitted,
            eviction_sent,
            eviction_blocked,
            pre_warm_queued,
            throttle,
            overrides,
            elapsed_ms,
        )

        # ── 6. FFI Path A: write decode config to shared file ─────────────────
        _write_decode_config(dv)

        return result


# ── FFI Path A: shared decode config file ────────────────────────────────────
#
# The C++ batch_executor reads KORITH_AMF_DECODE_CONFIG (default
# /tmp/amf_decode_config.json) on each step via amf_decode_config_changed().
# Writing < 0.1ms overhead; the engine polls only when the mtime changes.
#
# Path B (Rust FFI extension) replaces this when sub-millisecond latency is
# required.  Both are backward-compatible: if the file is absent or stale the
# Rust scheduler's existing behaviour is unchanged.

import json as _json
import os as _os

_DECODE_CONFIG_PATH = _os.environ.get(
    "KORITH_AMF_DECODE_CONFIG", "/tmp/amf_decode_config.json"
)


def _write_decode_config(dv) -> None:
    """
    Write the unified decode decisions to the shared config file.

    Called after every DecisionExecutor.execute() so the C++ engine can pick
    up the AI's spec decode and KV compression settings for the next request.

    The write is best-effort: if it fails (e.g. read-only /tmp in tests) the
    error is swallowed silently — the Rust scheduler continues as fallback.
    """
    try:
        config = {
            "spec_enable":        getattr(dv, "spec_decode_enable",     True),
            "spec_depth":         getattr(dv, "spec_decode_depth",      4),
            "spec_mode":          getattr(dv, "spec_mode",              "v2_clean"),
            "v3_block_size":      getattr(dv, "spec_v3_block_size",     8),
            "draft_temperature":  getattr(dv, "draft_temperature",      0.0),
            "kv_compress":        getattr(dv, "kv_compression_enable",  False),
            "kv_compress_ratio":  getattr(dv, "kv_compression_ratio",   0.0),
            "override_scheduler": getattr(dv, "adaptive_depth_override", False),
        }
        tmp_path = _DECODE_CONFIG_PATH + ".tmp"
        with open(tmp_path, "w") as fh:
            _json.dump(config, fh, separators=(",", ":"))
        _os.replace(tmp_path, _DECODE_CONFIG_PATH)  # atomic rename
    except Exception:
        pass  # fail-closed: Rust scheduler takes over


def read_decode_config(path: str = _DECODE_CONFIG_PATH) -> dict:
    """
    Read the current decode config written by _write_decode_config().

    Useful for C++-side inspection and tests.  Returns safe defaults if the
    file does not exist or cannot be parsed.
    """
    defaults = {
        "spec_enable": True,
        "spec_depth": 4,
        "spec_mode": "v2_clean",
        "v3_block_size": 8,
        "draft_temperature": 0.0,
        "kv_compress": False,
        "kv_compress_ratio": 0.0,
        "override_scheduler": False,
    }
    try:
        with open(path) as fh:
            data = _json.load(fh)
        defaults.update(data)
    except Exception:
        pass
    return defaults


# ── helpers ──────────────────────────────────────────────────────────────────

def _slot(tensor: np.ndarray, idx: int) -> float:
    """Safely read a slot from the metrics tensor."""
    if idx < len(tensor):
        return float(tensor[idx])
    return 0.0
