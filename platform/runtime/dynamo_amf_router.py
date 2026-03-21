"""AMF-aware router plugin for NVIDIA Dynamo KvRouter.

Extends Dynamo's native KV-overlap routing with a third signal: persistent
AMF snapshot availability.  Critical for long-context (128K+) workloads
where Dynamo's in-memory KV blocks have been evicted.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .amf_coordinator_client import AmfCoordinatorClient
from .prompt_canonicalization import canonicalize_prompt_text

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUTHY


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Cost estimation helpers
# ---------------------------------------------------------------------------


def estimate_recompute_cost_ms(num_tokens: int) -> float:
    """Estimate prefill recompute cost on H200 using quadratic approximation.

    Calibration data (Nemotron 120B):
      120K tokens → 243,610 ms
      252K tokens → 129,690 ms  (hybrid/SSM — lower than pure dense)
      1M   tokens → 439,358 ms

    We use a simple quadratic fit for dense transformers, which is a
    conservative (higher) estimate.
    """
    t = float(max(0, num_tokens))
    # Coefficients fit to (120K, 243610) and (1M, 439358).
    # ms ≈ 3.9e-7 * t^2 + 0.17 * t
    ms = 3.9e-7 * t * t + 0.17 * t
    return max(0.0, ms)


_RESTORE_TIER_MULTIPLIERS: Dict[str, float] = {
    "vram":   0.5,
    "ram":    1.0,
    "nvme":   2.0,
    "remote": 5.0,
    # Accept Dynamo tier labels too.
    "G1": 0.5,
    "G2": 1.0,
    "G3": 2.0,
    "G4": 5.0,
}


def estimate_restore_cost_ms(num_tokens: int, storage_tier: str = "nvme") -> float:
    """Estimate AMF restore latency from benchmark data (817-run average).

    Calibration data (NVMe tier):
      120K tokens → 11,446 ms
      252K tokens →  4,459 ms
      1M   tokens → 11,601 ms

    Tier multipliers applied relative to NVMe baseline:
      vram=0.5x, ram=1x (relative to nvme=2x multiplier → effectively ram=0.5x nvme)
    """
    mul = _RESTORE_TIER_MULTIPLIERS.get(storage_tier.lower(), 2.0)
    t = float(max(1, num_tokens))
    # Base (RAM reference): 0.046 * t ms → ~11,500 ms at 250K tokens.
    base_ms = 0.046 * t
    return max(1.0, base_ms * mul)


# ---------------------------------------------------------------------------
# WorkerScore
# ---------------------------------------------------------------------------


@dataclass
class WorkerScore:
    """Combined routing score for a single Dynamo worker."""

    worker_id: str
    node_id: str
    kv_overlap: float          # Dynamo's native in-memory KV overlap [0, 1]
    load: float                # Worker load estimate [0, 1]
    amf_restore_savings_ms: float  # Expected savings from AMF restore
    combined_score: float      # Final weighted score (higher is better)
    routed_by: str             # "dynamo" | "amf" — which signal dominated
    estimated_restore_ms: float
    estimated_recompute_ms: float


# ---------------------------------------------------------------------------
# AmfRouterPlugin
# ---------------------------------------------------------------------------


class AmfRouterPlugin:
    """Persistent-state-aware routing plugin for Dynamo KvRouter.

    For short context (< AMF_MIN_PREFIX_TOKENS_FOR_AMF), defers entirely to
    Dynamo's native KV overlap scoring — AMF adds no meaningful value there.
    For long context, AMF snapshot availability can dominate because Dynamo's
    in-memory blocks are likely already evicted.

    Scoring formula per worker::

        score = (kv_overlap * kv_weight)
              + (amf_restore_savings_pct * restore_weight)
              - (load * load_weight)

    where amf_restore_savings_pct = 1.0 if worker has an AMF snapshot else 0.0
    (savings are proportional to the recompute/restore ratio).
    """

    def __init__(self, coordinator_url: Optional[str] = None) -> None:
        url = coordinator_url or os.environ.get("AMF_COORDINATOR_URL", "")
        self._enabled = _env_bool("AMF_ROUTER_ENABLED", True) and bool(url)
        self._coordinator = AmfCoordinatorClient(url) if url else None

        self._restore_weight = _env_float("AMF_RESTORE_WEIGHT", 0.8)
        self._kv_overlap_weight = _env_float("AMF_KV_OVERLAP_WEIGHT", 0.5)
        self._load_weight = _env_float("AMF_LOAD_WEIGHT", 0.3)
        self._min_prefix_tokens = _env_int("AMF_MIN_PREFIX_TOKENS_FOR_AMF", 4096)

        self._total_queries = 0
        self._amf_routed = 0
        self._dynamo_routed = 0
        self._total_savings_ms = 0.0

        logger.info(
            "AmfRouterPlugin init: enabled=%s min_tokens=%d restore_w=%.2f kv_w=%.2f load_w=%.2f",
            self._enabled,
            self._min_prefix_tokens,
            self._restore_weight,
            self._kv_overlap_weight,
            self._load_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_workers(
        self,
        request_tokens: int,
        prefix_hash: str,
        tenant_id: str,
        dynamo_worker_scores: List[Dict[str, Any]],
    ) -> List[WorkerScore]:
        """Score all workers combining Dynamo KV overlap and AMF snapshot state.

        Args:
            request_tokens: Number of prefix tokens in the request.
            prefix_hash: Hash of the canonical prefix.
            tenant_id: Tenant identifier.
            dynamo_worker_scores: List of dicts with keys worker_id, node_id,
                kv_overlap (float 0-1), load (float 0-1).

        Returns:
            List of WorkerScore objects sorted by combined_score descending.
        """
        self._total_queries += 1

        use_amf = (
            self._enabled
            and self._coordinator is not None
            and request_tokens >= self._min_prefix_tokens
        )

        # Lookup AMF snapshot nodes.
        amf_nodes: set[str] = set()
        if use_amf:
            try:
                amf_rows = self._coordinator.lookup(
                    prefix_hash=prefix_hash,
                    tenant_id=tenant_id,
                )
                amf_nodes = {str(r.get("node_id", "") or "") for r in amf_rows if r}
            except Exception as exc:
                logger.debug("AmfRouterPlugin coordinator lookup failed: %s", exc)

        recompute_ms = estimate_recompute_cost_ms(request_tokens)
        scored: List[WorkerScore] = []

        for w in dynamo_worker_scores:
            worker_id = str(w.get("worker_id", "") or "")
            node_id = str(w.get("node_id", "") or "")
            kv_overlap = float(w.get("kv_overlap", 0.0) or 0.0)
            load = float(w.get("load", 0.0) or 0.0)

            has_amf = node_id in amf_nodes
            storage_tier = "nvme"  # Conservative default.
            if has_amf:
                # Try to find the best storage tier for this node.
                for r in (self._coordinator.lookup(prefix_hash=prefix_hash, tenant_id=tenant_id) if self._coordinator else []):
                    if str(r.get("node_id", "")) == node_id:
                        tier_raw = str((r.get("metadata") or {}).get("storage_tier", "G3") or "G3")
                        storage_tier = tier_raw
                        break

            restore_ms = estimate_restore_cost_ms(request_tokens, storage_tier) if has_amf else 0.0
            savings_ms = max(0.0, recompute_ms - restore_ms) if has_amf else 0.0
            # Normalise savings to [0, 1] for weighted scoring.
            savings_pct = min(1.0, savings_ms / max(1.0, recompute_ms))

            if use_amf:
                combined = (
                    kv_overlap * self._kv_overlap_weight
                    + savings_pct * self._restore_weight
                    - load * self._load_weight
                )
                routed_by = "amf" if has_amf else "dynamo"
            else:
                combined = kv_overlap * self._kv_overlap_weight - load * self._load_weight
                routed_by = "dynamo"

            scored.append(
                WorkerScore(
                    worker_id=worker_id,
                    node_id=node_id,
                    kv_overlap=kv_overlap,
                    load=load,
                    amf_restore_savings_ms=savings_ms,
                    combined_score=combined,
                    routed_by=routed_by,
                    estimated_restore_ms=restore_ms,
                    estimated_recompute_ms=recompute_ms,
                )
            )

        scored.sort(key=lambda s: s.combined_score, reverse=True)

        if scored:
            best = scored[0]
            if best.routed_by == "amf":
                self._amf_routed += 1
                self._total_savings_ms += best.amf_restore_savings_ms
            else:
                self._dynamo_routed += 1

        return scored

    def best_worker(
        self,
        request_tokens: int,
        prefix_hash: str,
        tenant_id: str,
        dynamo_worker_scores: List[Dict[str, Any]],
    ) -> Optional[WorkerScore]:
        """Return the single best worker for a request.

        Logs the routing decision: which worker, why, estimated savings.
        """
        scored = self.score_workers(
            request_tokens=request_tokens,
            prefix_hash=prefix_hash,
            tenant_id=tenant_id,
            dynamo_worker_scores=dynamo_worker_scores,
        )
        if not scored:
            return None

        best = scored[0]
        logger.info(
            "AmfRouterPlugin routed: worker=%s node=%s by=%s score=%.3f "
            "kv_overlap=%.2f amf_savings_ms=%.0f recompute_ms=%.0f tokens=%d",
            best.worker_id,
            best.node_id,
            best.routed_by,
            best.combined_score,
            best.kv_overlap,
            best.amf_restore_savings_ms,
            best.estimated_recompute_ms,
            request_tokens,
        )
        return best

    def hash_prefix(self, prompt: str, tenant_id: str = "default") -> str:
        """Compute a stable prefix hash after canonicalization.

        Uses canonicalize_prompt_text() to maximise AMF hit rate across
        minor prompt formatting variations.
        """
        canonical = canonicalize_prompt_text(prompt)
        key = f"{tenant_id}:{canonical}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    @property
    def stats(self) -> Dict[str, Any]:
        avg_savings = (
            self._total_savings_ms / float(self._amf_routed)
            if self._amf_routed > 0
            else 0.0
        )
        return {
            "total_queries": self._total_queries,
            "amf_routed": self._amf_routed,
            "dynamo_routed": self._dynamo_routed,
            "avg_savings_ms": avg_savings,
            "total_savings_ms": self._total_savings_ms,
        }
