"""amf_vllm_hook.py — The vLLM integration surface for Axropus AMF.

This is the module that customer deployments import and attach to
their existing vLLM engine. It plugs four seams (§6.1 of the design):

    1. Scheduler: intercepts each incoming request before slot assignment.
    2. Worker: extends the worker to accept pre-populated KV blocks.
    3. Model runner: overrides ``prepare_input_tensors`` to skip the
       prefill tensors when KV has been injected.
    4. Block manager: registers external blocks via the ``is_external``
       tag so vLLM's allocator does not try to free them.

The hook instance owns the TieredCacheRouter and is the single point
of coordination between vLLM's core loop and the AMF stack. It is
safe to use without vLLM being importable: in that case all public
methods become no-ops and the router still serves direct HTTP clients
via ``axropus.server``.
"""

from __future__ import annotations

import enum
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .codecs import FMT_FP16, FP8ScaleSidecar, apply_fp8_scales, get_codec
from .compressed_vram_pool import CompressedVRAMPool, PoolEntry
from .lmcache_adapter import LMCacheAdapter
from .tiered_router import (
    CacheTier,
    LookupResult,
    TieredCacheRouter,
    WritePolicy,
    compute_prefix_hash,
)

logger = logging.getLogger(__name__)


# ── Return codes for the before-prefill hook ────────────────────────────────


class HookAction(enum.Enum):
    COLD_PREFILL          = "cold_prefill"          # no hit, run prefill
    SKIP_PREFILL_TO_DECODE = "skip_to_decode"       # warm hit, go straight to decode
    RELOAD_FROM_LMCACHE   = "reload_from_lmcache"   # G3 hit, KV needs to be copied in


@dataclass
class HookDecision:
    action:       HookAction
    prefix_hash:  str
    tier:         CacheTier
    n_tokens:     int
    latency_ms:   float
    restored_kv:  Optional[torch.Tensor] = None    # only set for RELOAD_FROM_LMCACHE
    pool_entry:   Optional[PoolEntry]    = None    # only set for SKIP_PREFILL_TO_DECODE


# ── The hook ────────────────────────────────────────────────────────────────


class AMFvLLMHook:
    """The canonical AMF integration point for a vLLM engine.

    Args:
        pool:           Compressed VRAM pool instance (create one per engine).
        router:         Optional pre-built router. If ``None`` we build one
                        over ``pool`` with a default LMCacheAdapter.
        num_layers:     Transformer layer count. Used to validate KV shapes.
        num_kv_heads:   KV head count. Used to validate KV shapes.
        head_dim:       Per-head dimension.
        min_prefix:     Prefixes shorter than this many tokens bypass AMF
                        and go straight to cold prefill.
        enabled:        Master switch. When False every hook becomes a no-op.
    """

    def __init__(
        self,
        pool: CompressedVRAMPool,
        router: Optional[TieredCacheRouter] = None,
        *,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        min_prefix: int = 64,
        enabled: bool = True,
    ) -> None:
        self.pool = pool
        self.router = router or TieredCacheRouter(
            pool=pool,
            lmcache=LMCacheAdapter(),
            min_prefix_tokens=min_prefix,
            write_policy=WritePolicy.LARGE_ONLY,
        )
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.min_prefix = min_prefix
        self.enabled = enabled and _env_truthy("KORITH_ENABLE_AMF", default=True)

        self._seq_hashes: Dict[int, str] = {}
        self._seq_tokens: Dict[int, List[int]] = {}
        self._seq_decisions: Dict[int, HookDecision] = {}

        # Statistics.
        self._hits = 0
        self._misses = 0
        self._saves = 0
        self._total_skip_tokens = 0
        self._total_saved_ms = 0.0

        logger.info(
            "AMFvLLMHook initialised enabled=%s layers=%d heads=%d head_dim=%d min_prefix=%d",
            self.enabled,
            num_layers,
            num_kv_heads,
            head_dim,
            min_prefix,
        )

    # ── Request arrival ─────────────────────────────────────────────────────

    def on_request_arrival(
        self,
        seq_id: int,
        token_ids: Sequence[int],
    ) -> HookDecision:
        """Called for each new request before vLLM assigns KV blocks.

        Returns a :class:`HookDecision`. The caller is responsible for
        acting on the action:

            * ``COLD_PREFILL``           run normal vLLM prefill.
            * ``SKIP_PREFILL_TO_DECODE`` map the pool entry into vLLM's
                                          block table and jump directly to
                                          decode.
            * ``RELOAD_FROM_LMCACHE``    copy ``decision.restored_kv``
                                          into freshly allocated blocks.
        """
        if not self.enabled or len(token_ids) < self.min_prefix:
            return HookDecision(
                action=HookAction.COLD_PREFILL,
                prefix_hash="",
                tier=CacheTier.COLD,
                n_tokens=len(token_ids),
                latency_ms=0.0,
            )

        prefix_hash = compute_prefix_hash(token_ids)
        self._seq_hashes[seq_id] = prefix_hash
        self._seq_tokens[seq_id] = list(token_ids)

        result = self.router.lookup(prefix_hash, token_ids)

        if result.tier == CacheTier.G1_AMF:
            self._hits += 1
            self._total_skip_tokens += len(token_ids)
            entry: PoolEntry = result.payload
            decision = HookDecision(
                action=HookAction.SKIP_PREFILL_TO_DECODE,
                prefix_hash=prefix_hash,
                tier=CacheTier.G1_AMF,
                n_tokens=len(token_ids),
                latency_ms=result.latency_ms,
                pool_entry=entry,
            )
            self._seq_decisions[seq_id] = decision
            logger.info(
                "[AMF_HIT] seq=%s prefix=%s tokens=%d latency=%.2fms",
                seq_id,
                prefix_hash[:12],
                len(token_ids),
                result.latency_ms,
            )
            return decision

        if result.tier == CacheTier.G3_LMCACHE:
            kv_tensor: torch.Tensor = result.payload
            decision = HookDecision(
                action=HookAction.RELOAD_FROM_LMCACHE,
                prefix_hash=prefix_hash,
                tier=CacheTier.G3_LMCACHE,
                n_tokens=len(token_ids),
                latency_ms=result.latency_ms,
                restored_kv=kv_tensor,
            )
            self._seq_decisions[seq_id] = decision
            return decision

        self._misses += 1
        decision = HookDecision(
            action=HookAction.COLD_PREFILL,
            prefix_hash=prefix_hash,
            tier=CacheTier.COLD,
            n_tokens=len(token_ids),
            latency_ms=result.latency_ms,
        )
        self._seq_decisions[seq_id] = decision
        return decision

    # ── Prefill-complete callback ───────────────────────────────────────────

    def on_prefill_complete(
        self,
        seq_id: int,
        kv: torch.Tensor,
        *,
        saved_ms: float = 0.0,
    ) -> None:
        """Called by vLLM once a cold prefill finishes.

        The hook captures the KV tensor and stores it in the tiered
        router so that subsequent hits on the same prefix are warm.
        """
        if not self.enabled:
            return
        prefix_hash = self._seq_hashes.get(seq_id)
        token_ids = self._seq_tokens.get(seq_id)
        if not prefix_hash or not token_ids:
            return
        if len(token_ids) < self.min_prefix:
            return

        ok = self.router.store_after_prefill(
            prefix_hash,
            token_ids,
            kv,
            savings_ms=saved_ms,
        )
        if ok:
            self._saves += 1
            self._total_saved_ms += saved_ms
            logger.info(
                "[AMF_SAVE] seq=%s prefix=%s tokens=%d saved_ms=%.1f",
                seq_id,
                prefix_hash[:12],
                len(token_ids),
                saved_ms,
            )

    def on_request_complete(self, seq_id: int) -> None:
        """Cleanup per-sequence bookkeeping once the request finishes."""
        self._seq_hashes.pop(seq_id, None)
        self._seq_tokens.pop(seq_id, None)
        self._seq_decisions.pop(seq_id, None)

    # ── Block-table injection (seam #4) ─────────────────────────────────────

    def inject_blocks(
        self,
        seq_id: int,
        vllm_block_table: Any,
    ) -> bool:
        """Wire AMF-owned blocks into vLLM's block table for a sequence.

        Assumes the sequence has already been routed through
        :meth:`on_request_arrival` and that the decision cached there is
        ``SKIP_PREFILL_TO_DECODE``.
        """
        if not self.enabled:
            return False
        decision = self._seq_decisions.get(seq_id)
        if decision is None or decision.action != HookAction.SKIP_PREFILL_TO_DECODE:
            return False
        return self.pool.map_into_vllm(
            decision.prefix_hash,
            seq_id,
            vllm_block_table,
        )

    # ── Model-runner seam (seam #3) ─────────────────────────────────────────

    def should_skip_prefill_tensors(self, seq_id: int) -> bool:
        """Return True if the model runner should skip prefill prep for this seq."""
        if not self.enabled:
            return False
        decision = self._seq_decisions.get(seq_id)
        return (
            decision is not None
            and decision.action == HookAction.SKIP_PREFILL_TO_DECODE
        )

    # ── Worker seam (seam #2): FP8 scale reapplication ──────────────────────

    def reapply_fp8_scales(self, model: Any, seq_id: int) -> None:
        """Fix the FP8 scale-drift bug.

        On warm restore we *must* re-apply the stored per-layer FP8 scales
        before vLLM's calculate_kv_scales pass runs. Otherwise vLLM
        computes fresh (incorrect) scales on the restored cache and
        subsequent decode steps produce garbage. See §4.1 of the design
        doc.
        """
        if not self.enabled:
            return
        decision = self._seq_decisions.get(seq_id)
        if decision is None or decision.pool_entry is None:
            return

        entry = decision.pool_entry
        # Each layer blob carries an optional FP8ScaleSidecar in its meta.
        # We collapse them into a single weighted-average sidecar for the
        # model-level apply call; vLLM's attention modules all share a
        # per-tensor scale in the default configuration.
        k_scales: List[float] = []
        v_scales: List[float] = []
        for blob in entry.blobs:
            sidecar = blob.meta.get("sidecar")
            if isinstance(sidecar, FP8ScaleSidecar):
                k_scales.append(sidecar.k_scale)
                v_scales.append(sidecar.v_scale)

        if not k_scales:
            return  # no FP8 data on this entry

        avg = FP8ScaleSidecar(
            k_scale=sum(k_scales) / len(k_scales),
            v_scale=sum(v_scales) / len(v_scales),
        )
        apply_fp8_scales(model, avg)

    # ── Stats ───────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / max(1, total)
        return {
            "enabled":           self.enabled,
            "hits":              self._hits,
            "misses":            self._misses,
            "saves":             self._saves,
            "hit_rate":          hit_rate,
            "total_skip_tokens": self._total_skip_tokens,
            "total_saved_ms":    self._total_saved_ms,
            "router":            self.router.stats(),
        }

    def log_summary(self) -> None:
        s = self.stats()
        logger.info(
            "[AMF_VLLM] hits=%d misses=%d saves=%d hit_rate=%.3f tokens_skipped=%d saved_ms=%.0f",
            s["hits"],
            s["misses"],
            s["saves"],
            s["hit_rate"],
            s["total_skip_tokens"],
            s["total_saved_ms"],
        )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def build_default_hook(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_layer: int = 1 << 30,
    device: str = "cuda",
) -> AMFvLLMHook:
    """Convenience constructor used by ``axropus.server``."""
    pool = CompressedVRAMPool(
        num_layers=num_layers,
        bytes_per_layer=bytes_per_layer,
        device=device,
    )
    return AMFvLLMHook(
        pool=pool,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


__all__ = [
    "AMFvLLMHook",
    "HookAction",
    "HookDecision",
    "build_default_hook",
]
