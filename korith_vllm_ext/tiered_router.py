"""tiered_router.py — Five-tier cache router for AMF.

Routes prefix lookups through the storage hierarchy defined in §1.1 of
the design doc:

    G1  AMF compressed VRAM pool     sub-100 ms restore
    G2  vLLM paged attention active  (live sequence cache)
    G3  LMCache CPU offload          50-300 ms via PCIe
    G4  LMCache NVMe                 200-1500 ms
    G5  LMCache remote / S3          1-10 s

G2 is outside this module — it is managed by vLLM's own block manager
and the router only consults G1, G3, G4, G5. G3-G5 are all funneled
through the ``LMCacheAdapter`` which picks the actual backend.

The router also tracks miss reasons (§5.3) so operators can diagnose
hit-rate regressions, and supports a pre-fetch queue (§5.2) for warming
prefixes that the scheduler expects to need soon.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .compressed_vram_pool import CompressedVRAMPool, PoolEntry
from .lmcache_adapter import LMCacheAdapter

logger = logging.getLogger(__name__)


# ── Constants & enums ────────────────────────────────────────────────────────


class CacheTier(enum.Enum):
    G1_AMF       = "amf"
    G3_LMCACHE   = "lmcache"
    COLD         = "cold"


class MissReason(enum.Enum):
    FIRST_REQUEST      = "first_request"
    EVICTED_FROM_AMF   = "evicted_from_amf"
    LMCACHE_DISABLED   = "lmcache_disabled"
    LMCACHE_MISS       = "lmcache_miss"
    LMCACHE_ERROR      = "lmcache_error"
    FORMAT_MISMATCH    = "format_mismatch"
    PREFIX_TOO_SHORT   = "prefix_too_short"


class WritePolicy(enum.Enum):
    """Where to mirror prefills after AMF insert."""

    ALWAYS       = "always"
    LARGE_ONLY   = "large_only"
    ON_EVICTION  = "on_eviction"
    NEVER        = "never"


# ── Return payload ───────────────────────────────────────────────────────────


@dataclass
class LookupResult:
    """Result of a router lookup."""

    prefix_hash: str
    tier:        CacheTier
    payload:     Any             # PoolEntry for G1, torch.Tensor for G3, None for COLD
    latency_ms:  float = 0.0
    miss_reason: Optional[MissReason] = None


# ── Utility: prefix hashing ──────────────────────────────────────────────────


def compute_prefix_hash(token_ids: Sequence[int]) -> str:
    """Content-addressed prefix hash.

    Uses SHA-256 over the little-endian 32-bit int representation of
    the token ids and truncates to 16 hex chars (64 bits). Matches the
    scheme described in §1.2 and §6.2 of the design doc.
    """
    h = hashlib.sha256()
    for t in token_ids:
        h.update(int(t & 0xFFFFFFFF).to_bytes(4, "little"))
    return h.hexdigest()[:16]


# ── The router ───────────────────────────────────────────────────────────────


class TieredCacheRouter:
    """Coordinates lookups across AMF pool and LMCache tiers."""

    def __init__(
        self,
        pool: CompressedVRAMPool,
        lmcache: Optional[LMCacheAdapter] = None,
        *,
        min_prefix_tokens: int = 64,
        write_policy: WritePolicy = WritePolicy.LARGE_ONLY,
        large_write_threshold: int = 2048,
    ) -> None:
        self.pool = pool
        self.lmcache = lmcache or LMCacheAdapter()
        self.min_prefix_tokens = min_prefix_tokens
        self.write_policy = write_policy
        self.large_write_threshold = large_write_threshold

        # Track which hashes have *ever* been seen, so we can distinguish
        # FIRST_REQUEST from EVICTED_FROM_AMF.
        self._seen: set = set()

        # Per-reason miss counters (reported via Prometheus by the server).
        self._miss_counts: Dict[MissReason, int] = {r: 0 for r in MissReason}
        self._hit_counts: Dict[CacheTier, int] = {t: 0 for t in CacheTier}
        self._latency_sum: Dict[CacheTier, float] = {t: 0.0 for t in CacheTier}

        # Prefetch plumbing.
        self._prefetch_queue: "queue.PriorityQueue[Tuple[int, str, Any]]" = queue.PriorityQueue()
        self._prefetch_stop = threading.Event()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_resolver: Optional[Callable[[str], Optional[torch.Tensor]]] = None

        self._lock = threading.RLock()

    # ── Lookup ─────────────────────────────────────────────────────────────

    def lookup(
        self,
        prefix_hash: str,
        token_ids: Optional[Sequence[int]] = None,
    ) -> LookupResult:
        """Walk the tier hierarchy and return the first hit."""
        t0 = time.monotonic()

        if token_ids is not None and len(token_ids) < self.min_prefix_tokens:
            return self._miss(prefix_hash, MissReason.PREFIX_TOO_SHORT, t0)

        # G1: AMF compressed VRAM pool.
        entry = self.pool.get(prefix_hash)
        if entry is not None:
            latency = (time.monotonic() - t0) * 1000.0
            with self._lock:
                self._hit_counts[CacheTier.G1_AMF] += 1
                self._latency_sum[CacheTier.G1_AMF] += latency
                self._seen.add(prefix_hash)
            return LookupResult(
                prefix_hash=prefix_hash,
                tier=CacheTier.G1_AMF,
                payload=entry,
                latency_ms=latency,
            )

        # G3+ (LMCache).
        if not self.lmcache.enabled:
            return self._miss(prefix_hash, MissReason.LMCACHE_DISABLED, t0)

        try:
            kv = self.lmcache.lookup(prefix_hash)
        except Exception as exc:  # pragma: no cover - depends on backend
            logger.warning("LMCache lookup failed: %s", exc)
            return self._miss(prefix_hash, MissReason.LMCACHE_ERROR, t0)

        if kv is None:
            return self._miss(prefix_hash, MissReason.LMCACHE_MISS, t0)

        # G3 hit: promote asynchronously to G1 so subsequent hits are faster.
        self._schedule_promotion(prefix_hash, kv)

        latency = (time.monotonic() - t0) * 1000.0
        with self._lock:
            self._hit_counts[CacheTier.G3_LMCACHE] += 1
            self._latency_sum[CacheTier.G3_LMCACHE] += latency
            self._seen.add(prefix_hash)
        return LookupResult(
            prefix_hash=prefix_hash,
            tier=CacheTier.G3_LMCACHE,
            payload=kv,
            latency_ms=latency,
        )

    def _miss(
        self,
        prefix_hash: str,
        reason: MissReason,
        t0: float,
    ) -> LookupResult:
        latency = (time.monotonic() - t0) * 1000.0
        with self._lock:
            # Upgrade FIRST_REQUEST to EVICTED_FROM_AMF if we've seen it before.
            if reason == MissReason.LMCACHE_MISS and prefix_hash in self._seen:
                reason = MissReason.EVICTED_FROM_AMF
            elif reason == MissReason.LMCACHE_DISABLED and prefix_hash in self._seen:
                reason = MissReason.EVICTED_FROM_AMF
            elif reason == MissReason.LMCACHE_MISS and prefix_hash not in self._seen:
                reason = MissReason.FIRST_REQUEST
            elif reason == MissReason.LMCACHE_DISABLED and prefix_hash not in self._seen:
                reason = MissReason.FIRST_REQUEST
            self._miss_counts[reason] += 1
            self._hit_counts[CacheTier.COLD] += 1
            self._latency_sum[CacheTier.COLD] += latency
        return LookupResult(
            prefix_hash=prefix_hash,
            tier=CacheTier.COLD,
            payload=None,
            latency_ms=latency,
            miss_reason=reason,
        )

    # ── Store-after-prefill ────────────────────────────────────────────────

    def store_after_prefill(
        self,
        prefix_hash: str,
        token_ids: Sequence[int],
        kv: torch.Tensor,
        *,
        savings_ms: float = 0.0,
        format: Optional[str] = None,
    ) -> bool:
        """Insert a freshly prefilled KV tensor into the tiers.

        Always populates G1 (the compressed VRAM pool). Additionally
        mirrors to LMCache according to ``write_policy``.
        """
        with self._lock:
            self._seen.add(prefix_hash)

        ok_g1 = self.pool.put_from_raw(
            prefix_hash,
            kv,
            format=format,
            savings_ms=savings_ms,
        )

        if self._should_mirror(token_ids, is_eviction=False):
            try:
                self.lmcache.store(prefix_hash, kv.detach().cpu())
            except Exception as exc:  # pragma: no cover
                logger.warning("LMCache store failed: %s", exc)
        return ok_g1

    def _should_mirror(
        self,
        token_ids: Sequence[int],
        *,
        is_eviction: bool,
    ) -> bool:
        policy = self.write_policy
        if policy == WritePolicy.NEVER:
            return False
        if policy == WritePolicy.ALWAYS:
            return True
        if policy == WritePolicy.LARGE_ONLY:
            return len(token_ids) >= self.large_write_threshold
        if policy == WritePolicy.ON_EVICTION:
            return is_eviction
        return False

    # ── Prefetch ───────────────────────────────────────────────────────────

    def set_prefetch_resolver(
        self,
        resolver: Callable[[str], Optional[torch.Tensor]],
    ) -> None:
        """Register a function that turns a prefix hash into a raw KV tensor.

        The prefetch worker calls this for each queued hash, then inserts
        the result into the AMF pool. Typically the resolver is backed by
        LMCache or by a cold prefill on a spare GPU.
        """
        self._prefetch_resolver = resolver

    def prefetch(self, prefix_hash: str, priority: int = 0, hint: Any = None) -> None:
        """Queue a prefix for warming.

        Lower ``priority`` values are pulled first.
        """
        self._prefetch_queue.put((priority, prefix_hash, hint))
        self._ensure_prefetch_running()

    def _ensure_prefetch_running(self) -> None:
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._prefetch_stop.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name="axropus-prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

    def stop_prefetch(self) -> None:
        self._prefetch_stop.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1.0)
            self._prefetch_thread = None

    def _prefetch_worker(self) -> None:
        while not self._prefetch_stop.is_set():
            try:
                priority, prefix_hash, hint = self._prefetch_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if prefix_hash in self.pool:
                    continue  # already warm
                kv = None
                if self._prefetch_resolver is not None:
                    kv = self._prefetch_resolver(prefix_hash)
                if kv is None and self.lmcache.enabled:
                    kv = self.lmcache.lookup(prefix_hash)
                if kv is not None:
                    self.pool.put_from_raw(prefix_hash, kv)
                    logger.debug("prefetch: promoted %s", prefix_hash[:12])
            except Exception as exc:  # pragma: no cover
                logger.warning("prefetch error for %s: %s", prefix_hash, exc)

    def _schedule_promotion(self, prefix_hash: str, kv: torch.Tensor) -> None:
        """Promote a G3 hit to G1 asynchronously."""
        try:
            self.pool.put_from_raw(prefix_hash, kv)
        except Exception as exc:  # pragma: no cover
            logger.warning("promotion from G3 failed: %s", exc)

    # ── Telemetry ──────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            hits = dict(self._hit_counts)
            latencies = {
                k.value: (self._latency_sum[k] / max(1, self._hit_counts[k]))
                for k in self._hit_counts
            }
            misses = {k.value: v for k, v in self._miss_counts.items()}
        return {
            "tier_hits":      {k.value: v for k, v in hits.items()},
            "tier_latency":   latencies,
            "miss_reasons":   misses,
            "pool_stats":     self.pool.stats(),
            "lmcache_stats":  self.lmcache.stats(),
            "write_policy":   self.write_policy.value,
        }

    def log_summary(self) -> None:
        s = self.stats()
        hits = s["tier_hits"]
        total = sum(hits.values())
        amf_rate = hits.get("amf", 0) / max(1, total)
        lmc_rate = hits.get("lmcache", 0) / max(1, total)
        cold_rate = hits.get("cold", 0) / max(1, total)
        logger.info(
            "[TieredRouter] total=%d amf=%.2f%% lmcache=%.2f%% cold=%.2f%% policy=%s",
            total,
            amf_rate * 100,
            lmc_rate * 100,
            cold_rate * 100,
            self.write_policy.value,
        )


__all__ = [
    "CacheTier",
    "LookupResult",
    "MissReason",
    "TieredCacheRouter",
    "WritePolicy",
    "compute_prefix_hash",
]
