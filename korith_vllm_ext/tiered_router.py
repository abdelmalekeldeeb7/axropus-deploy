"""tiered_router.py — Orchestrator for the Axropus AMF + LMCache tier stack.

Tiered cache hierarchy:

    G1 (hot, compressed)   — Axropus AMF compressed VRAM pool       [owned here]
    G2 (active batch)      — vLLM paged attention                   [vLLM owns]
    G3 (CPU offload)       — LMCache                                [LMCache owns]
    G4 (NVMe / disk)       — LMCache                                [LMCache owns]
    G5 (remote / S3)       — LMCache                                [LMCache owns]

The router wraps a ``CompressedVRAMPool`` (G1) and an optional
``LMCacheAdapter`` (G3-G5). Public flow:

  lookup(prefix_hash, token_ids) -> (kv_handle, source)

    1. Try G1 (AMF compressed pool) — if hit, return ("amf")
    2. Try G3+ (LMCache)             — if hit, promote to G1, return ("lmcache")
    3. Otherwise                     — return (None, "cold")

  store_after_prefill(prefix_hash, token_ids, kv_tensor)
    - Always stores into G1.
    - Optionally write-through to LMCache so the next AMF eviction has
      a fallback in CPU / NVMe without having to re-prefill.

Environment variables:
    AXROPUS_LMCACHE_FALLBACK         = false  # master switch
    AXROPUS_PROMOTE_LMCACHE_HITS     = true   # promote G3 hits → G1
    AXROPUS_LMCACHE_WRITE_THROUGH    = true   # mirror cold prefills to LMCache
    AXROPUS_LMCACHE_LOOKUP_TIMEOUT_MS = 200   # bail out of G3 if slow
    LMCACHE_CONFIG_PATH              = unset  # forwarded to LMCacheEngineConfig
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from .compressed_vram_pool import CompressedVRAMPool
    from .lmcache_adapter import LMCacheAdapter
    import torch

logger = logging.getLogger("axropus.tiered_router")


# ── Env helpers ──────────────────────────────────────────────────────────────

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ── Lookup source sentinel ───────────────────────────────────────────────────

SOURCE_AMF     = "amf"       # G1 compressed VRAM pool
SOURCE_LMCACHE = "lmcache"   # G3-G5 via LMCacheAdapter
SOURCE_COLD    = "cold"      # full fallthrough — vLLM must prefill


# ── Router ───────────────────────────────────────────────────────────────────

class TieredCacheRouter:
    """Orchestrates the AMF → LMCache → cold fallthrough path.

    The router is intentionally source-agnostic from the perspective of
    ``amf_vllm_hook``: callers ask for ``lookup(prefix_hash, token_ids)``
    and get back either a handle from AMF, a raw KV tensor from LMCache,
    or ``None`` meaning "cold prefill required".
    """

    def __init__(
        self,
        amf_pool: "CompressedVRAMPool",
        lmcache: Optional["LMCacheAdapter"] = None,
    ) -> None:
        self.amf_pool = amf_pool
        self.lmcache = lmcache

        # Auto-construct an LMCacheAdapter if one wasn't passed in and
        # the fallback flag is set. Import-guarded so this is a no-op
        # when LMCache isn't installed.
        if self.lmcache is None and _env_truthy("AXROPUS_LMCACHE_FALLBACK"):
            try:
                from .lmcache_adapter import LMCacheAdapter
                cfg = os.environ.get("LMCACHE_CONFIG_PATH") or None
                self.lmcache = LMCacheAdapter(config_path=cfg)
            except Exception as exc:
                logger.warning(
                    "[TIERED] lazy LMCacheAdapter init failed: %s", exc
                )
                self.lmcache = None

        self.lmcache_enabled = bool(
            _env_truthy("AXROPUS_LMCACHE_FALLBACK")
            and self.lmcache is not None
            and self.lmcache.available
        )
        self.promote_on_hit = _env_truthy(
            "AXROPUS_PROMOTE_LMCACHE_HITS", default=True
        )
        self.write_through = _env_truthy(
            "AXROPUS_LMCACHE_WRITE_THROUGH", default=True
        )
        self.lookup_timeout_ms = _env_int(
            "AXROPUS_LMCACHE_LOOKUP_TIMEOUT_MS", default=200
        )

        # Stats
        self._lock = threading.Lock()
        self.stats: dict = {
            "total_lookups": 0,
            "amf_hits":      0,
            "lmcache_hits":  0,
            "cold_misses":   0,
            "promotions":    0,
            "promotion_fails": 0,
            "lmcache_timeouts": 0,
            "writethrough_ok": 0,
            "writethrough_err": 0,
        }

        logger.info(
            "[TIERED] init lmcache_enabled=%s promote_on_hit=%s "
            "write_through=%s lookup_timeout_ms=%d",
            self.lmcache_enabled,
            self.promote_on_hit,
            self.write_through,
            self.lookup_timeout_ms,
        )

    # ── Lookup path ───────────────────────────────────────────────────────────

    def lookup(
        self,
        prefix_hash: str,
        token_ids: List[int],
    ) -> Tuple[Optional[Any], str]:
        """Return ``(kv_handle, source)``.

        ``source`` is one of ``"amf"``, ``"lmcache"``, ``"cold"``.
        When source is ``"cold"``, ``kv_handle`` is ``None`` and the
        caller must run a real prefill.
        """
        with self._lock:
            self.stats["total_lookups"] += 1

        # ── G1: Axropus compressed VRAM pool ─────────────────────────────
        if self.amf_pool.contains(prefix_hash):
            with self._lock:
                self.stats["amf_hits"] += 1
            # The caller typically calls amf_pool.restore(...) directly
            # since the VRAM pool scatters into live KV blocks. We return
            # a truthy marker so the hook knows it was an AMF hit without
            # re-fetching the entry.
            return (prefix_hash, SOURCE_AMF)

        # ── G3-G5: LMCache fallback ──────────────────────────────────────
        if self.lmcache_enabled and self.lmcache is not None:
            t0 = time.perf_counter()
            raw_kv = self.lmcache.lookup(prefix_hash, token_ids)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if dt_ms > self.lookup_timeout_ms:
                # Too slow — log, count, but still use the result if we got
                # one. The timeout is informational unless dt exceeded the
                # budget AND the result is None (i.e. a slow miss).
                with self._lock:
                    self.stats["lmcache_timeouts"] += 1
                logger.debug(
                    "[TIERED] LMCache lookup slow: %.1f ms > %d ms budget",
                    dt_ms, self.lookup_timeout_ms,
                )

            if raw_kv is not None:
                with self._lock:
                    self.stats["lmcache_hits"] += 1
                logger.info(
                    "[TIERED] G3 hit prefix=%s... in %.1f ms",
                    prefix_hash[:8], dt_ms,
                )

                # Promote into AMF pool so the NEXT hit is a G1 (warm).
                if self.promote_on_hit:
                    self._promote(prefix_hash, token_ids, raw_kv)

                return (raw_kv, SOURCE_LMCACHE)

        # ── G5 miss: cold prefill ────────────────────────────────────────
        with self._lock:
            self.stats["cold_misses"] += 1
        return (None, SOURCE_COLD)

    # ── Store path (called after a cold prefill) ──────────────────────────────

    def store_after_prefill(
        self,
        prefix_hash: str,
        token_ids: List[int],
        gpu_cache: Any,
        block_table: List[int],
        n_tokens: int,
        raw_kv: Optional["torch.Tensor"] = None,
    ) -> None:
        """Write a freshly-prefilled KV into G1 (always) and LMCache (optional).

        The AMF pool needs live gpu_cache + block_table to gather from;
        LMCache wants a materialized raw KV tensor. If ``raw_kv`` is not
        supplied but write-through is enabled, write-through is skipped
        with a debug log (caller should pre-materialize if they want it).
        """
        # G1 store is mandatory.
        try:
            self.amf_pool.put(prefix_hash, gpu_cache, block_table, n_tokens)
        except Exception as exc:
            logger.warning("[TIERED] AMF pool put failed: %s", exc)

        # G3 write-through is optional.
        if not (self.write_through and self.lmcache_enabled
                and self.lmcache is not None):
            return
        if raw_kv is None:
            logger.debug(
                "[TIERED] write-through enabled but raw_kv=None — skipping"
            )
            return
        try:
            ok = self.lmcache.store(prefix_hash, token_ids, raw_kv)
            with self._lock:
                if ok:
                    self.stats["writethrough_ok"] += 1
                else:
                    self.stats["writethrough_err"] += 1
        except Exception as exc:
            with self._lock:
                self.stats["writethrough_err"] += 1
            logger.warning("[TIERED] LMCache write-through failed: %s", exc)

    # ── Promotion: LMCache (raw) → AMF (compressed) ───────────────────────────

    def _promote(
        self,
        prefix_hash: str,
        token_ids: List[int],
        raw_kv: "torch.Tensor",
    ) -> None:
        """Insert a raw KV tensor from LMCache into the compressed VRAM pool."""
        try:
            self.amf_pool.insert_from_raw(prefix_hash, raw_kv, len(token_ids))
            with self._lock:
                self.stats["promotions"] += 1
            logger.debug(
                "[TIERED] promoted prefix=%s... from LMCache → AMF",
                prefix_hash[:8],
            )
        except Exception as exc:
            with self._lock:
                self.stats["promotion_fails"] += 1
            logger.warning(
                "[TIERED] promotion failed prefix=%s: %s", prefix_hash[:8], exc
            )

    # ── Reporting ─────────────────────────────────────────────────────────────

    def hit_rate(self) -> dict:
        """Return hit-rate breakdown across tiers."""
        with self._lock:
            s = dict(self.stats)
        total = max(s["total_lookups"], 1)
        return {
            "amf_hit_rate":     s["amf_hits"]     / total,
            "lmcache_hit_rate": s["lmcache_hits"] / total,
            "combined_hit_rate": (
                (s["amf_hits"] + s["lmcache_hits"]) / total
            ),
            "cold_miss_rate":   s["cold_misses"]  / total,
            **s,
        }

    def config(self) -> dict:
        """Return the router configuration (for logs / debugging)."""
        return {
            "lmcache_enabled":     self.lmcache_enabled,
            "promote_on_hit":      self.promote_on_hit,
            "write_through":       self.write_through,
            "lookup_timeout_ms":   self.lookup_timeout_ms,
        }
