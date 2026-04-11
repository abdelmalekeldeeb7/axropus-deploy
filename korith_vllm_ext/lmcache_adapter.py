"""lmcache_adapter.py — Isolated adapter for the LMCache fallback tier.

Axropus AMF is the G1 (compressed GPU hot) tier in a tiered cache stack.
LMCache sits below as G3-G5 (CPU / NVMe / remote). This module wraps
LMCache's API behind a stable interface so that:

  1. Axropus runs fine with or without LMCache installed (import-guarded).
  2. LMCache API changes are absorbed here and don't ripple through AMF.
  3. LMCache hits return raw KV so AMF can re-compress and promote them
     into the G1 compressed pool — the "second hit is warm" pattern.

Public API:
    LMCacheAdapter(config_path=None)
      .available -> bool
      .lookup(prefix_hash, token_ids) -> Optional[Tensor]
      .store(prefix_hash, token_ids, kv_tensor) -> bool

All errors are caught and logged — LMCache is a fallback tier, so any
failure just falls through to a cold prefill instead of breaking AMF.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

logger = logging.getLogger("axropus.lmcache")

# ── Import guard ─────────────────────────────────────────────────────────────
#
# LMCache is an optional dependency. Everything below is guarded so that
# ``from korith_vllm_ext.lmcache_adapter import LMCacheAdapter`` always
# succeeds — even if lmcache is not installed, in which case the adapter
# becomes a no-op that always reports ``available=False``.

try:
    from lmcache.cache_engine import LMCacheEngine  # type: ignore
    from lmcache.config import LMCacheEngineConfig  # type: ignore
    try:
        from lmcache.utils import CacheEngineKey  # type: ignore
    except ImportError:
        # Newer LMCache API moved CacheEngineKey under cache_engine
        from lmcache.cache_engine import CacheEngineKey  # type: ignore
    LMCACHE_AVAILABLE = True
except ImportError:
    LMCACHE_AVAILABLE = False
    LMCacheEngine = None  # type: ignore
    LMCacheEngineConfig = None  # type: ignore
    CacheEngineKey = None  # type: ignore


if TYPE_CHECKING:
    import torch


class LMCacheAdapter:
    """Stable wrapper around the LMCache CacheEngine.

    All public methods are safe to call when LMCache is not installed;
    they will log at debug level and return ``None`` / ``False``.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.engine: Optional[Any] = None
        self._lookup_errors: int = 0
        self._store_errors: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._stores: int = 0

        if not LMCACHE_AVAILABLE:
            logger.info(
                "[LMCACHE] library not installed — LMCacheAdapter is a no-op"
            )
            return

        try:
            if config_path:
                cfg = LMCacheEngineConfig.from_file(config_path)
            else:
                cfg = LMCacheEngineConfig.from_env()
            self.engine = LMCacheEngine(cfg)
            logger.info(
                "[LMCACHE] adapter initialized (config_path=%s)", config_path or "env"
            )
        except Exception as exc:
            # Swallow any config / init error — LMCache is a fallback tier,
            # not a hard dependency. The rest of Axropus keeps working.
            logger.warning(
                "[LMCACHE] init failed, running without fallback: %s", exc
            )
            self.engine = None

    # ── Health ────────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True iff LMCache is installed and the engine initialized cleanly."""
        return self.engine is not None

    # ── Lookup ────────────────────────────────────────────────────────────────

    def lookup(
        self,
        prefix_hash: str,
        token_ids: List[int],
    ) -> Optional["torch.Tensor"]:
        """Look up KV cache for a prefix. Returns raw KV tensor on hit.

        On error / miss returns ``None``. Caller (TieredCacheRouter) is
        responsible for promoting the returned tensor into the AMF pool.
        """
        if not self.available:
            return None

        try:
            key = self._make_key(prefix_hash, token_ids)
            kv = self.engine.retrieve(key)  # type: ignore[union-attr]
            if kv is None:
                self._misses += 1
                return None
            self._hits += 1
            return kv
        except Exception as exc:
            self._lookup_errors += 1
            # Debug level: LMCache misses for nonexistent prefixes are normal.
            logger.debug("[LMCACHE] lookup miss/error: %s", exc)
            return None

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(
        self,
        prefix_hash: str,
        token_ids: List[int],
        kv_tensor: "torch.Tensor",
    ) -> bool:
        """Store a KV tensor into LMCache. Backgrounded by LMCache itself."""
        if not self.available:
            return False

        try:
            key = self._make_key(prefix_hash, token_ids)
            self.engine.store(key, kv_tensor)  # type: ignore[union-attr]
            self._stores += 1
            return True
        except Exception as exc:
            self._store_errors += 1
            logger.warning("[LMCACHE] store failed: %s", exc)
            return False

    # ── Key translation ───────────────────────────────────────────────────────
    #
    # Axropus uses a single SHA256 prefix hash for the whole prefix.
    # LMCache uses block-level keys (16 or 256 tokens per block).
    #
    # TODO(v2): Support partial-prefix matching by splitting AMF prefixes
    # into LMCache block-sized chunks. For v1 we use exact-prefix keys only,
    # which gives 100% compatibility but loses LMCache's partial-prefix
    # matching feature.

    def _make_key(self, prefix_hash: str, token_ids: List[int]) -> Any:
        """Translate Axropus prefix hash → LMCache CacheEngineKey."""
        if not LMCACHE_AVAILABLE or CacheEngineKey is None:
            return None

        # LMCache's CacheEngineKey constructor signature has varied across
        # versions. Try a few known combinations; fall back to positional.
        attempts = [
            # Newer API
            lambda: CacheEngineKey(  # type: ignore[call-arg]
                fmt="axropus",
                model_name="unknown",
                world_size=1,
                worker_id=0,
                chunk_hash=prefix_hash,
            ),
            # Older API
            lambda: CacheEngineKey(  # type: ignore[call-arg]
                prefix_hash=prefix_hash,
                token_count=len(token_ids),
            ),
            # Simplest positional
            lambda: CacheEngineKey(prefix_hash),  # type: ignore[call-arg]
        ]
        last_exc: Optional[Exception] = None
        for attempt in attempts:
            try:
                return attempt()
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            f"LMCache CacheEngineKey constructor incompatible: {last_exc}"
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "available":      self.available,
            "hits":           self._hits,
            "misses":         self._misses,
            "stores":         self._stores,
            "lookup_errors":  self._lookup_errors,
            "store_errors":   self._store_errors,
        }
