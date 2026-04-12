"""lmcache_adapter.py — Thin adapter for the LMCache G3/G4/G5 tiers.

LMCache ships its own persistence layers (CPU, NVMe, remote). AMF uses
it for tiers below G1 (the compressed VRAM pool) and G2 (vLLM's active
paged attention). This module provides a minimal interface that the
tiered router can target without taking a hard dependency on LMCache:
if LMCache is not installed the adapter degrades to a no-op.

Interface:

    class LMCacheAdapter:
        enabled: bool
        def lookup(prefix_hash: str) -> Optional[torch.Tensor]
        def store(prefix_hash: str, kv: torch.Tensor) -> bool
        def evict(prefix_hash: str) -> bool

The router treats any non-``None`` return from ``lookup`` as a G3 hit.
The returned tensor is the same five-dim shape the compressed VRAM pool
expects for ``put_from_raw``.

Environment variables:
    AXROPUS_LMCACHE_ENABLE     set to ``1`` to enable the LMCache tier.
    AXROPUS_LMCACHE_BACKEND    one of ``cpu``, ``nvme``, ``remote``.
    AXROPUS_LMCACHE_PATH       filesystem path for ``nvme``.
    AXROPUS_LMCACHE_URL        remote URL for ``remote``.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class LMCacheAdapter:
    """Adapter over the LMCache library with graceful fallback."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        backend: Optional[str] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        env_enabled = os.environ.get("AXROPUS_LMCACHE_ENABLE", "").lower() in (
            "1", "true", "yes", "on",
        )
        self.enabled = enabled if enabled is not None else env_enabled
        self.backend = backend or os.environ.get("AXROPUS_LMCACHE_BACKEND", "cpu")
        self.path = Path(path or os.environ.get("AXROPUS_LMCACHE_PATH", "/tmp/axropus_lmcache"))
        self.url = url or os.environ.get("AXROPUS_LMCACHE_URL", "")

        self._lmcache_engine: Any = None  # real LMCache handle when available
        self._cpu_store: Dict[str, torch.Tensor] = {}
        self._stat_hits = 0
        self._stat_misses = 0
        self._stat_stores = 0
        self._stat_bytes_in = 0
        self._stat_bytes_out = 0

        if self.enabled:
            self._try_init_lmcache()

        # Filesystem backend: make sure the directory exists.
        if self.enabled and self.backend == "nvme":
            self.path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "LMCacheAdapter enabled=%s backend=%s path=%s url=%s library=%s",
            self.enabled,
            self.backend,
            self.path,
            self.url,
            "native" if self._lmcache_engine is not None else "fallback",
        )

    # ── LMCache library probe ──────────────────────────────────────────────

    def _try_init_lmcache(self) -> None:
        try:
            import lmcache  # type: ignore
            from lmcache.engine import LMCacheEngine  # type: ignore
            from lmcache.config import LMCacheEngineConfig  # type: ignore

            cfg = LMCacheEngineConfig.from_defaults()
            if self.backend == "cpu":
                cfg.use_cpu = True
            elif self.backend == "nvme":
                cfg.use_disk = True
                cfg.disk_path = str(self.path)
            elif self.backend == "remote":
                cfg.use_remote = True
                cfg.remote_url = self.url
            self._lmcache_engine = LMCacheEngine(cfg)
            logger.info("LMCache engine initialised (backend=%s)", self.backend)
        except Exception as exc:
            logger.info("LMCache library unavailable (%s); using fallback store", exc)
            self._lmcache_engine = None

    # ── Public API ─────────────────────────────────────────────────────────

    def lookup(self, prefix_hash: str) -> Optional[torch.Tensor]:
        """Return a cached KV tensor or ``None`` on miss."""
        if not self.enabled:
            self._stat_misses += 1
            return None

        t0 = time.monotonic()
        kv = self._native_lookup(prefix_hash)
        if kv is None:
            kv = self._fallback_lookup(prefix_hash)

        if kv is None:
            self._stat_misses += 1
            return None

        self._stat_hits += 1
        self._stat_bytes_out += kv.numel() * kv.element_size()
        logger.debug(
            "LMCacheAdapter hit %s in %.1f ms", prefix_hash[:12], (time.monotonic() - t0) * 1000
        )
        return kv

    def store(self, prefix_hash: str, kv: torch.Tensor) -> bool:
        if not self.enabled:
            return False
        ok = self._native_store(prefix_hash, kv)
        if not ok:
            ok = self._fallback_store(prefix_hash, kv)
        if ok:
            self._stat_stores += 1
            self._stat_bytes_in += kv.numel() * kv.element_size()
        return ok

    def evict(self, prefix_hash: str) -> bool:
        if not self.enabled:
            return False
        # Native LMCache does not always expose eviction; for the fallback
        # store we just pop the key.
        removed = prefix_hash in self._cpu_store
        self._cpu_store.pop(prefix_hash, None)
        return removed

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled":    self.enabled,
            "backend":    self.backend,
            "hits":       self._stat_hits,
            "misses":     self._stat_misses,
            "stores":     self._stat_stores,
            "bytes_in":   self._stat_bytes_in,
            "bytes_out":  self._stat_bytes_out,
        }

    # ── Native LMCache paths ───────────────────────────────────────────────

    def _native_lookup(self, prefix_hash: str) -> Optional[torch.Tensor]:
        if self._lmcache_engine is None:
            return None
        try:
            blob = self._lmcache_engine.get(prefix_hash)
            if blob is None:
                return None
            return torch.as_tensor(blob)
        except Exception as exc:  # pragma: no cover - depends on library version
            logger.warning("LMCache native lookup failed: %s", exc)
            return None

    def _native_store(self, prefix_hash: str, kv: torch.Tensor) -> bool:
        if self._lmcache_engine is None:
            return False
        try:
            self._lmcache_engine.put(prefix_hash, kv)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("LMCache native store failed: %s", exc)
            return False

    # ── Fallback (in-process) paths ────────────────────────────────────────

    def _fallback_lookup(self, prefix_hash: str) -> Optional[torch.Tensor]:
        if self.backend == "nvme":
            fp = self.path / f"{prefix_hash}.pt"
            if fp.exists():
                try:
                    return torch.load(fp, map_location="cpu")
                except Exception as exc:  # pragma: no cover
                    logger.warning("nvme lookup failed: %s", exc)
            return None
        # CPU / remote fallback → in-process dict.
        return self._cpu_store.get(prefix_hash)

    def _fallback_store(self, prefix_hash: str, kv: torch.Tensor) -> bool:
        if self.backend == "nvme":
            try:
                fp = self.path / f"{prefix_hash}.pt"
                torch.save(kv.detach().cpu(), fp)
                return True
            except Exception as exc:  # pragma: no cover
                logger.warning("nvme store failed: %s", exc)
                return False
        self._cpu_store[prefix_hash] = kv.detach().cpu()
        return True


__all__ = ["LMCacheAdapter"]
