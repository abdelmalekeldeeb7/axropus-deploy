"""lmcache_bridge.py — LMCache block-coverage query for AMF routing.

LMCache operates at the KV block level (16-32 tokens/block) and is
integrated directly into vLLM via the KV connector interface.  This
bridge exposes a single function that the AMF router uses to check
block coverage for a given prefix — without duplicating LMCache's
internal state.

Cache hierarchy with LMCache:

  AMF VRAM   (10–50 ms)   ← full sequence, zero H→D, TurboQuant
  LMCache    (100–500 ms) ← block-level, GPU/CPU memory, vLLM-native
  AMF NVMe   (3,044 ms)   ← full sequence, TurboQuant, cold backup
  Cold       (243,610 ms) ← full prefill → saves to all tiers above

LMCache reduces NVMe pressure: blocks that would cause AMF NVMe reads
(cold session but shared prefix blocks) are served from LMCache's
GPU/CPU pool instead.  AMF NVMe becomes a last resort for sessions
with no block overlap at all.

Enable via:
  KORITH_LMCACHE_ENABLED=1
  KORITH_LMCACHE_URL=http://localhost:8100   # LMCache HTTP API (optional)
  KORITH_LMCACHE_BLOCK_SIZE=16              # tokens per KV block
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUTHY


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


# ── LMCache estimated latency model ──────────────────────────────────────────
# From LMCache paper / production data:
#   GPU-resident blocks:  ~50–150 ms  (already in vLLM's block pool)
#   CPU-offloaded blocks: ~150–400 ms (PCIe transfer)
#   NVMe blocks:          ~400–1000 ms (disk read — LMCache's own NVMe tier)
#
# We model the expected latency as a function of block coverage and tier.

_LMCACHE_LATENCY_GPU_MS:  float = 100.0   # mid estimate, GPU-resident blocks
_LMCACHE_LATENCY_CPU_MS:  float = 300.0   # mid estimate, CPU-offloaded blocks
_LMCACHE_LATENCY_NVME_MS: float = 700.0   # mid estimate, LMCache NVMe tier


@dataclass
class LMCacheCoverage:
    """Block coverage result for a given prefix."""
    coverage:    float   # fraction of prefix blocks found [0.0, 1.0]
    tier:        str     # "gpu" | "cpu" | "nvme" | "none"
    latency_ms:  float   # estimated retrieval latency for covered blocks
    block_count: int     # number of covered blocks
    total_blocks: int    # total blocks in prefix


def query_coverage(
    prefix_hash: str,
    num_tokens: int,
    worker_id: str = "",
    block_size: Optional[int] = None,
) -> LMCacheCoverage:
    """Query LMCache for KV block coverage of a given prefix.

    In production this calls the LMCache HTTP API or local gRPC endpoint.
    Falls back to a conservative zero-coverage estimate if LMCache is
    unavailable or disabled — safe because the router treats it as no-op.

    Args:
        prefix_hash:  AMF canonical prefix hash (hex string).
        num_tokens:   Number of tokens in the prefix.
        worker_id:    Dynamo worker ID (LMCache is per-worker in vLLM).
        block_size:   KV block size in tokens (default: KORITH_LMCACHE_BLOCK_SIZE).

    Returns:
        LMCacheCoverage describing how much of the prefix is cached.
    """
    if not _env_bool("KORITH_LMCACHE_ENABLED", False):
        return _no_coverage(num_tokens, block_size)

    bs = block_size or _env_int("KORITH_LMCACHE_BLOCK_SIZE", 16)
    total_blocks = max(1, num_tokens // bs)

    url = os.environ.get("KORITH_LMCACHE_URL", "").strip()
    if url:
        try:
            return _query_http(url, prefix_hash, worker_id, total_blocks, bs)
        except Exception as exc:
            logger.debug("LMCache HTTP query failed (%s): %s — using no-coverage", url, exc)

    # LMCache is running in-process via vLLM KV connector — we can't query it
    # directly without the vLLM internals.  Return no-coverage so the router
    # falls back to Dynamo's kv_overlap signal (which reflects block reuse).
    return _no_coverage(num_tokens, bs)


def estimate_lmcache_cost_ms(num_tokens: int, coverage: LMCacheCoverage) -> float:
    """Estimate total request cost when LMCache serves cached blocks.

    Cost = (uncovered fraction × full prefill cost) + block retrieval latency

    Args:
        num_tokens:  Prefix length in tokens.
        coverage:    LMCacheCoverage from query_coverage().

    Returns:
        Estimated milliseconds to serve the request via LMCache partial hit.
    """
    if coverage.coverage <= 0.0:
        return float("inf")  # no blocks — not a hit

    # Prefill cost for uncovered tokens (linear approximation).
    uncovered_frac  = 1.0 - coverage.coverage
    # 0.046 ms/token is the AMF restore calibration baseline (RAM tier).
    # Prefill is ~5x slower per token than restore → ~0.23 ms/token approx.
    prefill_ms_per_token = 0.23
    uncovered_prefill_ms = uncovered_frac * num_tokens * prefill_ms_per_token

    return uncovered_prefill_ms + coverage.latency_ms


# ── Internal helpers ──────────────────────────────────────────────────────────

def _no_coverage(num_tokens: int, block_size: int) -> LMCacheCoverage:
    total = max(1, num_tokens // max(1, block_size))
    return LMCacheCoverage(
        coverage=0.0, tier="none", latency_ms=float("inf"),
        block_count=0, total_blocks=total,
    )


def _query_http(
    url: str,
    prefix_hash: str,
    worker_id: str,
    total_blocks: int,
    block_size: int,
) -> LMCacheCoverage:
    """Query LMCache REST API for block coverage."""
    import urllib.request, json

    endpoint = url.rstrip("/") + "/v1/coverage"
    payload   = json.dumps({
        "prefix_hash": prefix_hash,
        "worker_id":   worker_id,
        "block_size":  block_size,
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=0.05) as resp:   # 50 ms timeout
        data = json.loads(resp.read())

    covered = int(data.get("covered_blocks", 0))
    tier    = str(data.get("tier", "cpu")).lower()
    coverage = min(1.0, covered / max(1, total_blocks))

    latency = {
        "gpu":  _LMCACHE_LATENCY_GPU_MS,
        "cpu":  _LMCACHE_LATENCY_CPU_MS,
        "nvme": _LMCACHE_LATENCY_NVME_MS,
    }.get(tier, _LMCACHE_LATENCY_CPU_MS)

    return LMCacheCoverage(
        coverage=coverage,
        tier=tier,
        latency_ms=latency,
        block_count=covered,
        total_blocks=total_blocks,
    )
