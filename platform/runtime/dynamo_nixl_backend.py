"""AMF NIXL storage backend for NVIDIA Dynamo KVBM.

Implements the NIXL block-level storage interface (register_volume, put, get,
delete) making AMF the intelligent G3/G4 storage layer behind Dynamo's KV
Block Manager.  Adds correctness validation, ROI-gated admission, and
value-aware eviction on top of what Dynamo sees as opaque blob storage.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .amf_coordinator_client import AmfCoordinatorClient

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUTHY


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NixlBackendConfig:
    """Configuration read from environment variables."""

    snapshot_dir: str = field(default_factory=lambda: os.environ.get("AMF_NIXL_SNAPSHOT_DIR", "/data/amf/snapshots"))
    min_admit_roi: float = field(default_factory=lambda: _env_float("AMF_NIXL_MIN_ADMIT_ROI", 1.5))
    agg_window_ms: float = field(default_factory=lambda: _env_float("AMF_NIXL_AGG_WINDOW_MS", 500.0))
    max_storage_bytes: int = field(default_factory=lambda: _env_int("AMF_NIXL_MAX_STORAGE_BYTES", 100 * 1024 * 1024 * 1024))  # 100 GiB
    eviction_watermark: float = field(default_factory=lambda: _env_float("AMF_NIXL_EVICTION_WATERMARK", 0.85))
    coordinator_url: str = field(default_factory=lambda: os.environ.get("AMF_COORDINATOR_URL", "http://localhost:8500"))
    node_id: str = field(default_factory=lambda: os.environ.get("AMF_NODE_ID", "amf-node-0"))


# ---------------------------------------------------------------------------
# AdmissionGate
# ---------------------------------------------------------------------------


@dataclass
class _AdmissionRecord:
    prefix_id: str
    num_tokens: int
    estimated_prefill_ms: float
    estimated_restore_ms: float
    roi: float
    admitted: bool


def _estimate_prefill_ms(num_tokens: int) -> float:
    """Quadratic approximation for dense-transformer prefill on H200."""
    t = float(num_tokens)
    return max(0.0, 1e-6 * t * t + 0.3 * t)


def _estimate_restore_ms(num_tokens: int, storage_tier: str) -> float:
    """Estimate restore latency from benchmark data.

    Calibration points (817-run average, NVMe tier):
      120K tokens → 11,446 ms
      252K tokens →  4,459 ms
      1M   tokens → 11,601 ms

    Tier multipliers: vram=0.5x, ram=1x, nvme=2x, remote=5x
    """
    tier_mul = {"vram": 0.5, "ram": 1.0, "nvme": 2.0, "remote": 5.0, "G1": 0.5, "G2": 1.0, "G3": 2.0, "G4": 5.0}
    mul = tier_mul.get(storage_tier, 2.0)
    t = float(max(1, num_tokens))
    # Piecewise-linear fit through calibration points; scale to a RAM baseline.
    base_ms = 0.046 * t  # ≈11,500 ms at 250K tokens (RAM reference)
    return max(1.0, base_ms * mul)


class AdmissionGate:
    """ROI-gated admission control for AMF snapshot storage.

    Computes ROI = estimated_prefill_ms / estimated_restore_ms and rejects
    prefixes below the configured min_admit_roi threshold.
    """

    def __init__(self, min_admit_roi: float = 1.5) -> None:
        self._min_roi = max(0.0, float(min_admit_roi))
        self.admission_count = 0
        self.rejection_count = 0
        self.total_bytes_admitted = 0
        self.total_bytes_rejected = 0

    def evaluate(
        self,
        prefix_id: str,
        num_tokens: int,
        storage_tier: str,
        estimated_bytes: int = 0,
    ) -> _AdmissionRecord:
        prefill_ms = _estimate_prefill_ms(num_tokens)
        restore_ms = _estimate_restore_ms(num_tokens, storage_tier)
        roi = prefill_ms / max(1.0, restore_ms)
        admitted = roi >= self._min_roi

        if admitted:
            self.admission_count += 1
            self.total_bytes_admitted += estimated_bytes
        else:
            self.rejection_count += 1
            self.total_bytes_rejected += estimated_bytes

        logger.debug(
            "AdmissionGate prefix=%s tokens=%d roi=%.2f threshold=%.2f admitted=%s",
            prefix_id[:12],
            num_tokens,
            roi,
            self._min_roi,
            admitted,
        )
        return _AdmissionRecord(
            prefix_id=prefix_id,
            num_tokens=num_tokens,
            estimated_prefill_ms=prefill_ms,
            estimated_restore_ms=restore_ms,
            roi=roi,
            admitted=admitted,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "admission_count": self.admission_count,
            "rejection_count": self.rejection_count,
            "total_bytes_admitted": self.total_bytes_admitted,
            "total_bytes_rejected": self.total_bytes_rejected,
        }


# ---------------------------------------------------------------------------
# CorrectnessValidator
# ---------------------------------------------------------------------------


class CorrectnessValidator:
    """Validates runtime context fields before allowing a snapshot restore.

    This is AMF's key differentiator vs LMCache: we reject restores when the
    model configuration has changed (different rope scaling, quantisation, etc.)
    rather than silently serving stale KV state.
    """

    def __init__(self) -> None:
        self._pass_count = 0
        self._fail_count = 0

    def validate(
        self,
        stored_meta: Dict[str, Any],
        request_meta: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check stored metadata against the request's runtime context.

        Returns (ok, reason).  If ok is False, the caller should treat this as
        a cache miss and trigger recompute.
        """
        checks = [
            ("model_hash", True),
            ("kv_version", True),
            ("rope_base_bits", False),
            ("rope_scale_bits", False),
            ("sampling_hash", False),
        ]

        for key, required in checks:
            stored_val = stored_meta.get(key)
            request_val = request_meta.get(key)

            if stored_val is None and request_val is None:
                continue
            if stored_val is None and not required:
                continue
            if request_val is None and not required:
                continue

            if stored_val != request_val:
                reason = f"{key} mismatch: stored={stored_val!r} request={request_val!r}"
                self._fail_count += 1
                logger.debug("CorrectnessValidator FAIL: %s", reason)
                return False, reason

        self._pass_count += 1
        return True, "ok"

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "pass_count": self._pass_count,
            "fail_count": self._fail_count,
        }


# ---------------------------------------------------------------------------
# BlockToSnapshotAggregator
# ---------------------------------------------------------------------------


@dataclass
class _BlockBuffer:
    prefix_id: str
    blocks: List[Tuple[str, bytes, Dict[str, Any]]]  # (block_id, data, meta)
    total_bytes: int
    num_tokens: int
    model_id: str
    tenant_id: str
    storage_tier: str
    first_put_ms: float
    metadata: Dict[str, Any]


class BlockToSnapshotAggregator:
    """Buffers Dynamo block-level puts into atomic AMF prefix snapshots.

    Dynamo sends individual 16-256 token blocks.  AMF stores whole prefixes.
    This class collects related blocks for `agg_window_ms` and flushes them
    as a single snapshot.  Flushed snapshots are passed to a callback.
    """

    def __init__(
        self,
        agg_window_ms: float = 500.0,
        on_flush: Optional[Callable[[_BlockBuffer], None]] = None,
    ) -> None:
        self._window_ms = max(10.0, float(agg_window_ms))
        self._on_flush = on_flush
        self._buffers: Dict[str, _BlockBuffer] = {}
        self._mu = threading.Lock()
        self._flush_task: Optional[asyncio.Task] = None  # set in async context

    def _prefix_for_block(self, block_id: str, metadata: Dict[str, Any]) -> str:
        """Derive a stable prefix ID from the block's context."""
        token_hash = str(metadata.get("token_hash", "") or block_id)
        worker_id = str(metadata.get("worker_id", "") or "")
        key = f"{worker_id}:{token_hash}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def put(
        self,
        block_id: str,
        kv_data: bytes,
        metadata: Dict[str, Any],
    ) -> None:
        """Buffer a block.  May flush synchronously if window exceeded."""
        prefix_id = self._prefix_for_block(block_id, metadata)
        now_ms = time.time() * 1000

        with self._mu:
            if prefix_id not in self._buffers:
                self._buffers[prefix_id] = _BlockBuffer(
                    prefix_id=prefix_id,
                    blocks=[],
                    total_bytes=0,
                    num_tokens=0,
                    model_id=str(metadata.get("model_id", "") or ""),
                    tenant_id=str(metadata.get("tenant_id", "default") or "default"),
                    storage_tier=str(metadata.get("storage_tier", "G3") or "G3"),
                    first_put_ms=now_ms,
                    metadata=dict(metadata),
                )
            buf = self._buffers[prefix_id]
            buf.blocks.append((block_id, kv_data, metadata))
            buf.total_bytes += len(kv_data)
            buf.num_tokens += int(metadata.get("num_tokens", 0) or 0)

            if (now_ms - buf.first_put_ms) >= self._window_ms:
                self._flush_locked(prefix_id)

    def flush_all(self) -> None:
        """Force flush all pending buffers (e.g. on shutdown)."""
        with self._mu:
            for prefix_id in list(self._buffers.keys()):
                self._flush_locked(prefix_id)

    def _flush_locked(self, prefix_id: str) -> None:
        buf = self._buffers.pop(prefix_id, None)
        if buf is None or not buf.blocks:
            return
        if self._on_flush is not None:
            try:
                self._on_flush(buf)
            except Exception as exc:
                logger.exception("BlockToSnapshotAggregator flush callback error: %s", exc)


# ---------------------------------------------------------------------------
# Snapshot storage helpers
# ---------------------------------------------------------------------------


@dataclass
class _SnapshotEntry:
    prefix_id: str
    snapshot_path: str
    num_tokens: int
    size_bytes: int
    model_hash: str
    kv_version: str
    metadata: Dict[str, Any]
    stored_at: float
    hit_count: int = 0
    roi: float = 0.0
    value_per_byte: float = 0.0

    def update_value(self) -> None:
        prefill_ms = _estimate_prefill_ms(self.num_tokens)
        restore_ms = _estimate_restore_ms(self.num_tokens, self.metadata.get("storage_tier", "G3"))
        self.roi = prefill_ms / max(1.0, restore_ms)
        self.value_per_byte = (prefill_ms * (1 + self.hit_count)) / max(1, self.size_bytes)


# ---------------------------------------------------------------------------
# AmfNixlBackend
# ---------------------------------------------------------------------------


class AmfNixlBackend:
    """AMF implementation of the NIXL storage interface.

    Dynamo's KVBM calls register_volume(), put(), get(), delete() and stats()
    on this object when AMF is configured as the G3/G4 storage tier.
    """

    def __init__(self, config: Optional[NixlBackendConfig] = None) -> None:
        self._config = config or NixlBackendConfig()
        self._coordinator = AmfCoordinatorClient(self._config.coordinator_url)
        self._admission = AdmissionGate(min_admit_roi=self._config.min_admit_roi)
        self._validator = CorrectnessValidator()

        self._mu = threading.Lock()
        self._entries: Dict[str, _SnapshotEntry] = {}        # prefix_id → entry
        self._block_to_prefix: Dict[str, str] = {}           # block_id → prefix_id
        self._total_bytes = 0
        self._hit_count = 0
        self._miss_count = 0
        self._volume_registered = False

        self._aggregator = BlockToSnapshotAggregator(
            agg_window_ms=self._config.agg_window_ms,
            on_flush=self._on_snapshot_ready,
        )

        logger.info(
            "AmfNixlBackend init: snapshot_dir=%s min_roi=%.1f agg_window=%.0fms max_bytes=%d",
            self._config.snapshot_dir,
            self._config.min_admit_roi,
            self._config.agg_window_ms,
            self._config.max_storage_bytes,
        )

    # ------------------------------------------------------------------
    # NIXL interface
    # ------------------------------------------------------------------

    def register_volume(self, volume_descriptor: Dict[str, Any]) -> bool:
        """Register AMF's snapshot directory as a NIXL volume."""
        snap_dir = Path(self._config.snapshot_dir)
        try:
            snap_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create snapshot dir %s: %s", snap_dir, exc)
            return False

        self._volume_registered = True
        logger.info("AmfNixlBackend: registered volume at %s", snap_dir)
        return True

    def unregister_volume(self) -> None:
        """Cleanup — flush pending aggregation buffers."""
        self._aggregator.flush_all()
        self._volume_registered = False
        logger.info("AmfNixlBackend: volume unregistered")

    def put(
        self,
        block_id: str,
        kv_data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Receive a KV block from KVBM.

        Blocks are aggregated into prefix snapshots and admitted only when
        their ROI exceeds the configured threshold.
        """
        meta = metadata or {}
        self._aggregator.put(block_id, kv_data, meta)
        return True

    def get(
        self,
        block_id: str,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[bytes]:
        """Return a KV block if AMF holds a valid snapshot for it.

        Performs correctness validation before returning data.  Returns None
        on miss or validation failure (KVBM falls back to recompute).
        """
        request_meta = request_metadata or {}

        with self._mu:
            prefix_id = self._block_to_prefix.get(block_id)
            if prefix_id is None:
                self._miss_count += 1
                return None

            entry = self._entries.get(prefix_id)
            if entry is None:
                self._block_to_prefix.pop(block_id, None)
                self._miss_count += 1
                return None

        # Correctness validation (outside lock for speed).
        ok, reason = self._validator.validate(entry.metadata, request_meta)
        if not ok:
            logger.info("AmfNixlBackend correctness FAIL block=%s: %s", block_id, reason)
            self._miss_count += 1
            return None

        # Load snapshot data from disk.
        try:
            data = Path(entry.snapshot_path).read_bytes()
        except OSError as exc:
            logger.warning("AmfNixlBackend cannot read snapshot %s: %s", entry.snapshot_path, exc)
            self._miss_count += 1
            return None

        with self._mu:
            entry.hit_count += 1
            entry.update_value()
        self._hit_count += 1
        return data

    def delete(self, block_id: str) -> bool:
        """Remove a block and its associated snapshot if no other blocks reference it."""
        with self._mu:
            prefix_id = self._block_to_prefix.pop(block_id, None)
            if prefix_id is None:
                return False

            # Check if any other blocks still reference this prefix.
            refs = sum(1 for pid in self._block_to_prefix.values() if pid == prefix_id)
            if refs > 0:
                return True  # Other blocks still alive.

            entry = self._entries.pop(prefix_id, None)
            if entry is not None:
                self._total_bytes -= entry.size_bytes

        if entry is not None:
            try:
                Path(entry.snapshot_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("AmfNixlBackend: could not delete snapshot %s: %s", entry.snapshot_path, exc)

        return True

    def stats(self) -> Dict[str, Any]:
        with self._mu:
            entry_count = len(self._entries)
            total_bytes = self._total_bytes

        return {
            "entry_count": entry_count,
            "total_bytes": total_bytes,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / max(1, self._hit_count + self._miss_count),
            "volume_registered": self._volume_registered,
            "admission": self._admission.stats,
            "validation": self._validator.stats,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_snapshot_ready(self, buf: _BlockBuffer) -> None:
        """Callback from BlockToSnapshotAggregator when a prefix is flushed."""
        # Admission gate.
        record = self._admission.evaluate(
            prefix_id=buf.prefix_id,
            num_tokens=buf.num_tokens,
            storage_tier=buf.storage_tier,
            estimated_bytes=buf.total_bytes,
        )
        if not record.admitted:
            logger.debug(
                "AmfNixlBackend admission REJECT prefix=%s tokens=%d roi=%.2f",
                buf.prefix_id[:12],
                buf.num_tokens,
                record.roi,
            )
            return

        # Check storage budget — evict if needed.
        self._maybe_evict(buf.total_bytes)

        # Determine snapshot path.
        snap_dir = Path(self._config.snapshot_dir) / buf.tenant_id / buf.model_id
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{buf.prefix_id}.amf"

        # Write concatenated block data to snapshot.
        combined = b"".join(data for _, data, _ in buf.blocks)
        try:
            snap_path.write_bytes(combined)
        except OSError as exc:
            logger.error("AmfNixlBackend: failed to write snapshot %s: %s", snap_path, exc)
            return

        # Build metadata for validation.
        meta = dict(buf.metadata)
        meta.update({
            "model_id": buf.model_id,
            "num_tokens": buf.num_tokens,
            "storage_tier": buf.storage_tier,
            "estimated_prefill_ms": record.estimated_prefill_ms,
            "estimated_restore_ms": record.estimated_restore_ms,
            "roi": record.roi,
            "block_count": len(buf.blocks),
            "source": "nixl_backend",
        })

        entry = _SnapshotEntry(
            prefix_id=buf.prefix_id,
            snapshot_path=str(snap_path),
            num_tokens=buf.num_tokens,
            size_bytes=buf.total_bytes,
            model_hash=str(meta.get("model_hash", "") or ""),
            kv_version=str(meta.get("kv_version", "") or ""),
            metadata=meta,
            stored_at=time.time(),
        )
        entry.update_value()

        with self._mu:
            self._entries[buf.prefix_id] = entry
            for block_id, _, _ in buf.blocks:
                self._block_to_prefix[block_id] = buf.prefix_id
            self._total_bytes += buf.total_bytes

        # Register with AMF coordinator.
        self._coordinator.register(
            prefix_hash=buf.prefix_id,
            tenant_id=buf.tenant_id,
            node_id=self._config.node_id,
            worker_id=str(meta.get("worker_id", "") or ""),
            metadata=meta,
        )

        logger.info(
            "AmfNixlBackend stored prefix=%s tokens=%d bytes=%d roi=%.2f",
            buf.prefix_id[:12],
            buf.num_tokens,
            buf.total_bytes,
            record.roi,
        )

    def _maybe_evict(self, incoming_bytes: int) -> None:
        """Evict lowest-value entries to make room for new snapshot."""
        watermark = self._config.max_storage_bytes * self._config.eviction_watermark
        with self._mu:
            if self._total_bytes + incoming_bytes <= watermark:
                return
            # Sort by value_per_byte ascending — evict lowest value first.
            sorted_entries = sorted(self._entries.values(), key=lambda e: e.value_per_byte)

        for entry in sorted_entries:
            with self._mu:
                if self._total_bytes + incoming_bytes <= watermark:
                    break
                if entry.prefix_id not in self._entries:
                    continue
                self._entries.pop(entry.prefix_id, None)
                self._total_bytes -= entry.size_bytes
                # Remove block mappings.
                stale = [bid for bid, pid in self._block_to_prefix.items() if pid == entry.prefix_id]
                for bid in stale:
                    self._block_to_prefix.pop(bid, None)

            try:
                Path(entry.snapshot_path).unlink(missing_ok=True)
            except OSError:
                pass

            logger.debug(
                "AmfNixlBackend evicted prefix=%s (value_per_byte=%.4f)",
                entry.prefix_id[:12],
                entry.value_per_byte,
            )
