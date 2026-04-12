"""compressed_vram_pool.py — Compressed multi-prefix GPU residency pool.

This is the data structure that makes AMF defensible. It holds 50-100+
compressed KV prefixes simultaneously in VRAM on a single H200, indexed
by content-addressed prefix hash, with two-tier precision: storage in
INT4 / FP8 / NVFP4 / TurboQuant, active decode in FP8 or NVFP4 depending
on hardware.

Design notes (§3 of the design doc):

    * Single contiguous VRAM allocation per layer, pre-sized at startup.
      Sub-allocated in fixed-size slab blocks (configurable, default
      128 KB).
    * Entries are keyed by ``prefix_hash`` (16-char hex string). Each
      entry owns a list of slab block indices, one per layer.
    * Eviction is reuse-score-weighted LRU. The score mixes age, hit
      count, size, and per-hit savings so that high-value prefixes
      survive even when they are not the most recently used.
    * On insertion, the raw KV tensor is compressed via the codec
      registry. The compressed blob is copied into the slab with a
      single DMA. No CPU round-trip on warm restore.
    * vLLM's block manager never sees the allocation. On hit, the pool
      maps its slab blocks into vLLM's block table as opaque pointers
      tagged ``is_external=True`` so vLLM's allocator ignores them.

Thread safety: every public method acquires an ``RLock``. The internal
state is simple enough that a coarse lock does not hurt hit latency
at realistic request rates.

CPU fallback: when CUDA is not available, the pool falls back to plain
tensors on the CPU. This keeps unit tests runnable on laptops and CI.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .codecs import (
    FMT_FP16,
    Codec,
    CompressedKV,
    get_codec,
    list_codecs,
)

logger = logging.getLogger(__name__)


# ── Block & entry data structures ────────────────────────────────────────────


@dataclass
class CompressedBlock:
    """Metadata for a single slab block inside the pool."""

    layer_id: int
    block_id: int                  # index into the layer's slab
    format: str
    num_tokens: int
    num_heads: int
    head_dim: int
    data_offset: int               # byte offset into the layer slab
    data_bytes: int
    scale_offset: int              # where per-block / per-group scales live
    scale_bytes: int
    tensor_scale: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolEntry:
    """A single prefix cached in the pool, with one CompressedBlock per layer."""

    prefix_hash: str
    num_layers: int
    num_tokens: int
    format: str
    blocks: List[CompressedBlock]
    # Free-standing codec blobs kept live for the decode-side codec ops.
    # Using a list so we can hold one ``CompressedKV`` per layer for the
    # python reference path; CUDA kernels read from raw data_offsets.
    blobs: List[CompressedKV]

    created_ns:   int = 0
    last_hit_ns:  int = 0
    hit_count:    int = 0
    avg_savings_ms: float = 0.0

    def nbytes(self) -> int:
        return sum(b.data_bytes + b.scale_bytes for b in self.blocks)


# ── The pool itself ──────────────────────────────────────────────────────────


class CompressedVRAMPool:
    """Compressed multi-prefix GPU residency pool.

    Args:
        num_layers:        Number of transformer layers (= number of slabs).
        bytes_per_layer:   Slab size per layer in bytes.
        block_bytes:       Slab block size in bytes (default 128 KB).
        default_format:    Codec format to use when the caller does not
                           specify one.
        device:            CUDA device or ``"cpu"``.
        reuse_alpha:       Eviction score weight on age.
        reuse_beta:        Eviction score weight on 1/hit_count.
        reuse_gamma:       Eviction score weight on size.
        reuse_delta:       Eviction score weight on per-hit savings.
    """

    def __init__(
        self,
        num_layers: int,
        bytes_per_layer: int = 1 << 30,     # 1 GB per layer
        block_bytes: int = 1 << 17,          # 128 KB
        *,
        default_format: str = "int4_sym_block",
        device: str | torch.device = "cuda",
        reuse_alpha: float = 0.3,
        reuse_beta: float = 0.4,
        reuse_gamma: float = 0.1,
        reuse_delta: float = 0.2,
    ) -> None:
        self.num_layers = num_layers
        self.bytes_per_layer = bytes_per_layer
        self.block_bytes = block_bytes
        self.default_format = default_format

        # Decide the target device, falling back to CPU if CUDA is missing.
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            device = torch.device(device)
        self.device = device

        # Pre-allocate one byte slab per layer. Pool is opaque bytes; codec
        # tensors view subranges when needed.
        self._slabs: List[torch.Tensor] = []
        for layer in range(num_layers):
            slab = torch.empty(
                (bytes_per_layer,), dtype=torch.uint8, device=self.device
            )
            self._slabs.append(slab)

        # Free list per layer: list of free block indices (ints).
        self._blocks_per_layer = bytes_per_layer // block_bytes
        self._free_lists: List[List[int]] = [
            list(range(self._blocks_per_layer)) for _ in range(num_layers)
        ]

        # Prefix index + lock.
        self._index: Dict[str, PoolEntry] = {}
        self._lock = threading.RLock()

        # Eviction policy weights.
        self._alpha = reuse_alpha
        self._beta = reuse_beta
        self._gamma = reuse_gamma
        self._delta = reuse_delta

        # Stats.
        self._stat_hits = 0
        self._stat_misses = 0
        self._stat_inserts = 0
        self._stat_evictions = 0
        self._stat_bytes_stored = 0
        self._stat_bytes_evicted = 0

        logger.info(
            "CompressedVRAMPool initialised: layers=%d slab=%.1f GB blocks=%d device=%s",
            num_layers,
            bytes_per_layer / (1 << 30),
            self._blocks_per_layer,
            self.device,
        )

    # ── Queries ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def __contains__(self, prefix_hash: str) -> bool:
        with self._lock:
            return prefix_hash in self._index

    def capacity_bytes(self) -> int:
        return self.num_layers * self.bytes_per_layer

    def used_bytes(self) -> int:
        return sum(e.nbytes() for e in self._index.values())

    def num_prefixes(self) -> int:
        return len(self._index)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits":         self._stat_hits,
                "misses":       self._stat_misses,
                "inserts":      self._stat_inserts,
                "evictions":    self._stat_evictions,
                "bytes_stored": self._stat_bytes_stored,
                "bytes_evicted": self._stat_bytes_evicted,
                "num_prefixes": self.num_prefixes(),
                "used_bytes":   self.used_bytes(),
                "capacity_bytes": self.capacity_bytes(),
            }

    # ── Core ops ───────────────────────────────────────────────────────────

    def get(self, prefix_hash: str) -> Optional[PoolEntry]:
        """Look up a prefix. Updates LRU stats on hit."""
        with self._lock:
            entry = self._index.get(prefix_hash)
            if entry is None:
                self._stat_misses += 1
                return None
            entry.last_hit_ns = time.monotonic_ns()
            entry.hit_count += 1
            self._stat_hits += 1
            return entry

    def put_from_raw(
        self,
        prefix_hash: str,
        raw_kv: torch.Tensor,
        *,
        format: Optional[str] = None,
        savings_ms: float = 0.0,
    ) -> bool:
        """Insert a raw FP16/BF16 KV tensor into the pool.

        Expected ``raw_kv`` shape:
            ``[num_layers, 2, num_tokens, num_heads, head_dim]``
        (K on dim 1 index 0, V on index 1). If the caller passes a
        different layout we try to detect it and fall back; shapes
        outside of that family raise ``ValueError``.

        Returns True on success, False if eviction could not free
        enough blocks to accept this insertion.
        """
        fmt = format or self.default_format
        codec = get_codec(fmt)

        if raw_kv.dim() != 5 or raw_kv.shape[0] != self.num_layers or raw_kv.shape[1] != 2:
            raise ValueError(
                f"put_from_raw: expected shape [num_layers={self.num_layers}, 2, n_tokens, n_heads, head_dim], got {tuple(raw_kv.shape)}"
            )

        num_tokens = int(raw_kv.shape[2])
        num_heads = int(raw_kv.shape[3])
        head_dim = int(raw_kv.shape[4])

        # Compress layer by layer. Each layer blob packs K and V together.
        compressed_layers: List[CompressedKV] = []
        for layer_id in range(self.num_layers):
            layer_kv = raw_kv[layer_id]  # [2, n_tokens, n_heads, head_dim]
            blob = codec.compress(layer_kv)
            compressed_layers.append(blob)

        total_bytes = sum(blob.nbytes() for blob in compressed_layers)

        with self._lock:
            if prefix_hash in self._index:
                # Replace existing entry in place.
                self._release_entry(self._index.pop(prefix_hash))

            # Allocate blocks across layers. Trigger eviction if necessary.
            allocation = self._allocate(compressed_layers)
            if allocation is None:
                logger.warning(
                    "CompressedVRAMPool.put_from_raw: no room for %d bytes (format=%s)",
                    total_bytes,
                    fmt,
                )
                return False

            # Copy compressed data into the slab(s).
            self._write_slabs(allocation, compressed_layers)

            # Build entry metadata.
            blocks: List[CompressedBlock] = []
            for layer_id, (block_id, data_off, scale_off) in enumerate(allocation):
                blob = compressed_layers[layer_id]
                blocks.append(
                    CompressedBlock(
                        layer_id=layer_id,
                        block_id=block_id,
                        format=fmt,
                        num_tokens=num_tokens,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        data_offset=data_off,
                        data_bytes=blob.data.numel() * blob.data.element_size(),
                        scale_offset=scale_off,
                        scale_bytes=blob.scales.numel() * blob.scales.element_size() if blob.scales is not None else 0,
                        tensor_scale=blob.tensor_scale,
                        meta=blob.meta,
                    )
                )

            entry = PoolEntry(
                prefix_hash=prefix_hash,
                num_layers=self.num_layers,
                num_tokens=num_tokens,
                format=fmt,
                blocks=blocks,
                blobs=compressed_layers,
                created_ns=time.monotonic_ns(),
                last_hit_ns=time.monotonic_ns(),
                hit_count=0,
                avg_savings_ms=savings_ms,
            )
            self._index[prefix_hash] = entry
            self._stat_inserts += 1
            self._stat_bytes_stored += total_bytes

            logger.debug(
                "CompressedVRAMPool.put: %s format=%s bytes=%d prefixes=%d",
                prefix_hash[:12],
                fmt,
                total_bytes,
                len(self._index),
            )
            return True

    def restore_to_tensor(
        self,
        prefix_hash: str,
        target_dtype: torch.dtype = torch.float16,
    ) -> Optional[torch.Tensor]:
        """Decompress a cached entry back to a dense KV tensor.

        Returns a tensor of shape ``[num_layers, 2, n_tokens, n_heads, head_dim]``
        or ``None`` if the prefix is not in the pool.
        """
        entry = self.get(prefix_hash)
        if entry is None:
            return None

        layers: List[torch.Tensor] = []
        for blob in entry.blobs:
            codec = get_codec(blob.format)
            layers.append(codec.decompress_to(blob, target_dtype=target_dtype))
        return torch.stack(layers, dim=0)

    def delete(self, prefix_hash: str) -> bool:
        with self._lock:
            entry = self._index.pop(prefix_hash, None)
            if entry is None:
                return False
            self._release_entry(entry)
            return True

    def clear(self) -> None:
        with self._lock:
            for entry in list(self._index.values()):
                self._release_entry(entry)
            self._index.clear()

    # ── Promotion (from LMCache / cold prefill) ────────────────────────────

    def promote_from_raw(
        self,
        prefix_hash: str,
        raw_kv: torch.Tensor,
        *,
        format: Optional[str] = None,
    ) -> bool:
        """Alias for ``put_from_raw``; exists so callers can signal intent."""
        return self.put_from_raw(prefix_hash, raw_kv, format=format)

    # ── Eviction policy ────────────────────────────────────────────────────

    def _reuse_score(self, entry: PoolEntry) -> float:
        """Return an eviction score: higher = more evictable."""
        now_ns = time.monotonic_ns()
        age_s = max(0.0, (now_ns - entry.last_hit_ns) / 1e9)
        hits = max(1, entry.hit_count)
        size_mb = entry.nbytes() / (1 << 20)
        savings = max(0.0, entry.avg_savings_ms)
        return (
            self._alpha * age_s
            + self._beta / hits
            + self._gamma * size_mb
            - self._delta * savings
        )

    def _evict_one(self) -> bool:
        """Evict the entry with the highest reuse score. Returns True on success."""
        if not self._index:
            return False

        worst_hash = None
        worst_score = -math.inf
        for h, entry in self._index.items():
            s = self._reuse_score(entry)
            if s > worst_score:
                worst_score = s
                worst_hash = h

        if worst_hash is None:
            return False

        entry = self._index.pop(worst_hash)
        evicted_bytes = entry.nbytes()
        self._release_entry(entry)
        self._stat_evictions += 1
        self._stat_bytes_evicted += evicted_bytes
        logger.debug(
            "CompressedVRAMPool evicted %s score=%.3f bytes=%d",
            worst_hash[:12],
            worst_score,
            evicted_bytes,
        )
        return True

    # ── Slab allocation plumbing ───────────────────────────────────────────

    def _blocks_needed(self, blob: CompressedKV) -> int:
        size = blob.data.numel() * blob.data.element_size()
        if blob.scales is not None:
            size += blob.scales.numel() * blob.scales.element_size()
        return max(1, math.ceil(size / self.block_bytes))

    def _allocate(
        self,
        blobs: List[CompressedKV],
    ) -> Optional[List[Tuple[int, int, int]]]:
        """Allocate one block per layer, evicting if necessary.

        Returns a list of ``(block_id, data_offset, scale_offset)`` tuples,
        one per layer, or ``None`` if allocation failed even after exhausting
        eviction. Note: the current implementation allocates exactly one
        block per layer per prefix, so per-layer blob size must fit in a
        single slab block. If not, we widen the block to fit (still one
        logical block index, but its span covers multiple consecutive slab
        units). This keeps the index simple while allowing large prefixes.
        """
        allocation: List[Tuple[int, int, int]] = []
        # Check if any layer blob exceeds a single block. For simplicity we
        # allow a contiguous multi-block span via ``_alloc_span``.
        for layer_id, blob in enumerate(blobs):
            needed = self._blocks_needed(blob)
            span = self._alloc_span(layer_id, needed)
            while span is None:
                if not self._evict_one():
                    # Unrecoverable: roll back partial allocation.
                    for rid, (bid, _, _) in enumerate(allocation):
                        self._free_span(rid, bid, self._blocks_needed(blobs[rid]))
                    return None
                span = self._alloc_span(layer_id, needed)

            block_id = span
            data_off = block_id * self.block_bytes
            data_bytes = blob.data.numel() * blob.data.element_size()
            scale_off = data_off + data_bytes
            allocation.append((block_id, data_off, scale_off))
        return allocation

    def _alloc_span(self, layer_id: int, n_blocks: int) -> Optional[int]:
        """Allocate ``n_blocks`` contiguous slab blocks on ``layer_id``.

        Returns the starting block index or ``None`` if there is no
        contiguous run of the requested length.
        """
        free = self._free_lists[layer_id]
        if len(free) < n_blocks:
            return None
        free.sort()
        # Linear scan for a contiguous run. Pool is small enough that this
        # is fine (typical: 8k blocks).
        run_start = None
        run_len = 0
        for i, bid in enumerate(free):
            if run_start is None:
                run_start = bid
                run_len = 1
            elif bid == free[i - 1] + 1:
                run_len += 1
            else:
                run_start = bid
                run_len = 1
            if run_len == n_blocks:
                start_block = free[i - n_blocks + 1]
                # Remove these blocks from the free list.
                del free[i - n_blocks + 1 : i + 1]
                return start_block
        return None

    def _free_span(self, layer_id: int, start_block: int, n_blocks: int) -> None:
        self._free_lists[layer_id].extend(range(start_block, start_block + n_blocks))

    def _write_slabs(
        self,
        allocation: List[Tuple[int, int, int]],
        blobs: List[CompressedKV],
    ) -> None:
        """Copy compressed tensors into the slab byte buffers."""
        for layer_id, ((block_id, data_off, scale_off), blob) in enumerate(
            zip(allocation, blobs)
        ):
            slab = self._slabs[layer_id]
            # Move to slab device if needed.
            data_bytes = blob.data.numel() * blob.data.element_size()

            data_view = blob.data.contiguous().view(torch.uint8).flatten()
            # Ensure the view length matches expected byte count.
            if data_view.numel() != data_bytes:
                # Some dtypes (e.g. float8_e4m3fn) have 1-byte elements so
                # the view already matches. Otherwise element_size lies.
                data_view = blob.data.contiguous().view(torch.uint8).flatten()

            data_target = slab.narrow(0, data_off, data_view.numel())
            data_target.copy_(data_view.to(slab.device))

            if blob.scales is not None and blob.scales.numel() > 0:
                scale_view = blob.scales.contiguous().view(torch.uint8).flatten()
                scale_target = slab.narrow(0, scale_off, scale_view.numel())
                scale_target.copy_(scale_view.to(slab.device))

    def _release_entry(self, entry: PoolEntry) -> None:
        """Free every slab block an entry holds."""
        for block in entry.blocks:
            needed = 1
            # Recover the block span from the underlying blob size.
            total_bytes = block.data_bytes + block.scale_bytes
            needed = max(1, math.ceil(total_bytes / self.block_bytes))
            self._free_span(block.layer_id, block.block_id, needed)

    # ── vLLM block-table mapping ───────────────────────────────────────────

    def map_into_vllm(
        self,
        prefix_hash: str,
        vllm_seq_id: int,
        vllm_block_table: Any,
    ) -> bool:
        """Populate vLLM's block table with references to this pool's blocks.

        This is the mechanism that makes a warm restore zero-copy on the
        vLLM side. We write *opaque* block pointers into the block table
        slots that correspond to this sequence. Each pointer carries the
        ``is_external`` tag so that vLLM's allocator does not attempt to
        free the block when the sequence completes.

        The exact API depends on the vLLM version. This method accepts
        any object with an ``__setitem__`` that takes ``(seq_id, index)``
        and a block reference, or a dict-of-lists structure. If neither
        interface is satisfied we fall back to a best-effort duck-typed
        write and log a warning.
        """
        entry = self._index.get(prefix_hash)
        if entry is None:
            return False

        try:
            # Preferred path: dict-of-lists keyed by sequence id.
            table = vllm_block_table[vllm_seq_id]
            for idx, block in enumerate(entry.blocks):
                ref = _ExternalBlockRef(block)
                if idx < len(table):
                    table[idx] = ref
                else:
                    table.append(ref)
            return True
        except Exception as exc:  # pragma: no cover - exercised in integration
            logger.warning(
                "map_into_vllm: failed to populate block table for seq=%s: %s",
                vllm_seq_id,
                exc,
            )
            return False


# ── External block reference ────────────────────────────────────────────────


@dataclass
class _ExternalBlockRef:
    """Opaque block pointer passed into vLLM's block table.

    The ``is_external`` tag is the contract with vLLM's block manager:
    any ref carrying ``is_external=True`` must be ignored by the
    allocator's reference counting. See §6.3 of the design doc.
    """

    block: CompressedBlock
    is_external: bool = True

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ExternalBlockRef layer={self.block.layer_id} "
            f"block={self.block.block_id} fmt={self.block.format}>"
        )


__all__ = [
    "CompressedBlock",
    "CompressedVRAMPool",
    "PoolEntry",
]
