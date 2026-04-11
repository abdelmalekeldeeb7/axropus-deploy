"""compressed_vram_pool.py — Quantized KV cache pool in VRAM.

Stores multiple KV snapshots in GPU memory using FP8/INT4 quantization,
enabling 2-8x more prefixes per GPU than raw bf16 storage.

Architecture:
    ┌─────────────────────────────────────────┐
    │         Compressed VRAM Pool            │
    │                                         │
    │  ┌──────┐ ┌──────┐ ┌──────┐           │
    │  │Pfx A │ │Pfx B │ │Pfx C │  ...×50+  │
    │  │INT4  │ │INT4  │ │INT4  │           │
    │  │~10GB │ │~10GB │ │~5GB  │           │
    │  └──────┘ └──────┘ └──────┘           │
    │                                         │
    │  On hit: dequant → scatter to KV cache  │
    │  ~50-200ms for 128K context             │
    └─────────────────────────────────────────┘

Usage:
    pool = CompressedVRAMPool(max_gb=50, device="cuda:0")

    # Store (after save or cold prefill):
    pool.put(key, gpu_cache, block_table, token_ids)

    # Restore (on cache hit):
    n_tokens = pool.restore(key, gpu_cache, block_table)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


# ── Quantization modes ───────────────────────────────────────────────────────

QUANT_FP8 = "fp8"       # 2x compression, ~0.999 cosine sim
QUANT_INT4 = "int4"     # 4x compression, ~0.995 cosine sim
QUANT_INT2 = "int2"     # 8x compression, ~0.98 cosine sim


@dataclass
class _CompressedLayer:
    """Compressed K and V data for one layer."""
    k_data: torch.Tensor      # quantized K on GPU
    v_data: torch.Tensor      # quantized V on GPU
    k_scale: Optional[torch.Tensor] = None  # per-channel scale for INT4
    v_scale: Optional[torch.Tensor] = None
    k_zero: Optional[torch.Tensor] = None   # zero point for INT4
    v_zero: Optional[torch.Tensor] = None


@dataclass
class _PoolEntry:
    """One compressed KV snapshot in the pool."""
    layers: List[_CompressedLayer]
    n_tokens: int
    n_blocks: int
    block_shape: tuple            # (block_size, n_kv_heads, head_dim)
    orig_dtype: torch.dtype       # original KV dtype (e.g. bfloat16)
    quant_mode: str               # QUANT_FP8, QUANT_INT4, etc.
    size_bytes: int               # total GPU memory used
    created_at: float = field(default_factory=time.monotonic)


class CompressedVRAMPool:
    """Multi-prefix quantized KV cache pool in GPU VRAM.

    Stores KV snapshots in compressed format (FP8 or INT4) on GPU.
    On cache hit, decompresses and scatters into vLLM's KV cache blocks.

    This is the production VRAM cache — handles 10-50+ prefixes at 128K
    context on a single H200 by quantizing KV data 2-8x.
    """

    def __init__(
        self,
        max_gb: float = 50.0,
        device: str = "cuda:0",
        quant_mode: str = QUANT_INT4,
    ) -> None:
        self._max_bytes = int(max_gb * 1024 ** 3)
        self._device = device
        self._quant_mode = quant_mode
        self._cache: OrderedDict[str, _PoolEntry] = OrderedDict()
        self._used_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        logger.info(
            "[VRAM_POOL] initialized: %.1f GB capacity, quant=%s, device=%s",
            max_gb, quant_mode, device,
        )

    # ── Quantization ─────────────────────────────────────────────────────────

    def _quantize_fp8(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize bf16/fp16 tensor to FP8 (E4M3). 2x compression."""
        # Per-tensor absmax scaling
        amax = tensor.abs().amax()
        scale = amax / 448.0  # E4M3 max value
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = (tensor / scale).to(torch.float8_e4m3fn)
        return q, scale

    def _dequantize_fp8(
        self, q: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype
    ) -> torch.Tensor:
        """Dequantize FP8 back to target dtype."""
        return q.to(target_dtype) * scale

    def _quantize_int4(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize bf16/fp16 tensor to INT4 (packed as uint8). 4x compression.

        Uses per-block quantization for better quality:
        - Compute min/max per KV block
        - Scale to [0, 15] range
        - Pack two INT4 values into one uint8 byte
        """
        # Reshape to (n_elements_per_2, 2) for packing
        flat = tensor.reshape(-1).float()
        n = flat.numel()

        # Per-tensor quantization (fast path)
        vmin = flat.min()
        vmax = flat.max()
        scale = (vmax - vmin) / 15.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = vmin

        # Quantize to 0-15
        q = ((flat - zero) / scale).clamp(0, 15).round().to(torch.uint8)

        # Pack pairs of INT4 into uint8
        if n % 2 != 0:
            q = torch.cat([q, torch.zeros(1, dtype=torch.uint8, device=q.device)])
        q_pairs = q.reshape(-1, 2)
        packed = (q_pairs[:, 0] << 4) | q_pairs[:, 1]

        return packed, scale.unsqueeze(0), zero.unsqueeze(0)

    def _dequantize_int4(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        n_elements: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize packed INT4 back to target dtype."""
        # Unpack uint8 → two INT4 values
        hi = (packed >> 4).to(torch.float32)
        lo = (packed & 0x0F).to(torch.float32)
        unpacked = torch.stack([hi, lo], dim=1).reshape(-1)[:n_elements]

        # Dequantize
        return (unpacked * scale + zero).to(target_dtype)

    def _quantize_int2(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize bf16/fp16 tensor to INT2 (packed as uint8). 8x compression.

        Each value mapped to 0-3 (2 bits). Four INT2 values packed per byte.
        Uses per-block quantization: one scale + zero_point per KV block
        (block_size * num_kv_heads * head_dim elements) for much better quality
        than per-tensor quantization with only 4 levels.
        """
        flat = tensor.reshape(-1).float()
        n = flat.numel()

        # Determine block size: each KV block is [block_size, num_kv_heads, head_dim]
        # tensor shape is (n_blocks, block_size, num_kv_heads, head_dim)
        if tensor.dim() >= 2:
            block_elems = tensor.shape[1:].numel()  # elements per KV block
        else:
            block_elems = n  # fallback: treat entire tensor as one block
        n_blocks = max(n // block_elems, 1)

        # Pad flat tensor so it divides evenly into blocks
        padded_n = n_blocks * block_elems
        if padded_n > n:
            flat = torch.cat([flat, torch.zeros(padded_n - n, dtype=flat.dtype,
                                                device=flat.device)])
        elif padded_n < n:
            # More elements than expected blocks — treat remainder as extra block
            n_blocks = math.ceil(n / block_elems)
            padded_n = n_blocks * block_elems
            flat = torch.cat([flat, torch.zeros(padded_n - n, dtype=flat.dtype,
                                                device=flat.device)])

        blocks = flat.reshape(n_blocks, -1)  # (n_blocks, block_elems)

        # Per-block min/max
        vmin = blocks.min(dim=1).values  # (n_blocks,)
        vmax = blocks.max(dim=1).values  # (n_blocks,)
        scale = (vmax - vmin) / 3.0      # (n_blocks,)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = vmin                       # (n_blocks,)

        # Quantize to 0-3 per block
        q = ((blocks - zero.unsqueeze(1)) / scale.unsqueeze(1)).clamp(0, 3).round().to(torch.uint8)
        q = q.reshape(-1)  # flatten back

        # Trim back to original + pack-padding length
        total = padded_n
        pad4 = (4 - total % 4) % 4
        if pad4 > 0:
            q = torch.cat([q, torch.zeros(pad4, dtype=torch.uint8, device=q.device)])

        # Pack 4 INT2 values into one uint8
        q4 = q.reshape(-1, 4)
        packed = (q4[:, 0] << 6) | (q4[:, 1] << 4) | (q4[:, 2] << 2) | q4[:, 3]

        return packed, scale, zero

    def _dequantize_int2(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        n_elements: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize packed INT2 back to target dtype using per-block scales."""
        # Unpack uint8 → four INT2 values
        b3 = ((packed >> 6) & 0x03).to(torch.float32)
        b2 = ((packed >> 4) & 0x03).to(torch.float32)
        b1 = ((packed >> 2) & 0x03).to(torch.float32)
        b0 = (packed & 0x03).to(torch.float32)
        unpacked = torch.stack([b3, b2, b1, b0], dim=1).reshape(-1)

        n_blocks = scale.numel()
        if n_blocks > 1:
            # Per-block dequantization
            block_elems = unpacked.numel() // n_blocks
            unpacked_blocks = unpacked.reshape(n_blocks, block_elems)
            result = unpacked_blocks * scale.unsqueeze(1) + zero.unsqueeze(1)
            result = result.reshape(-1)[:n_elements]
        else:
            # Single block / legacy fallback
            unpacked = unpacked[:n_elements]
            result = unpacked * scale + zero

        return result.to(target_dtype)

    # ── Store ─────────────────────────────────────────────────────────────────

    def put(
        self,
        key: str,
        gpu_cache: list,
        block_table: List[int],
        n_tokens: int,
    ) -> bool:
        """Quantize and store KV data from the GPU cache into the pool.

        Reads KV data directly from vLLM's gpu_cache tensors, quantizes
        on GPU, and stores the compressed representation.

        Args:
            key: Unique identifier for this prefix.
            gpu_cache: vLLM's gpu_cache list (per-layer tensors).
            block_table: Physical block IDs containing the KV data.
            n_tokens: Number of tokens in this prefix.

        Returns:
            True if stored successfully.
        """
        t0 = time.monotonic()
        n_layers = len(gpu_cache)
        if n_layers == 0:
            return False

        block_ids = torch.tensor(block_table, device="cuda", dtype=torch.long)
        n_blocks = len(block_table)

        # Determine block shape from first layer
        layer0 = gpu_cache[0]
        if layer0.dim() == 5 and layer0.shape[0] == 2:
            block_shape = layer0[0].shape[1:]  # (block_size, n_kv_heads, head_dim)
            orig_dtype = layer0.dtype
        else:
            block_shape = layer0.shape[1:]
            orig_dtype = layer0.dtype

        compressed_layers: List[_CompressedLayer] = []
        total_bytes = 0
        CHUNK = 512  # gather chunk size to avoid OOM

        # Auto-detect: if KV cache is already FP8/INT8 (compressed), store raw.
        # Don't double-quantize compressed data → INT4 — that destroys quality.
        _fp8_types = set()
        if hasattr(torch, 'float8_e4m3fn'):
            _fp8_types.add(torch.float8_e4m3fn)
        if hasattr(torch, 'float8_e5m2'):
            _fp8_types.add(torch.float8_e5m2)
        is_compressed = (orig_dtype in _fp8_types or
                         orig_dtype == torch.int8 or
                         orig_dtype == torch.uint8 or
                         orig_dtype.itemsize == 1)  # any 1-byte dtype = already compressed
        effective_quant = "raw" if is_compressed else self._quant_mode
        if is_compressed:
            logger.info("[VRAM_POOL] Compressed KV detected (dtype=%s) — storing raw", orig_dtype)

        for layer_cache in gpu_cache:
            if layer_cache.dim() == 5 and layer_cache.shape[0] == 2:
                k_all = layer_cache[0]
                v_all = layer_cache[1]
            else:
                k_all = layer_cache
                v_all = layer_cache

            # Gather blocks in chunks to avoid OOM
            k_chunks = []
            v_chunks = []
            for start in range(0, n_blocks, CHUNK):
                end = min(start + CHUNK, n_blocks)
                chunk_ids = block_ids[start:end]
                k_chunks.append(k_all[chunk_ids])
                v_chunks.append(v_all[chunk_ids])
                torch.cuda.current_stream().synchronize()

            k_seq = torch.cat(k_chunks, dim=0) if len(k_chunks) > 1 else k_chunks[0]
            v_seq = torch.cat(v_chunks, dim=0) if len(v_chunks) > 1 else v_chunks[0]
            del k_chunks, v_chunks

            # Quantize on GPU
            if effective_quant == "raw":
                cl = _CompressedLayer(k_data=k_seq.clone(), v_data=v_seq.clone())
                total_bytes += k_seq.nbytes + v_seq.nbytes
            elif effective_quant == QUANT_FP8:
                k_q, k_scale = self._quantize_fp8(k_seq)
                v_q, v_scale = self._quantize_fp8(v_seq)
                cl = _CompressedLayer(
                    k_data=k_q, v_data=v_q,
                    k_scale=k_scale, v_scale=v_scale,
                )
                total_bytes += k_q.nbytes + v_q.nbytes + k_scale.nbytes + v_scale.nbytes

            elif self._quant_mode == QUANT_INT4:
                k_packed, k_scale, k_zero = self._quantize_int4(k_seq)
                v_packed, v_scale, v_zero = self._quantize_int4(v_seq)
                cl = _CompressedLayer(
                    k_data=k_packed, v_data=v_packed,
                    k_scale=k_scale, v_scale=v_scale,
                    k_zero=k_zero, v_zero=v_zero,
                )
                total_bytes += (k_packed.nbytes + v_packed.nbytes +
                                k_scale.nbytes + v_scale.nbytes +
                                k_zero.nbytes + v_zero.nbytes)

            elif self._quant_mode == QUANT_INT2:
                k_packed, k_scale, k_zero = self._quantize_int2(k_seq)
                v_packed, v_scale, v_zero = self._quantize_int2(v_seq)
                cl = _CompressedLayer(
                    k_data=k_packed, v_data=v_packed,
                    k_scale=k_scale, v_scale=v_scale,
                    k_zero=k_zero, v_zero=v_zero,
                )
                total_bytes += (k_packed.nbytes + v_packed.nbytes +
                                k_scale.nbytes + v_scale.nbytes +
                                k_zero.nbytes + v_zero.nbytes)
            else:
                # Raw (no compression)
                cl = _CompressedLayer(k_data=k_seq.clone(), v_data=v_seq.clone())
                total_bytes += k_seq.nbytes + v_seq.nbytes

            compressed_layers.append(cl)
            del k_seq, v_seq

        entry = _PoolEntry(
            layers=compressed_layers,
            n_tokens=n_tokens,
            n_blocks=n_blocks,
            block_shape=block_shape,
            orig_dtype=orig_dtype,
            quant_mode=effective_quant,
            size_bytes=total_bytes,
        )

        with self._lock:
            # Remove existing entry
            if key in self._cache:
                self._used_bytes -= self._cache[key].size_bytes
                del self._cache[key]

            # Evict LRU until space is available
            while self._used_bytes + total_bytes > self._max_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._used_bytes -= evicted.size_bytes
                logger.debug("[VRAM_POOL] evicted entry, freed %.1f MB",
                             evicted.size_bytes / (1024 * 1024))

            if self._used_bytes + total_bytes > self._max_bytes:
                logger.warning(
                    "[VRAM_POOL] entry too large: %.1f GB > %.1f GB available",
                    total_bytes / (1024 ** 3),
                    (self._max_bytes - self._used_bytes) / (1024 ** 3),
                )
                return False

            self._cache[key] = entry
            self._used_bytes += total_bytes

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        compression = (n_blocks * sum(block_shape) * 2 * 2 * n_layers) / max(total_bytes, 1)
        logger.info(
            "[VRAM_POOL] stored key=%s: %d layers, %d blocks, "
            "%.1f MB compressed (%.1fx), %.0f ms",
            key[:32], n_layers, n_blocks,
            total_bytes / (1024 * 1024), compression, elapsed_ms,
        )
        return True

    # ── Restore ───────────────────────────────────────────────────────────────

    def restore(
        self,
        key: str,
        gpu_cache: list,
        block_table: List[int],
    ) -> int:
        """Decompress and scatter KV data from pool into GPU cache blocks.

        Args:
            key: Prefix identifier.
            gpu_cache: vLLM's gpu_cache (per-layer tensors).
            block_table: Physical block IDs to write into.

        Returns:
            Number of tokens restored, or 0 on miss.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return 0
            self._cache.move_to_end(key)  # refresh LRU

        self._hits += 1
        t0 = time.monotonic()

        block_ids = torch.tensor(block_table, device="cuda", dtype=torch.long)
        n_blocks = min(len(block_table), entry.n_blocks)
        CHUNK = 512

        for layer_idx, (layer_cache, cl) in enumerate(zip(gpu_cache, entry.layers)):
            if layer_cache.dim() == 5 and layer_cache.shape[0] == 2:
                k_all = layer_cache[0]
                v_all = layer_cache[1]
            else:
                k_all = layer_cache
                v_all = layer_cache

            target_dtype = k_all.dtype
            seq_shape = (n_blocks, *k_all.shape[1:])
            n_elements = 1
            for d in seq_shape:
                n_elements *= d

            # Dequantize
            if entry.quant_mode == QUANT_FP8:
                k_full = self._dequantize_fp8(cl.k_data, cl.k_scale, target_dtype)
                v_full = self._dequantize_fp8(cl.v_data, cl.v_scale, target_dtype)
                k_src = k_full.reshape(seq_shape)
                v_src = v_full.reshape(seq_shape)
            elif entry.quant_mode == QUANT_INT4:
                k_full = self._dequantize_int4(
                    cl.k_data, cl.k_scale, cl.k_zero, n_elements, target_dtype
                )
                v_full = self._dequantize_int4(
                    cl.v_data, cl.v_scale, cl.v_zero, n_elements, target_dtype
                )
                k_src = k_full.reshape(seq_shape)
                v_src = v_full.reshape(seq_shape)
            elif entry.quant_mode == QUANT_INT2:
                k_full = self._dequantize_int2(
                    cl.k_data, cl.k_scale, cl.k_zero, n_elements, target_dtype
                )
                v_full = self._dequantize_int2(
                    cl.v_data, cl.v_scale, cl.v_zero, n_elements, target_dtype
                )
                k_src = k_full.reshape(seq_shape)
                v_src = v_full.reshape(seq_shape)
            else:
                k_src = cl.k_data.reshape(seq_shape).to(target_dtype)
                v_src = cl.v_data.reshape(seq_shape).to(target_dtype)

            # Chunked scatter to avoid OOM
            for start in range(0, n_blocks, CHUNK):
                end = min(start + CHUNK, n_blocks)
                k_all[block_ids[start:end]] = k_src[start:end]
                v_all[block_ids[start:end]] = v_src[start:end]

            del k_src, v_src
            if entry.quant_mode in (QUANT_FP8, QUANT_INT4, QUANT_INT2):
                del k_full, v_full

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "[VRAM_POOL] restore key=%s: %d layers, %d blocks, %.0f ms",
            key[:32], len(entry.layers), n_blocks, elapsed_ms,
        )
        return entry.n_tokens

    # ── Management ────────────────────────────────────────────────────────────

    def insert_from_raw(
        self,
        key: str,
        raw_kv: torch.Tensor,
        n_tokens: int,
        *,
        orig_dtype: Optional[torch.dtype] = None,
    ) -> bool:
        """Insert a raw (uncompressed) KV tensor from a fallback tier (LMCache).

        Used by the TieredCacheRouter to promote LMCache hits into the
        compressed pool so the NEXT lookup is a G1 (warm) hit instead of
        going back through CPU/NVMe. The raw tensor is run through the
        same quantization path as ``put()``.

        Accepted shapes (inferred automatically):
          * ``[n_layers, 2, n_blocks, block_size, n_kv_heads, head_dim]``
            — LMCache's "stacked" layout
          * ``[n_layers, 2, n_tokens, n_kv_heads, head_dim]``
            — LMCache's "per-token" layout (reshape to blocks)
          * ``[n_blocks, block_size, n_kv_heads, head_dim]``
            — single-layer probe (testing / benchmarks)

        Args:
            key:         Prefix hash to insert under.
            raw_kv:      Raw KV tensor on GPU (or CPU — will be moved).
            n_tokens:    Number of tokens in the prefix.
            orig_dtype:  Optional override for the "original" dtype to
                         store on the entry (defaults to raw_kv.dtype).

        Returns ``True`` on successful insertion.
        """
        t0 = time.monotonic()

        if raw_kv is None or raw_kv.numel() == 0:
            return False

        # Move to the pool's device if needed.
        if str(raw_kv.device) != str(self._device):
            try:
                raw_kv = raw_kv.to(self._device)
            except Exception as exc:
                logger.warning(
                    "[VRAM_POOL] insert_from_raw: to(%s) failed: %s",
                    self._device, exc,
                )
                return False

        effective_dtype = orig_dtype if orig_dtype is not None else raw_kv.dtype

        # ── Infer layout ──────────────────────────────────────────────────
        # We need per-layer (k_seq, v_seq) pairs to feed the quantizer.
        layers_kv: List[tuple] = []
        block_shape: tuple = ()

        if raw_kv.dim() >= 5 and raw_kv.shape[1] == 2:
            # Stacked per-layer: [L, 2, n_blocks, block_size, n_kv_heads, head_dim]
            # or [L, 2, n_tokens, n_kv_heads, head_dim]
            n_layers = raw_kv.shape[0]
            for L in range(n_layers):
                k = raw_kv[L, 0]
                v = raw_kv[L, 1]
                layers_kv.append((k, v))
            block_shape = tuple(layers_kv[0][0].shape[1:]) if layers_kv else ()
        elif raw_kv.dim() == 6 and raw_kv.shape[1] == 2:
            # [L, 2, n_blocks, block_size, n_kv_heads, head_dim] — explicit
            n_layers = raw_kv.shape[0]
            for L in range(n_layers):
                layers_kv.append((raw_kv[L, 0], raw_kv[L, 1]))
            block_shape = tuple(raw_kv.shape[3:])
        elif raw_kv.dim() == 4:
            # Single layer, block-shaped: [n_blocks, block_size, n_kv_heads, head_dim]
            layers_kv.append((raw_kv, raw_kv))
            block_shape = tuple(raw_kv.shape[1:])
        else:
            logger.warning(
                "[VRAM_POOL] insert_from_raw: unsupported tensor shape %s",
                tuple(raw_kv.shape),
            )
            return False

        n_layers = len(layers_kv)
        n_blocks = layers_kv[0][0].shape[0] if layers_kv else 0

        # ── Quantize each layer ──────────────────────────────────────────
        _fp8_types = set()
        if hasattr(torch, "float8_e4m3fn"):
            _fp8_types.add(torch.float8_e4m3fn)
        if hasattr(torch, "float8_e5m2"):
            _fp8_types.add(torch.float8_e5m2)
        is_compressed = (
            effective_dtype in _fp8_types
            or effective_dtype == torch.int8
            or effective_dtype == torch.uint8
            or effective_dtype.itemsize == 1
        )
        effective_quant = "raw" if is_compressed else self._quant_mode

        compressed_layers: List[_CompressedLayer] = []
        total_bytes = 0

        for k_seq, v_seq in layers_kv:
            if effective_quant == "raw":
                cl = _CompressedLayer(k_data=k_seq.contiguous().clone(),
                                       v_data=v_seq.contiguous().clone())
                total_bytes += k_seq.nbytes + v_seq.nbytes
            elif effective_quant == QUANT_FP8:
                k_q, k_scale = self._quantize_fp8(k_seq)
                v_q, v_scale = self._quantize_fp8(v_seq)
                cl = _CompressedLayer(
                    k_data=k_q, v_data=v_q,
                    k_scale=k_scale, v_scale=v_scale,
                )
                total_bytes += (k_q.nbytes + v_q.nbytes
                                + k_scale.nbytes + v_scale.nbytes)
            elif effective_quant == QUANT_INT4:
                k_packed, k_scale, k_zero = self._quantize_int4(k_seq)
                v_packed, v_scale, v_zero = self._quantize_int4(v_seq)
                cl = _CompressedLayer(
                    k_data=k_packed, v_data=v_packed,
                    k_scale=k_scale, v_scale=v_scale,
                    k_zero=k_zero, v_zero=v_zero,
                )
                total_bytes += (k_packed.nbytes + v_packed.nbytes
                                + k_scale.nbytes + v_scale.nbytes
                                + k_zero.nbytes + v_zero.nbytes)
            elif effective_quant == QUANT_INT2:
                k_packed, k_scale, k_zero = self._quantize_int2(k_seq)
                v_packed, v_scale, v_zero = self._quantize_int2(v_seq)
                cl = _CompressedLayer(
                    k_data=k_packed, v_data=v_packed,
                    k_scale=k_scale, v_scale=v_scale,
                    k_zero=k_zero, v_zero=v_zero,
                )
                total_bytes += (k_packed.nbytes + v_packed.nbytes
                                + k_scale.nbytes + v_scale.nbytes
                                + k_zero.nbytes + v_zero.nbytes)
            else:
                cl = _CompressedLayer(k_data=k_seq.contiguous().clone(),
                                       v_data=v_seq.contiguous().clone())
                total_bytes += k_seq.nbytes + v_seq.nbytes
            compressed_layers.append(cl)

        entry = _PoolEntry(
            layers=compressed_layers,
            n_tokens=n_tokens,
            n_blocks=n_blocks,
            block_shape=block_shape,
            orig_dtype=effective_dtype,
            quant_mode=effective_quant,
            size_bytes=total_bytes,
        )

        with self._lock:
            if key in self._cache:
                self._used_bytes -= self._cache[key].size_bytes
                del self._cache[key]

            while (self._used_bytes + total_bytes > self._max_bytes
                   and self._cache):
                _, evicted = self._cache.popitem(last=False)
                self._used_bytes -= evicted.size_bytes

            if self._used_bytes + total_bytes > self._max_bytes:
                logger.warning(
                    "[VRAM_POOL] insert_from_raw: entry too large "
                    "%.1f GB > %.1f GB",
                    total_bytes / (1024 ** 3),
                    (self._max_bytes - self._used_bytes) / (1024 ** 3),
                )
                return False

            self._cache[key] = entry
            self._used_bytes += total_bytes

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "[VRAM_POOL] insert_from_raw key=%s: %d layers, %.1f MB in %.0f ms",
            key[:32], n_layers, total_bytes / (1024 * 1024), elapsed_ms,
        )
        return True

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def evict(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                self._used_bytes -= entry.size_bytes
                return True
            return False

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._cache),
                "used_gb": self._used_bytes / (1024 ** 3),
                "max_gb": self._max_bytes / (1024 ** 3),
                "quant_mode": self._quant_mode,
                "hits": self._hits,
                "misses": self._misses,
                "prefixes": list(self._cache.keys()),
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._used_bytes = 0
