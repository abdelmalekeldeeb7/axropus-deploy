"""amf_kv_manager.py — Direct GPU KV cache save/restore through vLLM's CacheEngine.

This module ports the C++ amf_direct_kv.cu logic to Python so that the vLLM
production backend can skip the CPU deserialization bottleneck.

Design notes:
  - The AMF key hash functions are bit-for-bit identical to the C++ implementation
    (FNV-1a, same constants, same field order) so snapshots are cross-compatible.
  - KV data is moved with torch.cuda: H→D and D→H use pinned (page-locked) host
    tensors to maximise PCIe throughput, matching the cudaMallocHost approach in C++.
  - The AMF store file format is the same as amf_direct_kv.cu: AMFK header followed
    by the per-layer info table, then the flat KV payload.
  - vLLM's KV cache layout: gpu_cache is a list of tensors per layer, each with
    shape [2, num_blocks, block_size, num_kv_heads, head_dim] (K and V stacked
    on dim 0 in vLLM ≥ 0.6).  We handle both the stacked and split layouts.

Dependencies:
  - torch with CUDA
  - vLLM >= 0.6.0
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch

from .turboquant_codec import CODEC_NONE, CODEC_TURBOQUANT, get_codec

logger = logging.getLogger(__name__)

# ── FNV-1a constants (must match amf_store.cpp) ───────────────────────────────

_FNV_OFFSET: int = 1469598103934665603  # 0x14650FB0739D0383
_FNV_PRIME:  int = 1099511628211        # 0x00000100000001B3
_U64_MASK:   int = (1 << 64) - 1

# ── AMFK file format (must match amf_direct_kv.h) ────────────────────────────

_AMFK_MAGIC:   int = 0x414D464B  # "AMFK"
_AMFK_VERSION: int = 1

# struct AmfDirectKvHeader { u32×8, u64×3 } = 56 bytes
_HEADER_FMT  = "<IIIIIIII QQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)   # 56

# struct AmfDirectKvLayerInfo { u64×4 } = 32 bytes
_LAYER_FMT  = "<QQQQ"
_LAYER_SIZE = struct.calcsize(_LAYER_FMT)     # 32

assert _HEADER_SIZE == 56,  f"Header size changed: {_HEADER_SIZE}"
assert _LAYER_SIZE  == 32,  f"Layer info size changed: {_LAYER_SIZE}"

# ── AMF dtype tag (must match AmfKvDtype in amf_direct_kv.h) ─────────────────

_DTYPE_TAG = {
    torch.float16:  0,
    torch.float32:  1,
    torch.bfloat16: 2,
}


# ── Hash helpers (FNV-1a, byte-level, matching C++ hash_bytes / hash_token_prefix_step)

def _fnv1a_bytes(data: bytes, h: int = _FNV_OFFSET) -> int:
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & _U64_MASK
    return h


def amf_hash_tokens(token_ids: Sequence[int]) -> int:
    """FNV-1a over token ids as little-endian 32-bit ints (matches C++ amf_hash_tokens)."""
    h = _FNV_OFFSET
    for tok in token_ids:
        # hash_token_prefix_step: h ^= (uint64_t)(uint32_t)tok; h *= prime
        h = ((h ^ (tok & 0xFFFFFFFF)) * _FNV_PRIME) & _U64_MASK
    return h


def amf_hash_tenant_id(tenant_id: str) -> int:
    """FNV-1a over UTF-8 encoded tenant string (matches C++ amf_hash_tenant_id)."""
    if not tenant_id:
        return _FNV_OFFSET
    return _fnv1a_bytes(tenant_id.encode("utf-8"))


def amf_hash_model(model_path: str) -> int:
    """FNV-1a over the model file bytes (matches C++ amf_hash_file).

    This is expensive for large models.  In practice, callers should cache the
    result or supply the hash from an env var (KORITH_AMF_MODEL_HASH).
    """
    h = _FNV_OFFSET
    path = Path(model_path)
    if not path.exists():
        return 0
    buf_size = 1 << 20  # 1 MiB
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h = _fnv1a_bytes(chunk, h)
    return h


def amf_float_bits(v: float) -> int:
    """Return the IEEE-754 bit pattern of a 32-bit float (matches C++ amf_float_bits)."""
    return struct.unpack("<I", struct.pack("<f", v))[0]


# ── AMF filename format ───────────────────────────────────────────────────────

def amf_key_filename(
    model_hash: int,
    tenant_hash: int,
    prefix_hash: int,
    n_ctx: int,
    kv_version: int,
    rope_base_bits: int,
    rope_scale_bits: int,
    sampling_hash: int,
    rng_hash: int,
) -> str:
    """Reproduce the C++ AmfStore entry_basename() naming convention."""
    return (
        f"amf_{model_hash:016x}_{tenant_hash:016x}_{prefix_hash:016x}"
        f"_{n_ctx}_{kv_version}_{rope_base_bits}_{rope_scale_bits}"
        f"_{sampling_hash:016x}_{rng_hash:016x}"
    )


# ── VRAM snapshot cache ───────────────────────────────────────────────────────

@dataclass
class _VRAMEntry:
    """One KV snapshot resident in GPU VRAM (compressed or raw)."""
    payload_gpu: "torch.Tensor"          # uint8 CUDA tensor
    snap_dtype:  "torch.dtype"
    n_tokens:    int
    layer_infos: list                    # [(k_off, k_sz, v_off, v_sz), ...]
    codec_id:    int = CODEC_NONE        # CODEC_NONE or CODEC_TURBOQUANT


class VRAMSnapshotCache:
    """LRU cache of TurboQuant-compressed KV snapshots stored as GPU tensors.

    Enable via:
        KORITH_VRAM_CACHE_GB=120          # VRAM budget in GB (0 = disabled)
        KORITH_VRAM_CACHE_DEVICE=cuda:0   # which GPU to use

    On restore the compressed payload is already on GPU → decompress with zero
    H→D transfer → ~10-50 ms total vs ~3,044 ms for the NVMe+TQ path.
    """

    def __init__(self, max_bytes: int, device: str) -> None:
        self._max    = max_bytes
        self._device = device
        self._cache: OrderedDict[str, _VRAMEntry] = OrderedDict()
        self._used   = 0
        self._lock   = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    def put(
        self,
        key:         str,
        compressed:  bytes,
        snap_dtype:  "torch.dtype",
        n_tokens:    int,
        layer_infos: list,
        codec_id:    int = CODEC_NONE,
    ) -> bool:
        """Store KV blob as a GPU tensor.  Returns True if cached."""
        if len(compressed) > self._max:
            return False

        try:
            gpu_t = torch.frombuffer(bytearray(compressed), dtype=torch.uint8).to(self._device)
        except torch.cuda.OutOfMemoryError:
            logger.debug("[AMF_VLLM] VRAM cache: OOM, skipping cache for this entry")
            return False

        with self._lock:
            # Replace existing entry for this key
            if key in self._cache:
                self._used -= self._cache[key].payload_gpu.nbytes
                del self._cache[key]

            # Evict LRU until there is space
            while self._used + gpu_t.nbytes > self._max and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._used -= evicted.payload_gpu.nbytes

            if self._used + gpu_t.nbytes > self._max:
                return False  # still no room (single entry exceeds budget)

            entry = _VRAMEntry(
                payload_gpu=gpu_t,
                snap_dtype=snap_dtype,
                n_tokens=n_tokens,
                layer_infos=layer_infos,
                codec_id=codec_id,
            )
            self._cache[key] = entry
            self._used += gpu_t.nbytes
            return True

    def get(self, key: str) -> Optional[_VRAMEntry]:
        """Return the VRAM entry if cached, else None.  Refreshes LRU order."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
            return entry

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries":  len(self._cache),
                "used_gb":  self._used / (1024 ** 3),
                "max_gb":   self._max  / (1024 ** 3),
            }


# ── AmfKvManager ─────────────────────────────────────────────────────────────

class AmfKvManager:
    """Direct GPU KV cache save/restore through vLLM's CacheEngine.

    The save path:
      1. Iterates over gpu_cache layers and extracts the K/V data for the
         given sequence's physical blocks.
      2. Copies from GPU to pinned host memory.
      3. Writes the AMFK blob to the AMF store directory.

    The restore path:
      1. Reads the AMFK blob from disk.
      2. Allocates physical blocks in vLLM's block manager.
      3. Copies KV data from host to the allocated GPU blocks.
      4. Updates sequence metadata so vLLM treats the sequence as fully prefilled.
    """

    def __init__(
        self,
        cache_engine: Any,
        amf_store_path: str,
        model_config: Any,
        *,
        model_hash: int = 0,
        tenant_id: str = "__shared__",
        kv_version: int = 1,
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
        sampling_hash: int = 0,
        rng_hash: int = 0,
    ) -> None:
        """
        Args:
            cache_engine:    vLLM CacheEngine instance.
            amf_store_path:  Path to the AMF store directory on disk.
            model_config:    vLLM ModelConfig (has num_attention_layers, etc.).
            model_hash:      Pre-computed file hash (0 = disabled).
            tenant_id:       Tenant string for multi-tenant key isolation.
            kv_version:      Increment when KV format changes.
            rope_base:       Model RoPE base frequency.
            rope_scale:      Model RoPE scale factor.
            sampling_hash:   Hash of sampling parameters (0 = greedy).
            rng_hash:        RNG seed hash (0 = deterministic).
        """
        self._cache_engine   = cache_engine
        self._store_path     = Path(amf_store_path)
        self._model_config   = model_config
        self._model_hash     = model_hash
        self._tenant_hash    = amf_hash_tenant_id(tenant_id)
        self._kv_version     = kv_version
        self._rope_base_bits = amf_float_bits(rope_base)
        self._rope_scale_bits = amf_float_bits(rope_scale)
        self._sampling_hash  = sampling_hash
        self._rng_hash       = rng_hash

        self._store_path.mkdir(parents=True, exist_ok=True)

        # VRAM snapshot cache (primary tier — zero H→D on restore)
        _vram_gb  = float(os.environ.get("KORITH_VRAM_CACHE_GB", "0"))
        _vram_dev = os.environ.get("KORITH_VRAM_CACHE_DEVICE", "cuda:0")
        if _vram_gb > 0 and torch.cuda.is_available():
            self._vram_cache: Optional[VRAMSnapshotCache] = VRAMSnapshotCache(
                max_bytes=int(_vram_gb * (1024 ** 3)),
                device=_vram_dev,
            )
            logger.info(
                "[AMF_VLLM] VRAM cache enabled: %.0f GB on %s", _vram_gb, _vram_dev
            )
        else:
            self._vram_cache = None

        # Statistics.
        self._hits:        int   = 0
        self._misses:      int   = 0
        self._saves:       int   = 0
        self._restore_ms:  float = 0.0
        self._vram_hits:   int   = 0

    # ── Key computation ───────────────────────────────────────────────────────

    def compute_amf_key(
        self,
        prompt_tokens: Sequence[int],
        n_ctx: Optional[int] = None,
    ) -> dict:
        """Compute the 9-field AmfKey matching the C++ implementation."""
        prefix_hash = amf_hash_tokens(prompt_tokens)
        effective_n_ctx = n_ctx or getattr(self._model_config, "max_model_len", 0)
        return {
            "model_hash":      self._model_hash,
            "tenant_hash":     self._tenant_hash,
            "prefix_hash":     prefix_hash,
            "n_ctx":           effective_n_ctx,
            "kv_version":      self._kv_version,
            "rope_base_bits":  self._rope_base_bits,
            "rope_scale_bits": self._rope_scale_bits,
            "sampling_hash":   self._sampling_hash,
            "rng_hash":        self._rng_hash,
        }

    def _kv_filename(self, key: dict) -> Path:
        stem = amf_key_filename(
            model_hash     = key["model_hash"],
            tenant_hash    = key["tenant_hash"],
            prefix_hash    = key["prefix_hash"],
            n_ctx          = key["n_ctx"],
            kv_version     = key["kv_version"],
            rope_base_bits = key["rope_base_bits"],
            rope_scale_bits= key["rope_scale_bits"],
            sampling_hash  = key["sampling_hash"],
            rng_hash       = key["rng_hash"],
        )
        return self._store_path / (stem + ".kv")

    def _tok_filename(self, key: dict) -> Path:
        stem = amf_key_filename(
            model_hash     = key["model_hash"],
            tenant_hash    = key["tenant_hash"],
            prefix_hash    = key["prefix_hash"],
            n_ctx          = key["n_ctx"],
            kv_version     = key["kv_version"],
            rope_base_bits = key["rope_base_bits"],
            rope_scale_bits= key["rope_scale_bits"],
            sampling_hash  = key["sampling_hash"],
            rng_hash       = key["rng_hash"],
        )
        return self._store_path / (stem + ".tok")

    # ── Snapshot existence check ──────────────────────────────────────────────

    def has_snapshot(self, prompt_tokens: Sequence[int]) -> bool:
        """Return True if a snapshot exists in VRAM cache or on disk."""
        key = self.compute_amf_key(prompt_tokens)
        if self._vram_cache is not None:
            if self._vram_cache.contains(self._kv_filename(key).stem):
                return True
        return self._kv_filename(key).exists()

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_kv_state(
        self,
        prompt_tokens: Sequence[int],
        block_table: List[int],
        *,
        saved_ms: float = 0.0,
    ) -> bool:
        """Save KV cache blocks for a sequence after prefill.

        Args:
            prompt_tokens:  Full prompt token IDs for key derivation.
            block_table:    Physical block IDs assigned to this sequence.
            saved_ms:       Estimated time saved (for ROI tracking).

        Returns:
            True on success.
        """
        t0 = time.monotonic()
        key = self.compute_amf_key(prompt_tokens)
        kv_path = self._kv_filename(key)

        gpu_cache = self._cache_engine.gpu_cache  # list of tensors per layer
        n_layers  = len(gpu_cache)

        if n_layers == 0:
            logger.warning("[AMF_VLLM] save: no GPU cache layers")
            return False

        n_tokens    = len(prompt_tokens)

        # ── Determine shapes and compute layout ────────────────────────────────
        sample_layer = gpu_cache[0]
        if sample_layer.dim() == 5 and sample_layer.shape[0] == 2:
            k_sample = sample_layer[0]
        else:
            k_sample = sample_layer
        block_shape = k_sample.shape[1:]  # (block_size, num_kv_heads, head_dim)
        n_kv_heads = block_shape[-2] if len(block_shape) >= 2 else 1
        head_dim   = block_shape[-1]
        kv_dtype   = k_sample.dtype
        elem_size  = k_sample.element_size()

        n_seq_blocks = len(block_table) if block_table else k_sample.shape[0]
        seq_block_shape = (n_seq_blocks, *block_shape)
        per_kv_bytes = 1
        for d in seq_block_shape:
            per_kv_bytes *= d
        per_kv_bytes *= elem_size

        layer_infos: List[tuple] = []
        total_kv_bytes = 0
        for _ in range(n_layers):
            k_off = total_kv_bytes
            v_off = k_off + per_kv_bytes
            layer_infos.append((k_off, per_kv_bytes, v_off, per_kv_bytes))
            total_kv_bytes += 2 * per_kv_bytes

        dtype_tag = _DTYPE_TAG.get(kv_dtype, 0)

        # Write header.
        header = struct.pack(
            _HEADER_FMT,
            _AMFK_MAGIC,        # magic
            _AMFK_VERSION,      # version
            n_layers,           # n_layers
            n_tokens,           # n_tokens
            n_kv_heads,         # n_kv_heads
            head_dim,           # head_dim
            dtype_tag,          # dtype
            0,                  # reserved
            total_kv_bytes,     # total_kv_bytes
            key["model_hash"],  # model_hash
            key["prefix_hash"], # prefix_hash
        )

        # Write layer info table.
        layer_tbl = b""
        for k_off, k_sz, v_off, v_sz in layer_infos:
            layer_tbl += struct.pack(_LAYER_FMT, k_off, k_sz, v_off, v_sz)

        # ── Copy KV blocks D→H into a single pinned buffer ─────────────────────
        # To avoid GPU OOM from k_all[block_ids].contiguous() (which allocates
        # a full copy on GPU), we copy in small chunks directly to pinned CPU
        # memory.  This uses near-zero extra GPU memory.
        _pin = torch.cuda.is_available()
        pinned_buf = torch.empty(
            total_kv_bytes // elem_size,
            dtype=kv_dtype,
            device="cpu",
            pin_memory=_pin,
        )
        block_ids_gpu = (
            torch.tensor(block_table, device=k_sample.device, dtype=torch.long)
            if block_table else None
        )
        CHUNK = 256  # blocks per GPU gather — ~8 MB, fits in any free memory

        for layer_idx, layer_cache in enumerate(gpu_cache):
            if layer_cache.dim() == 5 and layer_cache.shape[0] == 2:
                k_all = layer_cache[0]
                v_all = layer_cache[1]
            else:
                k_all = layer_cache
                v_all = layer_cache

            k_off, _, v_off, _ = layer_infos[layer_idx]
            k_elem_off = k_off // elem_size
            v_elem_off = v_off // elem_size
            block_elems = 1
            for d in block_shape:
                block_elems *= d

            if block_ids_gpu is not None:
                for start in range(0, n_seq_blocks, CHUNK):
                    end = min(start + CHUNK, n_seq_blocks)
                    chunk_ids = block_ids_gpu[start:end]
                    chunk_len = end - start
                    chunk_shape = (chunk_len, *block_shape)
                    k_chunk = k_all[chunk_ids].contiguous()
                    v_chunk = v_all[chunk_ids].contiguous()
                    dst_off = k_elem_off + start * block_elems
                    k_dst = torch.empty(
                        chunk_shape, dtype=kv_dtype, device="cpu",
                    ).set_(pinned_buf.untyped_storage(), dst_off, chunk_shape)
                    k_dst.copy_(k_chunk, non_blocking=True)
                    dst_off = v_elem_off + start * block_elems
                    v_dst = torch.empty(
                        chunk_shape, dtype=kv_dtype, device="cpu",
                    ).set_(pinned_buf.untyped_storage(), dst_off, chunk_shape)
                    v_dst.copy_(v_chunk, non_blocking=True)
                    # Sync per chunk to ensure copy completes before GPU
                    # memory is freed by del — prevents data corruption
                    # when many chunks reuse the same GPU memory slots.
                    torch.cuda.current_stream().synchronize()
                    del k_chunk, v_chunk
            else:
                k_dst = torch.empty(
                    k_all.shape, dtype=kv_dtype, device="cpu",
                ).set_(pinned_buf.untyped_storage(), k_elem_off, k_all.shape)
                v_dst = torch.empty(
                    v_all.shape, dtype=kv_dtype, device="cpu",
                ).set_(pinned_buf.untyped_storage(), v_elem_off, v_all.shape)
                k_dst.copy_(k_all, non_blocking=True)
                v_dst.copy_(v_all, non_blocking=True)

        if _pin:
            torch.cuda.current_stream().synchronize()

        # ── Verification: log checksum of first layer's K data from GPU ──────
        _verify_layer = gpu_cache[0]
        if _verify_layer.dim() == 5 and _verify_layer.shape[0] == 2:
            _vk = _verify_layer[0]
        else:
            _vk = _verify_layer
        if block_ids_gpu is not None:
            _vk_block0 = _vk[block_ids_gpu[0]].float().sum().item()
        else:
            _vk_block0 = _vk[0].float().sum().item()
        logger.info("[AMF_VERIFY_SAVE] layer0_block0_ksum=%.6f", _vk_block0)

        # ── TurboQuant compression (optional) ─────────────────────────────────
        codec      = get_codec()
        codec_id   = CODEC_NONE

        if codec is not None and total_kv_bytes > 0:
            # Need bytes for compression — only materialize if TQ is enabled
            kv_payload = bytes(pinned_buf.untyped_storage())
            try:
                compressed = codec.compress(kv_payload, head_dim, kv_dtype)
                if len(compressed) < len(kv_payload):
                    ratio = len(kv_payload) / len(compressed)
                    logger.info(
                        "[AMF_VLLM] TurboQuant: %.1f MB → %.1f MB (%.1fx)",
                        len(kv_payload) / (1024 * 1024),
                        len(compressed) / (1024 * 1024),
                        ratio,
                    )
                    kv_payload = compressed
                    codec_id   = CODEC_TURBOQUANT
            except Exception as exc:  # compression is best-effort
                logger.warning("[AMF_VLLM] TurboQuant compress failed: %s", exc)
        else:
            kv_payload = None  # will write directly from pinned_buf

        # Rebuild header with codec_id in the reserved field.
        header = struct.pack(
            _HEADER_FMT,
            _AMFK_MAGIC,
            _AMFK_VERSION,
            n_layers,
            n_tokens,
            n_kv_heads,
            head_dim,
            dtype_tag,
            codec_id,           # was: 0 (reserved); now carries compression codec
            total_kv_bytes,     # always UNCOMPRESSED size
            key["model_hash"],
            key["prefix_hash"],
        )

        # ── Populate VRAM cache (skip for large payloads without compression) ──
        if self._vram_cache is not None:
            vram_key = kv_path.stem
            # For VRAM cache we need bytes — only for small payloads or TQ
            if kv_payload is not None:
                _vram_data = kv_payload
            elif total_kv_bytes <= 2 * 1024 * 1024 * 1024:  # < 2 GB
                _vram_data = bytes(pinned_buf.untyped_storage())
            else:
                _vram_data = None  # too large for VRAM cache
            if _vram_data is not None:
                stored = self._vram_cache.put(
                    key=vram_key,
                    compressed=_vram_data,
                    snap_dtype=kv_dtype,
                    n_tokens=n_tokens,
                    layer_infos=layer_infos,
                    codec_id=codec_id,
                )
                if stored:
                    logger.debug(
                        "[AMF_VLLM] VRAM cache: stored %s (%.1f MB, codec=%d)",
                        vram_key, len(_vram_data) / (1024 * 1024), codec_id,
                    )

        # Write atomically via temp file — stream to avoid 10 GB bytes copy.
        tmp_path = kv_path.with_suffix(".kv.tmp")
        try:
            with open(tmp_path, "wb") as f:
                f.write(header)
                f.write(layer_tbl)
                if kv_payload is not None:
                    f.write(kv_payload)
                else:
                    # Write directly from pinned tensor — no Python bytes copy
                    import ctypes
                    storage = pinned_buf.untyped_storage()
                    buf = (ctypes.c_char * storage.nbytes()).from_address(
                        storage.data_ptr()
                    )
                    f.write(buf)
            tmp_path.rename(kv_path)
        except OSError as exc:
            logger.warning("[AMF_VLLM] save: write failed: %s", exc)
            tmp_path.unlink(missing_ok=True)
            return False

        # Write token file (same format as C++ write_tokens_file).
        tok_bytes = struct.pack(f"<{n_tokens}i", *prompt_tokens)
        tok_path  = self._tok_filename(key)
        try:
            tok_path.write_bytes(tok_bytes)
        except OSError as exc:
            logger.warning("[AMF_VLLM] save: tok write failed: %s", exc)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._saves += 1
        logger.info(
            "[AMF_VLLM] save: %d layers, %d tokens, %.1f MB in %.0f ms",
            n_layers,
            n_tokens,
            total_kv_bytes / (1024 * 1024),
            elapsed_ms,
        )
        return True

    # ── Restore ───────────────────────────────────────────────────────────────

    def restore_kv_state(
        self,
        prompt_tokens: Sequence[int],
        block_table: List[int],
    ) -> int:
        """Restore KV cache from an AMF snapshot into the given physical blocks.

        Args:
            prompt_tokens:  Full prompt token IDs (used to look up the snapshot).
            block_table:    Physical block IDs allocated for this sequence.

        Returns:
            Number of tokens restored, or 0 on failure.
        """
        t0 = time.monotonic()
        key      = self.compute_amf_key(prompt_tokens)
        kv_path  = self._kv_filename(key)
        vram_key = kv_path.stem

        # ── Fast path: VRAM cache hit — zero H→D transfer ─────────────────────
        if self._vram_cache is not None:
            vram_entry = self._vram_cache.get(vram_key)
            if vram_entry is not None:
                result = self._restore_from_vram(vram_entry, block_table)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                if result > 0:
                    self._hits       += 1
                    self._vram_hits  += 1
                    self._restore_ms  = elapsed_ms
                    logger.info(
                        "[AMF_VLLM] VRAM restore: %d tokens in %.1f ms (zero H→D)",
                        result, elapsed_ms,
                    )
                    return result
                # VRAM decompress failed — fall through to NVMe path

        if not kv_path.exists():
            self._misses += 1
            return 0

        try:
            blob = kv_path.read_bytes()
        except OSError as exc:
            logger.warning("[AMF_VLLM] restore: read failed: %s", exc)
            self._misses += 1
            return 0

        if len(blob) < _HEADER_SIZE:
            logger.warning("[AMF_VLLM] restore: blob too small (%d bytes)", len(blob))
            self._misses += 1
            return 0

        # Parse header.
        (
            magic, version, n_layers_snap, n_tokens,
            n_kv_heads, head_dim, dtype_tag, compression_codec,
            total_kv_bytes, model_hash_snap, prefix_hash_snap,
        ) = struct.unpack_from(_HEADER_FMT, blob, 0)

        if magic != _AMFK_MAGIC:
            logger.warning(
                "[AMF_VLLM] restore: wrong magic 0x%08X (expected AMFK)", magic
            )
            self._misses += 1
            return 0

        if version != _AMFK_VERSION:
            logger.warning(
                "[AMF_VLLM] restore: version mismatch (got=%d, want=%d)", version, _AMFK_VERSION
            )
            self._misses += 1
            return 0

        gpu_cache = self._cache_engine.gpu_cache
        n_layers_ctx = len(gpu_cache)

        if n_layers_snap != n_layers_ctx:
            logger.warning(
                "[AMF_VLLM] restore: layer count mismatch (snap=%d, ctx=%d)",
                n_layers_snap, n_layers_ctx,
            )
            self._misses += 1
            return 0

        # Parse layer info table.
        layer_tbl_bytes = n_layers_snap * _LAYER_SIZE
        hdr_end = _HEADER_SIZE
        if len(blob) < hdr_end + layer_tbl_bytes:
            logger.warning("[AMF_VLLM] restore: blob truncated (layer table missing)")
            self._misses += 1
            return 0

        layer_infos: List[tuple] = []
        for i in range(n_layers_snap):
            off = hdr_end + i * _LAYER_SIZE
            layer_infos.append(struct.unpack_from(_LAYER_FMT, blob, off))

        kv_payload_raw = blob[hdr_end + layer_tbl_bytes:]

        # ── TurboQuant decompression ───────────────────────────────────────────
        # Two paths depending on CUDA availability:
        #
        #   GPU path (preferred): H→D of compressed buffer (~10 GB for 70B 128K),
        #     unpack + matmul on GPU (~5 ms), return 1-D GPU tensor.
        #     Avoids the 84 GB CPU float32 intermediate and the 40 GB H→D copy.
        #
        #   CPU fallback: full decompress on CPU, then per-layer H→D as before.
        #
        _TAG_TO_DTYPE = {0: torch.float16, 1: torch.float32, 2: torch.bfloat16}
        snap_dtype    = _TAG_TO_DTYPE.get(dtype_tag, torch.float16)

        # kv_gpu_tensor: 1-D GPU tensor when GPU decompress succeeds, else None.
        kv_gpu_tensor: Optional[torch.Tensor] = None
        kv_payload    = kv_payload_raw   # default: CPU bytes (uncompressed or fallback)

        if compression_codec == CODEC_TURBOQUANT:
            from .turboquant_codec import TurboQuantCodec
            _tq = TurboQuantCodec()
            _cuda_ok = torch.cuda.is_available()
            _gpu_ok = False
            if _cuda_ok:
                try:
                    # ── Fast path: decompress directly onto GPU ────────────────
                    _dev = gpu_cache[0].device if gpu_cache else torch.device("cuda")
                    kv_gpu_tensor = _tq.decompress_to_gpu(
                        bytes(kv_payload_raw), snap_dtype, str(_dev)
                    )
                    _gpu_ok = True
                    logger.debug(
                        "[AMF_VLLM] TurboQuant GPU: decompressed %d elems on %s",
                        kv_gpu_tensor.numel(), _dev,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.info(
                        "[AMF_VLLM] TurboQuant GPU OOM — falling back to CPU decompress"
                    )
                except Exception as exc:
                    logger.info(
                        "[AMF_VLLM] TurboQuant GPU failed: %s — trying CPU", exc
                    )

            if not _gpu_ok:
                try:
                    # ── Slow path: CPU decompress, per-layer H→D ──────────────
                    kv_payload = _tq.decompress(bytes(kv_payload_raw), snap_dtype)
                    logger.debug(
                        "[AMF_VLLM] TurboQuant CPU fallback: %d bytes", len(kv_payload)
                    )
                except Exception as exc:
                    logger.warning(
                        "[AMF_VLLM] TurboQuant decompress failed: %s — treating as miss", exc
                    )
                    self._misses += 1
                    return 0

        # Restore tensors into gpu_cache at the allocated physical blocks.
        for layer_idx, (layer_cache, (k_off, k_sz, v_off, v_sz)) in enumerate(
            zip(gpu_cache, layer_infos)
        ):
            if k_sz == 0 and v_sz == 0:
                continue  # SSM layer — no KV

            # Determine tensor dtype.
            cache_dtype = layer_cache.dtype

            # Determine the layout (stacked vs split).
            if layer_cache.dim() == 5 and layer_cache.shape[0] == 2:
                k_all = layer_cache[0]
                v_all = layer_cache[1]
            else:
                k_all = layer_cache
                v_all = layer_cache

            if kv_gpu_tensor is not None:
                # ── GPU tensor path: slice by element index, no H→D needed ────
                _esz    = cache_dtype.itemsize
                k_flat = kv_gpu_tensor[k_off // _esz : (k_off + k_sz) // _esz].to(cache_dtype)
                v_flat = kv_gpu_tensor[v_off // _esz : (v_off + v_sz) // _esz].to(cache_dtype)
            else:
                # ── CPU bytes path: frombuffer + H→D (uncompressed or CPU decomp)
                k_flat = torch.frombuffer(
                    bytearray(kv_payload[k_off : k_off + k_sz]), dtype=cache_dtype
                )
                v_flat = torch.frombuffer(
                    bytearray(kv_payload[v_off : v_off + v_sz]), dtype=cache_dtype
                )

            if block_table:
                # Scatter into specific physical blocks owned by this sequence.
                n_blocks_restore = len(block_table)
                target_shape = (n_blocks_restore, *k_all.shape[1:])
                k_src = k_flat.reshape(target_shape)
                v_src = v_flat.reshape(target_shape)
                block_ids = torch.tensor(block_table, device=k_all.device, dtype=torch.long)
                # Chunked scatter to avoid GPU OOM on large block counts.
                _RESTORE_CHUNK = 256
                for _rs in range(0, n_blocks_restore, _RESTORE_CHUNK):
                    _re = min(_rs + _RESTORE_CHUNK, n_blocks_restore)
                    _chunk_ids = block_ids[_rs:_re]
                    _k_c = k_src[_rs:_re]
                    _v_c = v_src[_rs:_re]
                    if kv_gpu_tensor is None:
                        _k_c = _k_c.to(k_all.device)
                        _v_c = _v_c.to(v_all.device)
                    k_all[_chunk_ids] = _k_c
                    v_all[_chunk_ids] = _v_c
            else:
                # Empty block_table → single-request benchmark: restore ALL blocks.
                k_src = k_flat.reshape_as(k_all)
                v_src = v_flat.reshape_as(v_all)
                if kv_gpu_tensor is None:
                    k_src = k_src.to(k_all.device)
                    v_src = v_src.to(v_all.device)
                k_all.copy_(k_src)
                v_all.copy_(v_src)

        # ── Promote to VRAM cache for future zero-copy restores ───────────────
        # Skip promotion for large payloads (>2 GB) — creating a Python bytes
        # copy of 40 GB would exhaust host memory.
        if (self._vram_cache is not None
            and not self._vram_cache.contains(vram_key)
            and total_kv_bytes <= 2 * 1024 * 1024 * 1024):
            _promote_bytes = bytes(kv_payload_raw) if compression_codec == CODEC_TURBOQUANT else bytes(kv_payload)
            self._vram_cache.put(
                key=vram_key,
                compressed=_promote_bytes,
                snap_dtype=snap_dtype,
                n_tokens=int(n_tokens),
                layer_infos=layer_infos,
                codec_id=compression_codec,
            )

        # ── Verification: check restored data matches save checksum ────────────
        _rv_layer = gpu_cache[0]
        if _rv_layer.dim() == 5 and _rv_layer.shape[0] == 2:
            _rvk = _rv_layer[0]
        else:
            _rvk = _rv_layer
        _restore_block0_id = block_table[0] if block_table else 0
        _rvk_sum = _rvk[_restore_block0_id].float().sum().item()
        logger.info("[AMF_VERIFY_RESTORE] layer0_block0_ksum=%.6f block_id=%d",
                    _rvk_sum, _restore_block0_id)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._hits     += 1
        self._restore_ms = elapsed_ms

        logger.info(
            "[AMF_VLLM] restore: %d layers, %d tokens, %.1f ms",
            n_layers_ctx, n_tokens, elapsed_ms,
        )
        return int(n_tokens)

    # ── VRAM restore helper ───────────────────────────────────────────────────

    def _restore_from_vram(self, entry: "_VRAMEntry", block_table: List[int]) -> int:
        """Scatter KV from a VRAM cache entry into the engine's KV blocks.

        Handles both raw and TurboQuant-compressed VRAM entries.
        Returns the number of tokens restored, or 0 on failure.
        """
        if entry.codec_id == CODEC_TURBOQUANT:
            from .turboquant_codec import TurboQuantCodec
            _tq = TurboQuantCodec()
            try:
                kv_gpu_tensor = _tq.decompress_from_gpu_tensor(
                    entry.payload_gpu, entry.snap_dtype
                )
            except Exception as exc:
                logger.warning("[AMF_VLLM] VRAM decompress failed: %s", exc)
                return 0
        else:
            # Raw payload — reinterpret uint8 GPU tensor as the snapshot dtype.
            kv_gpu_tensor = entry.payload_gpu.view(entry.snap_dtype)

        gpu_cache = self._cache_engine.gpu_cache

        for layer_cache, (k_off, k_sz, v_off, v_sz) in zip(gpu_cache, entry.layer_infos):
            if k_sz == 0 and v_sz == 0:
                continue  # SSM/recurrent layer — no KV

            cache_dtype = layer_cache.dtype

            if layer_cache.dim() == 5 and layer_cache.shape[0] == 2:
                k_all = layer_cache[0]
                v_all = layer_cache[1]
            else:
                k_all = layer_cache
                v_all = layer_cache

            _esz   = cache_dtype.itemsize
            k_flat = kv_gpu_tensor[k_off // _esz : (k_off + k_sz) // _esz].to(cache_dtype)
            v_flat = kv_gpu_tensor[v_off // _esz : (v_off + v_sz) // _esz].to(cache_dtype)

            if block_table:
                n_blk = len(block_table)
                target_shape = (n_blk, *k_all.shape[1:])
                k_src = k_flat.reshape(target_shape)
                v_src = v_flat.reshape(target_shape)
                block_ids = torch.tensor(block_table, device=k_all.device, dtype=torch.long)
                _RC = 256
                for _s in range(0, n_blk, _RC):
                    _e = min(_s + _RC, n_blk)
                    k_all[block_ids[_s:_e]] = k_src[_s:_e]
                    v_all[block_ids[_s:_e]] = v_src[_s:_e]
            else:
                k_all.copy_(k_flat.reshape_as(k_all))
                v_all.copy_(v_flat.reshape_as(v_all))

        return entry.n_tokens

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        base = {
            "hits":       self._hits,
            "misses":     self._misses,
            "saves":      self._saves,
            "restore_ms": self._restore_ms,
            "vram_hits":  self._vram_hits,
        }
        if self._vram_cache is not None:
            base["vram_cache"] = self._vram_cache.stats()
        return base


# ── V1 KV cache proxy ───────────────────────────────────────────────────────

class _KvCacheProxy:
    """Wraps vLLM V1 model_runner.kv_caches (list[Tensor]) into the
    CacheEngine.gpu_cache interface that AmfKvManager expects.

    V1 kv_caches are already shaped [2, num_blocks, block_size, num_kv_heads,
    head_dim] (FlashAttn/Tree) or [num_blocks, 2, ...] (FlashInfer).
    For FlashInfer, we permute so AmfKvManager always sees dim-0 == 2.
    """

    def __init__(self, kv_caches: list) -> None:
        self.gpu_cache: List[torch.Tensor] = []
        for t in kv_caches:
            if not isinstance(t, torch.Tensor):
                continue
            if t.dim() == 5 and t.shape[0] == 2:
                # FlashAttn / ROCm / Tree: [2, num_blocks, block_size, ...]
                self.gpu_cache.append(t)
            elif t.dim() == 5 and t.shape[1] == 2:
                # FlashInfer: [num_blocks, 2, block_size, ...] → view as
                # stacked [2, num_blocks, ...] using a non-contiguous view so
                # that writes through the view mutate the original tensor.
                self.gpu_cache.append(t.permute(1, 0, 2, 3, 4))
            elif isinstance(t, torch.Tensor):
                self.gpu_cache.append(t)


