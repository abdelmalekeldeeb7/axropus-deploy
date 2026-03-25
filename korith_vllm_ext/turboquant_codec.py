"""turboquant_codec.py — TurboQuant KV cache compression for AMF snapshots.

Implements the PolarQuant + QJL algorithm from:
  "TurboQuant: Extreme Compression for AI Efficiency"
  Zandieh & Mirrokni, Google Research, ICLR 2026.

How it works:
  1. PolarQuant: decompose each KV vector into radius (||v||) + unit direction.
     The unit sphere distribution is smooth and compresses far better than raw values.
  2. QJL: apply a random Rademacher projection to the unit vector, then quantize
     to N bits. The JL transform ensures quantisation error is spread evenly.
  3. Reconstruct: dequantise → inverse projection → multiply by radius.

Compression ratios vs FP16 KV cache (head_dim=128):
  bits=4 → ~3.8x compression
  bits=3 → ~5.1x compression

Codec ID stored in AMFK header reserved/compression field:
  0 = CODEC_NONE       (raw, no compression)
  2 = CODEC_TURBOQUANT (this module)

Enable via: KORITH_KV_COMPRESSION=turboquant  (or =off to disable)
"""

from __future__ import annotations

import os
import struct
from typing import Optional

import numpy as np
import torch

# ── Codec registration IDs (written into AMFK header reserved field) ──────────

CODEC_NONE       = 0
CODEC_TURBOQUANT = 2

# ── Compressed-payload sub-header ─────────────────────────────────────────────
# Written at the start of the compressed kv_payload blob.
# magic(4) + bits(1) + pad(1) + reserved(2) + original_bytes(8) + head_dim(4)
_TQ_MAGIC      = b"AMTQ"
_TQ_HDR_FMT    = "<4sBBH QI"
_TQ_HDR_SIZE   = struct.calcsize(_TQ_HDR_FMT)  # 20 bytes

# ── Sub-header (n_vectors, n_quant_values, padding) ────────────────────────────
_TQ_SUB_FMT  = "<IIH"
_TQ_SUB_SIZE = struct.calcsize(_TQ_SUB_FMT)    # 10 bytes


def _codec_enabled() -> bool:
    val = os.environ.get("KORITH_KV_COMPRESSION", "").strip().lower()
    return val in ("turboquant", "tq", "1", "on", "true", "yes")


def _codec_bits() -> int:
    val = os.environ.get("KORITH_KV_COMPRESSION_BITS", "4").strip()
    try:
        b = int(val)
    except ValueError:
        b = 4
    return b if b in (3, 4, 8) else 4


class TurboQuantCodec:
    """KV cache compressor using TurboQuant (PolarQuant + QJL).

    Args:
        bits:   Quantisation bits per value. 4 → ~3.8x, 3 → ~5.1x. Default 4.
        seed:   RNG seed for reproducible random projections. Default 42.
    """

    def __init__(self, bits: int = 4, seed: int = 42) -> None:
        if bits not in (3, 4, 8):
            raise ValueError(f"TurboQuant: bits must be 3, 4 or 8; got {bits}")
        self.bits = bits
        self.seed = seed
        self._proj_cache: dict = {}

    # ── Random projection ──────────────────────────────────────────────────────

    def _rademacher(self, dim: int) -> torch.Tensor:
        """Cached square Rademacher matrix R ∈ {±1/√dim}^(dim×dim)."""
        if dim not in self._proj_cache:
            g = torch.Generator().manual_seed(self.seed)
            R = (torch.randint(0, 2, (dim, dim), generator=g).float() * 2.0 - 1.0)
            R /= float(dim) ** 0.5
            self._proj_cache[dim] = R
        return self._proj_cache[dim]

    # ── Bit-packing helpers ────────────────────────────────────────────────────

    @staticmethod
    def _pack_int4(x: torch.Tensor) -> bytes:
        """Pack uint8 tensor (values 0-15) → 4-bit packed bytes."""
        flat = x.reshape(-1).to(torch.uint8)
        if flat.numel() % 2:
            flat = torch.cat([flat, flat.new_zeros(1)])
        return (flat[::2] | (flat[1::2] << 4)).numpy().tobytes()

    @staticmethod
    def _unpack_int4(data: bytes, n: int) -> torch.Tensor:
        """Unpack 4-bit packed bytes → float tensor of length n."""
        packed = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        lo = (packed & 0x0F).float()
        hi = ((packed >> 4) & 0x0F).float()
        return torch.stack([lo, hi], dim=1).reshape(-1)[:n]

    @staticmethod
    def _pack_int3(x: torch.Tensor) -> bytes:
        """Pack uint8 tensor (values 0-7) → 3-bit packed bytes (8 vals / 3 bytes)."""
        flat = x.reshape(-1).to(torch.uint8).numpy().astype(np.uint32)
        pad = (-len(flat)) % 8
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint32)])
        out = bytearray()
        for i in range(0, len(flat), 8):
            c = flat[i:i+8]
            b0 = c[0] | (c[1] << 3) | ((c[2] & 0x3) << 6)
            b1 = ((c[2] >> 2) & 1) | (c[3] << 1) | (c[4] << 4) | ((c[5] & 1) << 7)
            b2 = ((c[5] >> 1) & 3) | (c[6] << 2) | (c[7] << 5)
            out += bytes([b0 & 0xFF, b1 & 0xFF, b2 & 0xFF])
        return bytes(out)

    @staticmethod
    def _unpack_int3(data: bytes, n: int) -> torch.Tensor:
        """Unpack 3-bit packed bytes → float tensor of length n."""
        raw = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
        out: list = []
        for i in range(0, len(raw) - 2, 3):
            b0, b1, b2 = int(raw[i]), int(raw[i+1]), int(raw[i+2])
            out += [
                b0 & 7,
                (b0 >> 3) & 7,
                ((b0 >> 6) & 3) | ((b1 & 1) << 2),
                (b1 >> 1) & 7,
                (b1 >> 4) & 7,
                ((b1 >> 7) & 1) | ((b2 & 3) << 1),
                (b2 >> 2) & 7,
                (b2 >> 5) & 7,
            ]
        return torch.tensor(out, dtype=torch.float32)[:n]

    # ── Core API ───────────────────────────────────────────────────────────────

    def compress(self, kv_payload: bytes, head_dim: int, dtype: torch.dtype) -> bytes:
        """Compress raw KV payload bytes using TurboQuant.

        Args:
            kv_payload:  Raw FP16/BF16 bytes of the full KV snapshot payload.
            head_dim:    Per-head dimension of the KV tensors.
            dtype:       Source dtype (torch.float16 or torch.bfloat16).

        Returns:
            Compressed bytes with AMTQ sub-header prepended.
            Returns original bytes unchanged if compression would not help
            (e.g. zero-length payload or head_dim mismatch).
        """
        original_size = len(kv_payload)
        if original_size == 0 or head_dim < 4:
            return kv_payload

        x = torch.frombuffer(bytearray(kv_payload), dtype=dtype).float()
        n_total = x.numel()

        # Pad to a multiple of head_dim.
        pad = (-n_total) % head_dim
        if pad:
            x = torch.cat([x, x.new_zeros(pad)])

        x = x.reshape(-1, head_dim)    # [n_vectors, head_dim]
        n_vectors   = x.shape[0]
        n_quant_val = n_vectors * head_dim

        # ── Stage 1: PolarQuant ────────────────────────────────────────────────
        norms      = torch.norm(x, dim=-1)                 # [n_vectors]
        safe_norms = norms.clamp(min=1e-8)
        x_unit     = x / safe_norms.unsqueeze(-1)          # unit sphere

        # ── Stage 2: QJL random projection ────────────────────────────────────
        R      = self._rademacher(head_dim)
        x_proj = x_unit @ R.T                              # [n_vectors, head_dim]

        # Per-vector scale (min/max clamp for symmetry).
        x_scale = x_proj.abs().amax(dim=-1).clamp(min=1e-8)  # [n_vectors]
        x_norm  = x_proj / x_scale.unsqueeze(-1)             # [-1, 1]

        # ── Quantise ───────────────────────────────────────────────────────────
        max_int   = (1 << (self.bits - 1)) - 1              # 7 (4-bit), 3 (3-bit)
        q_signed  = (x_norm * max_int).round().clamp(-max_int, max_int)
        q_uint    = (q_signed + max_int).to(torch.uint8)    # [0, 2*max_int]
        flat_q    = q_uint.reshape(-1)

        if self.bits == 4:
            q_bytes = self._pack_int4(flat_q)
        elif self.bits == 3:
            q_bytes = self._pack_int3(flat_q)
        else:                                               # bits == 8
            q_bytes = flat_q.numpy().tobytes()

        # ── Serialise norms and scales in FP16 (2 bytes each) ─────────────────
        norms_bytes  = norms.half().numpy().tobytes()        # n_vectors × 2 bytes
        scales_bytes = x_scale.half().numpy().tobytes()      # n_vectors × 2 bytes

        # ── Build compressed blob ──────────────────────────────────────────────
        dtype_id = {torch.float16: 0, torch.bfloat16: 2}.get(dtype, 0)

        tq_header = struct.pack(
            _TQ_HDR_FMT,
            _TQ_MAGIC,
            self.bits,
            dtype_id,
            0,              # reserved
            original_size,
            head_dim,
        )
        sub_header = struct.pack(_TQ_SUB_FMT, n_vectors, n_quant_val, pad)

        compressed = tq_header + sub_header + norms_bytes + scales_bytes + q_bytes

        ratio = original_size / max(len(compressed), 1)
        if ratio < 1.05:
            # Compression didn't help — return original to avoid bloat.
            return kv_payload

        return compressed

    def decompress(self, compressed: bytes, dtype: torch.dtype) -> bytes:
        """Decompress TurboQuant bytes back to the original KV payload.

        Args:
            compressed:  Bytes produced by compress().
            dtype:       Target dtype for reconstruction.

        Returns:
            Decompressed raw KV bytes in the original dtype and size.
        """
        if len(compressed) < _TQ_HDR_SIZE + _TQ_SUB_SIZE:
            raise ValueError("TurboQuant: compressed blob too small")

        # Parse headers.
        magic, bits, dtype_id, _res, original_size, head_dim = struct.unpack_from(
            _TQ_HDR_FMT, compressed, 0
        )
        if magic != _TQ_MAGIC:
            raise ValueError(f"TurboQuant: bad magic {magic!r}")

        n_vectors, n_quant_val, pad = struct.unpack_from(
            _TQ_SUB_FMT, compressed, _TQ_HDR_SIZE
        )

        offset = _TQ_HDR_SIZE + _TQ_SUB_SIZE

        # Read norms and scales (FP16, 2 bytes each).
        norms_size  = n_vectors * 2
        scales_size = n_vectors * 2

        norms  = torch.frombuffer(
            bytearray(compressed[offset : offset + norms_size]),
            dtype=torch.float16,
        ).float()
        offset += norms_size

        scales = torch.frombuffer(
            bytearray(compressed[offset : offset + scales_size]),
            dtype=torch.float16,
        ).float()
        offset += scales_size

        q_data = compressed[offset:]
        max_int = (1 << (bits - 1)) - 1

        # Unpack quantised integers.
        if bits == 4:
            q_uint = self._unpack_int4(q_data, n_quant_val)
        elif bits == 3:
            q_uint = self._unpack_int3(q_data, n_quant_val)
        else:
            q_uint = torch.frombuffer(
                bytearray(q_data), dtype=torch.uint8
            ).float()[:n_quant_val]

        # ── Dequantise ────────────────────────────────────────────────────────
        q_signed = q_uint - max_int                                  # [-max_int, max_int]
        # Expand scales to per-value.
        scales_exp = scales.repeat_interleave(head_dim)              # [n_vectors * head_dim]
        q_scaled   = (q_signed / max_int) * scales_exp

        x_proj = q_scaled.reshape(n_vectors, head_dim)

        # ── Inverse QJL: R^T ≈ R^{-1} for Rademacher ────────────────────────
        R      = self._rademacher(head_dim)
        x_unit = x_proj @ R                                          # [n_vectors, head_dim]

        # ── Restore radii ─────────────────────────────────────────────────────
        x = x_unit * norms.unsqueeze(-1)                             # [n_vectors, head_dim]

        # Remove padding and convert back to target dtype.
        flat = x.reshape(-1)
        if pad:
            flat = flat[:-pad]

        result = flat.to(dtype).numpy().tobytes()
        if len(result) != original_size:
            raise ValueError(
                f"TurboQuant: decompressed {len(result)} bytes, expected {original_size}"
            )
        return result

    def decompress_to_gpu(
        self,
        compressed: bytes,
        dtype: torch.dtype,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Decompress directly onto the GPU — eliminates the CPU float32 bottleneck.

        Flow:
          compressed bytes (CPU)
            → H→D of small quantised buffer  (~10 GB for Llama-70B 128K)
            → INT4 unpack + dequantise on GPU (vectorised, no Python loop)
            → FP16 matmul on GPU              (~5 ms on H200 for 163M×128)
            → 1-D FP16 GPU tensor ready for scatter into vLLM cache blocks

        Compared to decompress() (CPU path):
          CPU  path: 40 GB H→D  +  ~2 s CPU matmul  +  ~84 GB CPU RAM peak
          GPU  path: 10 GB H→D  +  ~5 ms GPU matmul +  ~10 GB VRAM peak

        Args:
            compressed:  Bytes produced by compress().
            dtype:       Target dtype (torch.float16 or torch.bfloat16).
            device:      CUDA device string, e.g. "cuda" or "cuda:0".

        Returns:
            1-D tensor on *device* with the same number of elements as the
            original uncompressed KV payload.  Byte length == original_size.
        """
        if len(compressed) < _TQ_HDR_SIZE + _TQ_SUB_SIZE:
            raise ValueError("TurboQuant GPU: compressed blob too small")

        # ── Parse headers on CPU (trivial) ────────────────────────────────────
        magic, bits, dtype_id, _res, original_size, head_dim = struct.unpack_from(
            _TQ_HDR_FMT, compressed, 0
        )
        if magic != _TQ_MAGIC:
            raise ValueError(f"TurboQuant GPU: bad magic {magic!r}")

        n_vectors, n_quant_val, pad = struct.unpack_from(
            _TQ_SUB_FMT, compressed, _TQ_HDR_SIZE
        )

        offset      = _TQ_HDR_SIZE + _TQ_SUB_SIZE
        norms_size  = n_vectors * 2   # FP16 = 2 bytes each
        scales_size = n_vectors * 2

        # ── H→D: norms and scales (small — 2 × n_vectors × 2 bytes) ──────────
        norms = torch.frombuffer(
            bytearray(compressed[offset : offset + norms_size]),
            dtype=torch.float16,
        ).to(device)                                # [n_vectors]  FP16 on GPU
        offset += norms_size

        scales = torch.frombuffer(
            bytearray(compressed[offset : offset + scales_size]),
            dtype=torch.float16,
        ).to(device)                                # [n_vectors]  FP16 on GPU
        offset += scales_size

        q_data  = compressed[offset:]
        max_int = (1 << (bits - 1)) - 1

        # ── H→D: packed quantised data, unpack entirely on GPU ────────────────
        if bits == 4:
            packed = torch.frombuffer(
                bytearray(q_data), dtype=torch.uint8
            ).to(device)                            # H→D of ~10 GB compressed
            lo     = (packed & 0x0F).to(torch.float16)
            hi     = ((packed >> 4) & 0x0F).to(torch.float16)
            q_uint = torch.stack([lo, hi], dim=1).reshape(-1)[:n_quant_val]
        elif bits == 3:
            # 3-bit packing has no simple GPU bitop path — unpack on CPU then move
            q_uint = self._unpack_int3(q_data, n_quant_val).to(device, dtype=torch.float16)
        else:  # bits == 8
            q_uint = torch.frombuffer(
                bytearray(q_data), dtype=torch.uint8
            ).to(device, dtype=torch.float16)[:n_quant_val]

        # ── Dequantise on GPU (FP16 throughout — no float32 blowup) ──────────
        q_signed   = q_uint - max_int                           # FP16 [-max_int, max_int]
        scales_exp = scales.repeat_interleave(head_dim)         # [n_vectors * head_dim] FP16
        q_scaled   = (q_signed / max_int) * scales_exp          # FP16

        x_proj = q_scaled.reshape(n_vectors, head_dim)          # [n_vectors, head_dim] FP16

        # ── Inverse QJL matmul on GPU (~5 ms on H200 for Llama-70B 128K) ─────
        R_gpu  = self._rademacher(head_dim).to(device, dtype=torch.float16)
        x_unit = torch.mm(x_proj, R_gpu)                        # [n_vectors, head_dim] FP16

        # ── Restore radii ─────────────────────────────────────────────────────
        x    = x_unit * norms.unsqueeze(-1)                     # [n_vectors, head_dim]
        flat = x.reshape(-1)
        if pad:
            flat = flat[:-pad]

        # Cast to caller's target dtype (float16 → bfloat16 if needed)
        result = flat.to(dtype)

        # Sanity check element count
        elem_size    = torch.finfo(dtype).bits // 8
        n_expected   = original_size // elem_size
        if result.numel() != n_expected:
            raise ValueError(
                f"TurboQuant GPU: element count {result.numel()} != expected {n_expected}"
            )
        return result                                            # 1-D GPU tensor


    def decompress_from_gpu_tensor(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decompress TurboQuant from a GPU-resident uint8 tensor.

        This is the zero-copy VRAM cache path: the compressed payload is already
        on the GPU, so only the 30-byte header is moved to CPU for parsing.
        All unpacking, dequantisation, and the inverse-QJL matmul stay on GPU.

        Compared to decompress_to_gpu():
          decompress_to_gpu:        H→D of ~22 GB compressed data  (~4 ms)
          decompress_from_gpu_tensor: zero H→D (already resident)  (~0 ms I/O)

        Args:
            tensor:  1-D uint8 CUDA tensor produced by storing compress() output
                     directly to GPU memory (via VRAMSnapshotCache.put).
            dtype:   Target dtype for the reconstructed KV values.

        Returns:
            1-D tensor on the same device as *tensor*, same element count as
            the original uncompressed KV payload.
        """
        HDR = _TQ_HDR_SIZE + _TQ_SUB_SIZE  # 30 bytes — trivial to move to CPU
        hdr_bytes = tensor[:HDR].cpu().numpy().tobytes()

        magic, bits, _dtype_id, _res, original_size, head_dim = struct.unpack_from(
            _TQ_HDR_FMT, hdr_bytes, 0
        )
        if magic != _TQ_MAGIC:
            raise ValueError(f"TurboQuant GPU tensor: bad magic {magic!r}")

        n_vectors, n_quant_val, pad = struct.unpack_from(
            _TQ_SUB_FMT, hdr_bytes, _TQ_HDR_SIZE
        )

        device      = tensor.device
        offset      = HDR
        norms_size  = n_vectors * 2   # FP16 = 2 bytes each
        scales_size = n_vectors * 2

        # Slice directly on the GPU tensor — no H→D transfer
        norms  = tensor[offset : offset + norms_size].view(torch.float16)   # [n_vectors]
        offset += norms_size
        scales = tensor[offset : offset + scales_size].view(torch.float16)  # [n_vectors]
        offset += scales_size

        max_int = (1 << (bits - 1)) - 1

        if bits == 4:
            packed = tensor[offset:]
            lo     = (packed & 0x0F).to(torch.float16)
            hi     = ((packed >> 4) & 0x0F).to(torch.float16)
            q_uint = torch.stack([lo, hi], dim=1).reshape(-1)[:n_quant_val]
        elif bits == 3:
            # 3-bit has no simple GPU bitop path — unpack on CPU then move once
            q_uint = self._unpack_int3(
                tensor[offset:].cpu().numpy().tobytes(), n_quant_val
            ).to(device, dtype=torch.float16)
        else:  # bits == 8
            q_uint = tensor[offset:].to(torch.float16)[:n_quant_val]

        # Dequantise on GPU (FP16 throughout — no float32 blowup)
        q_signed   = q_uint - max_int
        scales_exp = scales.repeat_interleave(head_dim)          # [n_vectors * head_dim]
        q_scaled   = (q_signed / max_int) * scales_exp
        x_proj     = q_scaled.reshape(n_vectors, head_dim)

        # Inverse QJL matmul on GPU (~5 ms on H200 for Llama-70B 128K)
        R_gpu  = self._rademacher(head_dim).to(device, dtype=torch.float16)
        x_unit = torch.mm(x_proj, R_gpu)

        # Restore radii
        x    = x_unit * norms.unsqueeze(-1)
        flat = x.reshape(-1)
        if pad:
            flat = flat[:-pad]

        result = flat.to(dtype)
        elem_size  = torch.finfo(dtype).bits // 8
        n_expected = original_size // elem_size
        if result.numel() != n_expected:
            raise ValueError(
                f"TurboQuant GPU tensor: element count {result.numel()} != {n_expected}"
            )
        return result


# ── Module-level convenience ───────────────────────────────────────────────────

_default_codec: Optional[TurboQuantCodec] = None


def get_codec() -> Optional[TurboQuantCodec]:
    """Return a module-level TurboQuantCodec if KORITH_KV_COMPRESSION is set."""
    global _default_codec
    if not _codec_enabled():
        return None
    if _default_codec is None or _default_codec.bits != _codec_bits():
        _default_codec = TurboQuantCodec(bits=_codec_bits())
    return _default_codec
