"""amf_codec.py — Pluggable codec layer for AMF KV snapshots.

Provides a single Codec interface that sits between "gather KV tensor" and
"write bytes to disk" (save side) / "read bytes" and "scatter KV tensor"
(restore side).

Design principle: **don't break the 166x v1 path**.
  - Raw codec and TurboQuant codec continue using the existing v1 "AMFK"
    file format, byte-for-byte identical to pre-codec behaviour.
  - New codecs (FP8 with scale sidecar, INT4, PolarQuant, QJL) use a new
    "AMF2" v2 format with a dedicated sidecar section for metadata.
  - Readers dispatch on magic bytes so v1 snapshots keep loading forever.

Codec IDs (4-byte field in header):
  0 = CODEC_NONE       raw, no metadata
  1 = CODEC_FP8        FP8 e4m3fn/e5m2 with k_scale/v_scale sidecar
  2 = CODEC_TURBOQUANT PolarQuant + QJL (existing, v1 format)
  3 = CODEC_INT4       per-block INT4 with scale/zero sidecar
  4 = CODEC_POLARQUANT PolarQuant only (no QJL)
  5 = CODEC_QJL        QJL only (no PolarQuant)
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# ── Codec IDs ────────────────────────────────────────────────────────────────

CODEC_NONE       = 0
CODEC_FP8        = 1
CODEC_TURBOQUANT = 2
CODEC_INT4       = 3
CODEC_POLARQUANT = 4
CODEC_QJL        = 5

CODEC_NAMES = {
    CODEC_NONE:       "raw",
    CODEC_FP8:        "fp8",
    CODEC_TURBOQUANT: "turboquant",
    CODEC_INT4:       "int4",
    CODEC_POLARQUANT: "polarquant",
    CODEC_QJL:        "qjl",
}

# Codecs that route through the v1 "AMFK" file format (no sidecar).
# Everything else uses "AMF2" v2 format.
V1_COMPATIBLE_CODECS = {CODEC_NONE, CODEC_TURBOQUANT}

# ── v2 file format ───────────────────────────────────────────────────────────

_AMF2_MAGIC   = 0x32464D41  # "AMF2"
_AMF2_VERSION = 2

# Header layout (little-endian):
#   magic(4) + version(4) + codec_id(4) + n_layers(4) + n_tokens(4) +
#   n_kv_heads(4) + head_dim(4) + dtype_tag(4) +
#   total_kv_bytes(8) + payload_bytes(8) + sidecar_bytes(8) +
#   model_hash(8) + prefix_hash(8)
# Total: 72 bytes
_AMF2_HEADER_FMT  = "<IIIIIIII QQQQQ"
_AMF2_HEADER_SIZE = struct.calcsize(_AMF2_HEADER_FMT)

assert _AMF2_HEADER_SIZE == 72, f"AMF2 header size changed: {_AMF2_HEADER_SIZE}"


@dataclass
class CodecContext:
    """Context passed to codec encode/decode methods.

    Contains everything a codec might need that is not in the raw tensor
    payload itself — model layers for scale capture, block shape, etc.
    """
    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_size: int
    kv_dtype: torch.dtype
    # Optional — populated by the worker extension when the codec needs
    # access to per-layer model state (FP8 scales, etc.).
    scales: Optional[dict] = None
    # Model runner reference for codecs that need to read/write layer state.
    model_runner: Any = None


@dataclass
class EncodedBlob:
    """Result of encoding a KV payload with a codec."""
    payload: bytes          # compressed/quantized/raw bytes
    sidecar: bytes          # codec-specific metadata (scales, zero points, etc.)
    codec_id: int           # which codec produced this blob


# ── Codec base class ─────────────────────────────────────────────────────────

class Codec:
    """Base class. Subclasses override encode/decode."""

    id: int = CODEC_NONE
    name: str = "raw"

    def encode(
        self,
        kv_payload: bytes,
        ctx: CodecContext,
    ) -> EncodedBlob:
        """Encode raw KV bytes into codec-specific payload + sidecar.

        The default implementation is the no-op (raw) codec — bytes pass
        through unchanged, no sidecar.
        """
        return EncodedBlob(payload=kv_payload, sidecar=b"", codec_id=self.id)

    def decode(
        self,
        payload: bytes,
        sidecar: bytes,
        ctx: CodecContext,
    ) -> bytes:
        """Decode codec payload back into raw KV bytes."""
        return payload


# ── Raw codec (v1 compatible) ────────────────────────────────────────────────

class RawCodec(Codec):
    id = CODEC_NONE
    name = "raw"


# ── FP8 codec with scale sidecar ─────────────────────────────────────────────

# Sidecar layout for FP8:
#   magic(4) "FP8S" + version(4)=1 + n_layers(4) + flags(4)
#   then per-layer: k_scale(f32) + v_scale(f32) + q_scale(f32) + prob_scale(f32)
_FP8_SIDECAR_MAGIC = 0x53385046  # "FP8S"
_FP8_SIDECAR_HDR_FMT = "<IIII"
_FP8_SIDECAR_HDR_SIZE = struct.calcsize(_FP8_SIDECAR_HDR_FMT)

_FP8_SCALES_PER_LAYER = 4  # k, v, q, prob
_FP8_SIDECAR_LAYER_FMT = "<ffff"
_FP8_SIDECAR_LAYER_SIZE = struct.calcsize(_FP8_SIDECAR_LAYER_FMT)


class Fp8Codec(Codec):
    """FP8 e4m3fn/e5m2 KV cache with per-layer scale sidecar.

    The payload is the raw FP8 bytes (byte-for-byte identical to what vLLM
    stored in the KV cache). The sidecar contains per-layer `_k_scale`,
    `_v_scale`, `_q_scale`, `_prob_scale` floats captured at save time.

    On restore, the sidecar is applied back to the model's attention layers
    AND `calculate_kv_scales` is forced to False so vLLM doesn't recompute
    scales from the warm batch's suffix tokens (which would desynchronize
    from the scales under which the saved bytes were quantized).
    """

    id = CODEC_FP8
    name = "fp8"

    def encode(self, kv_payload: bytes, ctx: CodecContext) -> EncodedBlob:
        scales = ctx.scales or {}
        n_layers = ctx.n_layers
        if not scales:
            logger.warning(
                "[AMF_CODEC] FP8 encode: no scales provided — dequant will "
                "be incorrect on restore. Pass ctx.scales from worker ext."
            )

        hdr = struct.pack(
            _FP8_SIDECAR_HDR_FMT,
            _FP8_SIDECAR_MAGIC,
            1,           # sidecar version
            n_layers,
            0,           # flags (reserved)
        )

        layer_bytes = bytearray()
        for layer_idx in range(n_layers):
            k = float(scales.get(f"L{layer_idx}.k", 1.0))
            v = float(scales.get(f"L{layer_idx}.v", 1.0))
            q = float(scales.get(f"L{layer_idx}.q", 1.0))
            p = float(scales.get(f"L{layer_idx}.prob", 1.0))
            layer_bytes += struct.pack(_FP8_SIDECAR_LAYER_FMT, k, v, q, p)

        sidecar = bytes(hdr) + bytes(layer_bytes)
        return EncodedBlob(
            payload=kv_payload,
            sidecar=sidecar,
            codec_id=self.id,
        )

    def decode(self, payload: bytes, sidecar: bytes, ctx: CodecContext) -> bytes:
        # Restore raw FP8 bytes — dequant happens inside the attention kernel.
        # The worker extension is responsible for applying the scale sidecar
        # back onto the model's attention layers (see parse_fp8_sidecar).
        return payload


def parse_fp8_sidecar(sidecar: bytes) -> dict:
    """Parse an FP8 scale sidecar back into a {layer_key: float} dict.

    Returns empty dict on parse failure.
    """
    if len(sidecar) < _FP8_SIDECAR_HDR_SIZE:
        return {}

    magic, version, n_layers, _flags = struct.unpack_from(
        _FP8_SIDECAR_HDR_FMT, sidecar, 0
    )
    if magic != _FP8_SIDECAR_MAGIC:
        logger.warning("[AMF_CODEC] FP8 sidecar: wrong magic 0x%08X", magic)
        return {}
    if version != 1:
        logger.warning("[AMF_CODEC] FP8 sidecar: unsupported version %d", version)
        return {}

    expected_size = _FP8_SIDECAR_HDR_SIZE + n_layers * _FP8_SIDECAR_LAYER_SIZE
    if len(sidecar) < expected_size:
        logger.warning(
            "[AMF_CODEC] FP8 sidecar: truncated (%d < %d)",
            len(sidecar), expected_size,
        )
        return {}

    scales: dict = {}
    for layer_idx in range(n_layers):
        off = _FP8_SIDECAR_HDR_SIZE + layer_idx * _FP8_SIDECAR_LAYER_SIZE
        k, v, q, p = struct.unpack_from(_FP8_SIDECAR_LAYER_FMT, sidecar, off)
        scales[f"L{layer_idx}.k"]    = k
        scales[f"L{layer_idx}.v"]    = v
        scales[f"L{layer_idx}.q"]    = q
        scales[f"L{layer_idx}.prob"] = p
    return scales


def capture_fp8_scales(model_runner: Any) -> dict:
    """Walk the model's attention layers and capture per-layer FP8 scales.

    Returns a dict keyed by "L{layer_idx}.{k|v|q|prob}" → float.
    Handles both ``_k_scale`` (tensor) and ``_k_scale_float`` (python float)
    attribute styles used across vLLM attention backends.

    Layer indexing: iterates ``named_modules()`` in order and assigns an
    incrementing layer index to each module that has a ``_k_scale`` or
    ``k_scale`` attribute. This matches the order used by the gpu_cache
    list so the layer indices align with the saved KV payload.
    """
    scales: dict = {}
    if model_runner is None or not hasattr(model_runner, "model"):
        return scales

    layer_idx = 0
    for name, module in model_runner.model.named_modules():
        # Identify attention layers by the presence of _k_scale or k_scale
        has_scale = any(
            hasattr(module, a)
            for a in ("_k_scale", "k_scale")
        )
        if not has_scale:
            continue

        def _read(attr_names):
            for attr in attr_names:
                if hasattr(module, attr):
                    v = getattr(module, attr)
                    if isinstance(v, torch.Tensor):
                        try:
                            return float(v.item())
                        except Exception:
                            return 1.0
                    if isinstance(v, (int, float)):
                        return float(v)
            return 1.0

        k = _read(("_k_scale", "k_scale"))
        v = _read(("_v_scale", "v_scale"))
        q = _read(("_q_scale", "q_scale"))
        p = _read(("_prob_scale", "prob_scale"))

        scales[f"L{layer_idx}.k"]    = k
        scales[f"L{layer_idx}.v"]    = v
        scales[f"L{layer_idx}.q"]    = q
        scales[f"L{layer_idx}.prob"] = p
        layer_idx += 1

    logger.info(
        "[AMF_CODEC] captured FP8 scales for %d layers", layer_idx
    )
    return scales


def apply_fp8_scales(model_runner: Any, scales: dict) -> int:
    """Apply per-layer FP8 scales back to the model's attention layers.

    Also forces ``calculate_kv_scales = False`` on every layer so vLLM does
    not recompute scales from subsequent forward passes.

    Returns the number of layers that had scales applied.
    """
    if model_runner is None or not hasattr(model_runner, "model") or not scales:
        return 0

    applied = 0
    layer_idx = 0
    for name, module in model_runner.model.named_modules():
        if not any(hasattr(module, a) for a in ("_k_scale", "k_scale")):
            continue

        def _write(attr_names, value: float, float_attrs: tuple):
            for attr in attr_names:
                if hasattr(module, attr):
                    param = getattr(module, attr)
                    if isinstance(param, torch.Tensor):
                        try:
                            param.fill_(value)
                        except Exception:
                            pass
                        break
            for fa in float_attrs:
                if hasattr(module, fa):
                    try:
                        setattr(module, fa, float(value))
                    except Exception:
                        pass

        k = scales.get(f"L{layer_idx}.k", 1.0)
        v = scales.get(f"L{layer_idx}.v", 1.0)
        q = scales.get(f"L{layer_idx}.q", 1.0)
        p = scales.get(f"L{layer_idx}.prob", 1.0)

        _write(("_k_scale", "k_scale"), k, ("_k_scale_float",))
        _write(("_v_scale", "v_scale"), v, ("_v_scale_float",))
        _write(("_q_scale", "q_scale"), q, ("_q_scale_float",))
        _write(("_prob_scale", "prob_scale"), p, ("_prob_scale_float",))

        # Lock the flag so vLLM can't recompute scales on the next forward.
        if hasattr(module, "calculate_kv_scales"):
            try:
                setattr(module, "calculate_kv_scales", False)
            except Exception:
                pass

        applied += 1
        layer_idx += 1

    logger.info(
        "[AMF_CODEC] applied FP8 scales to %d layers (locked recalculation)",
        applied,
    )
    return applied


# ── Codec registry ───────────────────────────────────────────────────────────

_CODEC_REGISTRY: dict = {
    CODEC_NONE: RawCodec,
    CODEC_FP8:  Fp8Codec,
}


def get_codec_by_id(codec_id: int) -> Codec:
    """Return a Codec instance for the given codec_id, or RawCodec."""
    cls = _CODEC_REGISTRY.get(codec_id, RawCodec)
    return cls()


def get_codec_by_name(name: str) -> Codec:
    """Look up a codec by short name (``raw``, ``fp8``, ...)."""
    for cid, cname in CODEC_NAMES.items():
        if cname == name.lower():
            return get_codec_by_id(cid)
    return RawCodec()


def codec_uses_v1_format(codec_id: int) -> bool:
    """True if this codec should be written with the legacy v1 'AMFK' format.

    The v1 format is preserved for CODEC_NONE and CODEC_TURBOQUANT so that
    existing on-disk snapshots and the proven 166x benchmark path are never
    touched by the codec refactor.
    """
    return codec_id in V1_COMPATIBLE_CODECS


# ── v2 header helpers ────────────────────────────────────────────────────────

_DTYPE_TAG_V2 = {
    torch.float16:  0,
    torch.float32:  1,
    torch.bfloat16: 2,
}
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_TAG_V2[torch.float8_e4m3fn] = 3
if hasattr(torch, "float8_e5m2"):
    _DTYPE_TAG_V2[torch.float8_e5m2] = 4

_TAG_TO_DTYPE_V2 = {v: k for k, v in _DTYPE_TAG_V2.items()}


def dtype_tag(dtype: torch.dtype) -> int:
    return _DTYPE_TAG_V2.get(dtype, 0)


def tag_to_dtype(tag: int) -> torch.dtype:
    return _TAG_TO_DTYPE_V2.get(tag, torch.float16)


def pack_v2_header(
    *,
    codec_id: int,
    n_layers: int,
    n_tokens: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_tag_val: int,
    total_kv_bytes: int,
    payload_bytes: int,
    sidecar_bytes: int,
    model_hash: int,
    prefix_hash: int,
) -> bytes:
    return struct.pack(
        _AMF2_HEADER_FMT,
        _AMF2_MAGIC,
        _AMF2_VERSION,
        codec_id,
        n_layers,
        n_tokens,
        n_kv_heads,
        head_dim,
        dtype_tag_val,
        total_kv_bytes,
        payload_bytes,
        sidecar_bytes,
        model_hash,
        prefix_hash,
    )


def unpack_v2_header(blob: bytes) -> dict:
    if len(blob) < _AMF2_HEADER_SIZE:
        raise ValueError(f"AMF2 header truncated: {len(blob)} < {_AMF2_HEADER_SIZE}")

    fields = struct.unpack_from(_AMF2_HEADER_FMT, blob, 0)
    (magic, version, codec_id, n_layers, n_tokens, n_kv_heads, head_dim,
     dtype_tag_val, total_kv_bytes, payload_bytes, sidecar_bytes,
     model_hash, prefix_hash) = fields

    if magic != _AMF2_MAGIC:
        raise ValueError(f"AMF2 wrong magic 0x{magic:08X}")
    if version != _AMF2_VERSION:
        raise ValueError(f"AMF2 version mismatch {version}")

    return {
        "codec_id":       codec_id,
        "n_layers":       n_layers,
        "n_tokens":       n_tokens,
        "n_kv_heads":     n_kv_heads,
        "head_dim":       head_dim,
        "dtype_tag":      dtype_tag_val,
        "total_kv_bytes": total_kv_bytes,
        "payload_bytes":  payload_bytes,
        "sidecar_bytes":  sidecar_bytes,
        "model_hash":     model_hash,
        "prefix_hash":    prefix_hash,
    }


def is_v2_file(blob: bytes) -> bool:
    """Peek at the magic bytes to decide v1 vs v2 dispatch."""
    if len(blob) < 4:
        return False
    magic = struct.unpack_from("<I", blob, 0)[0]
    return magic == _AMF2_MAGIC


# Public v2 constants for consumers
AMF2_HEADER_SIZE = _AMF2_HEADER_SIZE
AMF2_MAGIC       = _AMF2_MAGIC
