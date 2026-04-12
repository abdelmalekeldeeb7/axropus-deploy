"""codecs/base.py — Common interface for all KV cache compression codecs.

A Codec takes a raw KV tensor (FP16/BF16) and produces a CompressedKV
representation containing the quantized payload plus the scale metadata
needed to decompress. Every codec in AMF must implement this interface
so that the compressed pool and decode kernels can treat them uniformly.

Design notes:
  - The compressed format is codec-specific and opaque to the pool.
  - Scales are stored next to the payload but addressed separately so
    decode kernels can load them via a small side table.
  - decompress_to() accepts a target format. This allows the two-tier
    precision model from §2.1 of the design doc: a codec may store INT4
    and promote directly to FP8 on restore without a full dequant pass.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


# ── Format identifiers ────────────────────────────────────────────────────────
# These strings are the canonical names used throughout the tiered router,
# kernel dispatch table, and pool manager. Keep them in sync with
# kernels/dispatch.py.

FMT_FP16        = "fp16"
FMT_BF16        = "bf16"
FMT_FP8_E4M3    = "fp8_e4m3"
FMT_FP8_E5M2    = "fp8_e5m2"
FMT_INT4_SYM    = "int4_sym"          # symmetric per-channel
FMT_INT4_BLOCK  = "int4_sym_block"    # symmetric per-block
FMT_INT2_SYM    = "int2_sym"
FMT_NVFP4       = "nvfp4"
FMT_TURBOQUANT  = "turboquant"

ALL_FORMATS = (
    FMT_FP16,
    FMT_BF16,
    FMT_FP8_E4M3,
    FMT_FP8_E5M2,
    FMT_INT4_SYM,
    FMT_INT4_BLOCK,
    FMT_INT2_SYM,
    FMT_NVFP4,
    FMT_TURBOQUANT,
)


@dataclass
class CompressedKV:
    """Codec-agnostic container for a compressed KV blob.

    The ``data`` tensor holds the packed payload. For byte-packed formats
    (INT4, INT2, NVFP4) the dtype is ``torch.uint8``. For FP8 formats it is
    ``torch.float8_e4m3fn`` / ``torch.float8_e5m2`` when available, otherwise
    ``torch.uint8`` carrying the raw bytes.

    ``scales`` is a small float tensor indexed by block / channel / group.
    ``meta`` is a codec-specific dict that can carry extra state such as
    micro-block scales, projection matrices, etc.
    """

    format: str
    data: torch.Tensor
    scales: torch.Tensor
    tensor_scale: float = 1.0
    shape: tuple = ()
    dtype_original: torch.dtype = torch.float16
    meta: Dict[str, Any] = field(default_factory=dict)

    def nbytes(self) -> int:
        total = self.data.numel() * self.data.element_size()
        if self.scales is not None:
            total += self.scales.numel() * self.scales.element_size()
        return total

    def original_bytes(self) -> int:
        numel = 1
        for d in self.shape:
            numel *= d
        return numel * torch.tensor([], dtype=self.dtype_original).element_size()


class Codec(abc.ABC):
    """Abstract base class for all KV cache codecs."""

    # Canonical format name (e.g. "fp8_e4m3"). Subclasses set this.
    FORMAT: str = ""

    @abc.abstractmethod
    def compress(self, raw_kv: torch.Tensor) -> CompressedKV:
        """Compress a raw KV tensor.

        Args:
            raw_kv: Shape [..., num_tokens, num_heads, head_dim] in FP16/BF16.

        Returns:
            CompressedKV blob.
        """

    @abc.abstractmethod
    def decompress_to(
        self,
        compressed: CompressedKV,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Decompress back to a plain tensor of ``target_dtype``.

        Used on promotion paths and when the active decode kernel does
        not support the storage format natively.
        """

    @abc.abstractmethod
    def memory_ratio(self) -> float:
        """Compressed size / FP16 size. Lower is better."""

    # ── Convenience helpers ────────────────────────────────────────────────

    def compressed_bytes(self, numel: int) -> int:
        """Return the expected payload size in bytes for ``numel`` FP16 values."""
        fp16_bytes = numel * 2
        return int(round(fp16_bytes * self.memory_ratio()))

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in (torch.float16, torch.bfloat16, torch.float32)

    def __repr__(self) -> str:
        return f"<Codec format={self.FORMAT} ratio={self.memory_ratio():.3f}>"


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Codec] = {}


def register_codec(codec: Codec) -> None:
    """Register a codec by its FORMAT name."""
    if not codec.FORMAT:
        raise ValueError(f"Codec {codec!r} has no FORMAT string")
    _REGISTRY[codec.FORMAT] = codec


def get_codec(format: str) -> Codec:
    """Look up a registered codec."""
    if format not in _REGISTRY:
        raise KeyError(
            f"Unknown codec format {format!r}. Registered: {list(_REGISTRY)}"
        )
    return _REGISTRY[format]


def list_codecs() -> List[str]:
    return sorted(_REGISTRY.keys())
