"""Codec package — KV cache compression formats for AMF.

Every codec here implements the ``Codec`` interface from ``base.py`` and
registers itself in the global codec registry. The tiered router and the
compressed VRAM pool look up codecs by format name.
"""

from __future__ import annotations

from .base import (
    ALL_FORMATS,
    FMT_BF16,
    FMT_FP8_E4M3,
    FMT_FP8_E5M2,
    FMT_FP16,
    FMT_INT2_SYM,
    FMT_INT4_BLOCK,
    FMT_INT4_SYM,
    FMT_NVFP4,
    FMT_TURBOQUANT,
    Codec,
    CompressedKV,
    get_codec,
    list_codecs,
    register_codec,
)
from .amf_codec import (
    FP8E4M3Codec,
    FP8E5M2Codec,
    FP8ScaleSidecar,
    INT2PerChannelCodec,
    INT4PerBlockCodec,
    INT4PerChannelCodec,
    apply_fp8_scales,
)
from .turboquant_codec import TurboQuantCodec
from .nvfp4_codec import NVFP4Codec


def select_codec(
    hw_sm: int,
    accuracy_budget: float = 0.005,
    density_budget: float = 0.25,
) -> str:
    """Pick the best codec for the current hardware and budgets.

    Args:
        hw_sm: CUDA SM version (e.g. 80, 90, 100).
        accuracy_budget: max tolerated perplexity degradation (0.005 = 0.5%).
        density_budget: target compression ratio (0.25 = 4x smaller).

    Returns:
        Format string registered with ``register_codec``.
    """
    if hw_sm >= 100:  # Blackwell
        if density_budget <= 0.20 and accuracy_budget >= 0.005:
            return FMT_TURBOQUANT
        return FMT_NVFP4
    if hw_sm >= 90:  # Hopper
        if density_budget <= 0.20:
            return FMT_TURBOQUANT
        if density_budget <= 0.25:
            return FMT_INT4_BLOCK
        return FMT_FP8_E4M3
    if hw_sm >= 80:  # Ampere
        return FMT_INT4_BLOCK
    return FMT_FP16


__all__ = [
    "ALL_FORMATS",
    "Codec",
    "CompressedKV",
    "FMT_BF16",
    "FMT_FP8_E4M3",
    "FMT_FP8_E5M2",
    "FMT_FP16",
    "FMT_INT2_SYM",
    "FMT_INT4_BLOCK",
    "FMT_INT4_SYM",
    "FMT_NVFP4",
    "FMT_TURBOQUANT",
    "FP8E4M3Codec",
    "FP8E5M2Codec",
    "FP8ScaleSidecar",
    "INT2PerChannelCodec",
    "INT4PerBlockCodec",
    "INT4PerChannelCodec",
    "NVFP4Codec",
    "TurboQuantCodec",
    "apply_fp8_scales",
    "get_codec",
    "list_codecs",
    "register_codec",
    "select_codec",
]
