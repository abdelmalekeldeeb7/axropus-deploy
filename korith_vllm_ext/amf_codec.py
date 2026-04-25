"""Compatibility shim — imports moved to korith_vllm_ext.codecs.amf_codec."""

from .codecs.amf_codec import *  # noqa: F401,F403
from .codecs.amf_codec import (
    FP8E4M3Codec,
    FP8E5M2Codec,
    FP8ScaleSidecar,
    INT2PerChannelCodec,
    INT4PerBlockCodec,
    INT4PerChannelCodec,
    apply_fp8_scales,
)
from .codecs.base import get_codec  # noqa: F401

# Aliases for old numeric codec IDs.
CODEC_NONE = 0
CODEC_FP8 = 1
CODEC_TURBOQUANT = 2
CODEC_INT4 = 3
