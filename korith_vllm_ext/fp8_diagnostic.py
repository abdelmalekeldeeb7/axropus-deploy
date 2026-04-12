"""Compatibility shim — FP8 diagnostic now uses codecs.amf_codec scale tools."""

from .codecs.amf_codec import FP8ScaleSidecar, apply_fp8_scales  # noqa: F401
