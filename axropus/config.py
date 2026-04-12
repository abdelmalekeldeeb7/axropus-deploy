"""axropus.config — Configuration model for the AMF runtime.

A single ``AxropusConfig`` dataclass captures everything the server and
CLI need. Values load in this order (later sources override earlier):

    1. Built-in defaults
    2. YAML file passed via ``--config``
    3. Environment variables with the ``AXROPUS_`` prefix
    4. Command line flags

The dataclass is intentionally flat so it can round-trip through YAML
without any custom codecs.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class AxropusConfig:
    """Runtime configuration for the AMF server + router + pool."""

    # Model identity.
    model: str = "meta-llama/Llama-3.1-70B"
    tenant_id: str = "__shared__"

    # HTTP server.
    host: str = "0.0.0.0"
    port: int = 8001
    metrics_port: int = 9090
    api_key: str = ""

    # Pool shape.
    num_layers:       int = 80
    num_kv_heads:     int = 8
    head_dim:         int = 128
    bytes_per_layer:  int = 1 << 30      # 1 GB
    block_bytes:      int = 1 << 17      # 128 KB
    default_format:   str = "int4_sym_block"
    device:           str = "cuda"

    # Router policy.
    min_prefix_tokens:    int = 64
    write_policy:         str = "large_only"   # always | large_only | on_eviction | never
    large_write_threshold: int = 2048

    # LMCache.
    lmcache_enabled:  bool = False
    lmcache_backend:  str = "cpu"
    lmcache_path:     str = "/tmp/axropus_lmcache"
    lmcache_url:      str = ""

    # Kernel knobs.
    disable_cuda_build: bool = False

    # Telemetry.
    enable_prometheus:  bool = True
    log_level:          str = "INFO"

    # ── Construction helpers ───────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "AxropusConfig":
        """Build a config from ``AXROPUS_*`` environment variables."""
        cfg = cls()
        cfg.model             = _env_str("AXROPUS_MODEL",             cfg.model)
        cfg.tenant_id         = _env_str("AXROPUS_TENANT_ID",         cfg.tenant_id)
        cfg.host              = _env_str("AXROPUS_HOST",              cfg.host)
        cfg.port              = _env_int("AXROPUS_PORT",              cfg.port)
        cfg.metrics_port      = _env_int("AXROPUS_METRICS_PORT",      cfg.metrics_port)
        cfg.api_key           = _env_str("AXROPUS_API_KEY",           cfg.api_key)
        cfg.num_layers        = _env_int("AXROPUS_NUM_LAYERS",        cfg.num_layers)
        cfg.num_kv_heads      = _env_int("AXROPUS_NUM_KV_HEADS",      cfg.num_kv_heads)
        cfg.head_dim          = _env_int("AXROPUS_HEAD_DIM",          cfg.head_dim)
        cfg.bytes_per_layer   = _env_int("AXROPUS_BYTES_PER_LAYER",   cfg.bytes_per_layer)
        cfg.block_bytes       = _env_int("AXROPUS_BLOCK_BYTES",       cfg.block_bytes)
        cfg.default_format    = _env_str("AXROPUS_DEFAULT_FORMAT",    cfg.default_format)
        cfg.device            = _env_str("AXROPUS_DEVICE",            cfg.device)
        cfg.min_prefix_tokens = _env_int("AXROPUS_MIN_PREFIX_TOKENS", cfg.min_prefix_tokens)
        cfg.write_policy      = _env_str("AXROPUS_WRITE_POLICY",      cfg.write_policy)
        cfg.large_write_threshold = _env_int("AXROPUS_LARGE_WRITE_THRESHOLD", cfg.large_write_threshold)
        cfg.lmcache_enabled   = _env_bool("AXROPUS_LMCACHE_ENABLE",   cfg.lmcache_enabled)
        cfg.lmcache_backend   = _env_str("AXROPUS_LMCACHE_BACKEND",   cfg.lmcache_backend)
        cfg.lmcache_path      = _env_str("AXROPUS_LMCACHE_PATH",      cfg.lmcache_path)
        cfg.lmcache_url       = _env_str("AXROPUS_LMCACHE_URL",       cfg.lmcache_url)
        cfg.disable_cuda_build = _env_bool("AXROPUS_DISABLE_CUDA_BUILD", cfg.disable_cuda_build)
        cfg.enable_prometheus = _env_bool("AXROPUS_ENABLE_PROMETHEUS", cfg.enable_prometheus)
        cfg.log_level         = _env_str("AXROPUS_LOG_LEVEL",         cfg.log_level)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AxropusConfig":
        """Load a config from a YAML file (falls back to JSON if PyYAML missing)."""
        text = Path(path).read_text()
        data: Dict[str, Any]
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(text) or {}
        except ImportError:
            data = json.loads(text)

        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    @classmethod
    def load(
        cls,
        config_file: Optional[str | Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "AxropusConfig":
        """Compose config from (env → file → overrides)."""
        cfg = cls.from_env()
        if config_file:
            file_cfg = cls.from_yaml(config_file)
            for f in fields(cls):
                # Keep env-overrides only if they are non-default; otherwise
                # let the file supply the value.
                env_val = getattr(cfg, f.name)
                default_val = f.default if f.default is not dataclasses.MISSING else None
                if env_val == default_val:
                    setattr(cfg, f.name, getattr(file_cfg, f.name))
        if overrides:
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        d = self.to_dict()
        d.pop("api_key", None)  # never print
        return "\n".join(f"  {k}: {v}" for k, v in sorted(d.items()))


__all__ = ["AxropusConfig"]
