from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class EngineConfig:
    accel_enabled: bool = False
    cuda_device: int = 0
    cuda_dtype: str = "fp16"
    kv_layout_version: str = "v1"


@dataclass(frozen=True)
class SpecConfig:
    enabled: bool = False
    k: int = 6
    min_accept: float = 0.55
    disable_after_n: int = 50


@dataclass
class EngineMetrics:
    perf: Dict[str, Any] = field(default_factory=dict)
    amf: Dict[str, Any] = field(default_factory=dict)
    mf: Dict[str, Any] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=dict)
    cp: Dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "perf": self.perf,
            "amf": self.amf,
            "mf": self.mf,
            "spec": self.spec,
            "health": self.health,
            "cp": self.cp,
            "errors": self.errors,
        }


@dataclass
class EngineRunResult:
    exit_code: int
    total_ms: float
    output_text: str = ""
    metrics: EngineMetrics = field(default_factory=EngineMetrics)
    engine_events_path: str | None = None
    engine_errors: list[str] = field(default_factory=list)

