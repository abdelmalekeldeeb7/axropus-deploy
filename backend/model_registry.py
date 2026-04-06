"""Model registry — full catalog of supported models with metadata.

Each entry describes a model available on the Axropus Platform Hub,
including hardware requirements, AMF compatibility, licensing, and
per-token pricing information.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a single model variant."""

    id: str
    name: str
    family: str
    params: str
    context_window: int
    license: str
    source: str
    amf_tier: str
    default_quant: str
    min_vram_gb: int
    supports_openclaw: bool
    supports_nemoclaw: bool
    tags: tuple[str, ...]
    description: str
    cost_per_1k_tokens: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


# ---------------------------------------------------------------------------
# Full model catalog
# ---------------------------------------------------------------------------

MODEL_CATALOG: tuple[ModelSpec, ...] = (
    # ── Llama 3.3 family ──────────────────────────────────────────────────
    ModelSpec(
        id="llama-3.3-70b",
        name="Llama 3.3 70B Instruct",
        family="llama",
        params="70B",
        context_window=128_000,
        license="llama3.3",
        source="meta-llama/Llama-3.3-70B-Instruct",
        amf_tier="tier-1",
        default_quant="AWQ-4bit",
        min_vram_gb=40,
        supports_openclaw=True,
        supports_nemoclaw=True,
        tags=("chat", "instruct", "coding", "reasoning"),
        description=(
            "Meta's flagship 70B parameter model with 128K context. "
            "Excellent balance of capability and efficiency for production workloads."
        ),
        cost_per_1k_tokens=0.0004,
    ),
    ModelSpec(
        id="llama-3.3-8b",
        name="Llama 3.3 8B Instruct",
        family="llama",
        params="8B",
        context_window=128_000,
        license="llama3.3",
        source="meta-llama/Llama-3.3-8B-Instruct",
        amf_tier="tier-1",
        default_quant="FP16",
        min_vram_gb=16,
        supports_openclaw=True,
        supports_nemoclaw=False,
        tags=("chat", "instruct", "lightweight", "edge"),
        description=(
            "Compact 8B model ideal for latency-sensitive applications. "
            "Runs on a single consumer GPU with full AMF acceleration."
        ),
        cost_per_1k_tokens=0.0001,
    ),
    ModelSpec(
        id="llama-3.3-405b",
        name="Llama 3.3 405B Instruct",
        family="llama",
        params="405B",
        context_window=128_000,
        license="llama3.3",
        source="meta-llama/Llama-3.3-405B-Instruct",
        amf_tier="tier-2",
        default_quant="AWQ-4bit",
        min_vram_gb=320,
        supports_openclaw=True,
        supports_nemoclaw=True,
        tags=("chat", "instruct", "frontier", "reasoning", "coding"),
        description=(
            "Meta's largest open model. Frontier-class performance across all "
            "benchmarks. Requires multi-node or 8xH100 deployment."
        ),
        cost_per_1k_tokens=0.0012,
    ),
    # ── DeepSeek family ───────────────────────────────────────────────────
    ModelSpec(
        id="deepseek-r1-671b",
        name="DeepSeek R1 671B",
        family="deepseek",
        params="671B",
        context_window=128_000,
        license="deepseek",
        source="deepseek-ai/DeepSeek-R1",
        amf_tier="tier-2",
        default_quant="AWQ-4bit",
        min_vram_gb=340,
        supports_openclaw=True,
        supports_nemoclaw=True,
        tags=("reasoning", "math", "coding", "frontier", "moe"),
        description=(
            "DeepSeek's 671B MoE reasoning model with chain-of-thought. "
            "State-of-the-art on math and coding benchmarks."
        ),
        cost_per_1k_tokens=0.0014,
    ),
    ModelSpec(
        id="deepseek-v3-236b",
        name="DeepSeek V3 236B",
        family="deepseek",
        params="236B",
        context_window=128_000,
        license="deepseek",
        source="deepseek-ai/DeepSeek-V3",
        amf_tier="tier-1",
        default_quant="AWQ-4bit",
        min_vram_gb=160,
        supports_openclaw=True,
        supports_nemoclaw=True,
        tags=("chat", "instruct", "coding", "moe"),
        description=(
            "DeepSeek V3 236B MoE general-purpose model. Strong coding and "
            "instruction-following with efficient MoE architecture."
        ),
        cost_per_1k_tokens=0.0008,
    ),
    # ── Qwen 3 family ────────────────────────────────────────────────────
    ModelSpec(
        id="qwen-3-72b",
        name="Qwen 3 72B Instruct",
        family="qwen",
        params="72B",
        context_window=131_072,
        license="apache-2.0",
        source="Qwen/Qwen3-72B-Instruct",
        amf_tier="tier-1",
        default_quant="AWQ-4bit",
        min_vram_gb=40,
        supports_openclaw=True,
        supports_nemoclaw=True,
        tags=("chat", "instruct", "multilingual", "coding", "reasoning"),
        description=(
            "Alibaba's Qwen 3 72B with 131K context window. "
            "Excellent multilingual and coding performance with Apache 2.0 license."
        ),
        cost_per_1k_tokens=0.0004,
    ),
    ModelSpec(
        id="qwen-3-7b",
        name="Qwen 3 7B Instruct",
        family="qwen",
        params="7B",
        context_window=131_072,
        license="apache-2.0",
        source="Qwen/Qwen3-7B-Instruct",
        amf_tier="tier-1",
        default_quant="FP16",
        min_vram_gb=16,
        supports_openclaw=True,
        supports_nemoclaw=False,
        tags=("chat", "instruct", "multilingual", "lightweight", "edge"),
        description=(
            "Compact 7B Qwen 3 model suitable for edge deployment. "
            "Full AMF support with minimal hardware requirements."
        ),
        cost_per_1k_tokens=0.0001,
    ),
    # ── Mistral family ────────────────────────────────────────────────────
    ModelSpec(
        id="mistral-8x22b",
        name="Mixtral 8x22B Instruct",
        family="mistral",
        params="8x22B",
        context_window=65_536,
        license="apache-2.0",
        source="mistralai/Mixtral-8x22B-Instruct-v0.1",
        amf_tier="tier-1",
        default_quant="AWQ-4bit",
        min_vram_gb=80,
        supports_openclaw=True,
        supports_nemoclaw=True,
        tags=("chat", "instruct", "moe", "coding"),
        description=(
            "Mistral's flagship MoE model with 8 experts of 22B each. "
            "Strong performance with efficient sparse activation."
        ),
        cost_per_1k_tokens=0.0005,
    ),
    ModelSpec(
        id="mistral-7b",
        name="Mistral 7B Instruct v0.3",
        family="mistral",
        params="7B",
        context_window=32_768,
        license="apache-2.0",
        source="mistralai/Mistral-7B-Instruct-v0.3",
        amf_tier="tier-1",
        default_quant="FP16",
        min_vram_gb=16,
        supports_openclaw=True,
        supports_nemoclaw=False,
        tags=("chat", "instruct", "lightweight", "edge"),
        description=(
            "Compact and efficient 7B model from Mistral. "
            "Ideal for cost-sensitive deployments with full AMF acceleration."
        ),
        cost_per_1k_tokens=0.0001,
    ),
    # ── NVIDIA Nemotron ───────────────────────────────────────────────────
    ModelSpec(
        id="nemotron-70b",
        name="Nemotron 70B Instruct",
        family="nemotron",
        params="70B",
        context_window=128_000,
        license="nvidia-open",
        source="nvidia/Nemotron-4-340B-Instruct",
        amf_tier="tier-1",
        default_quant="AWQ-4bit",
        min_vram_gb=40,
        supports_openclaw=True,
        supports_nemoclaw=True,
        tags=("chat", "instruct", "reasoning", "enterprise"),
        description=(
            "NVIDIA's Nemotron 70B tuned for helpfulness and instruction following. "
            "Strong enterprise performance with NemoClaw agent support."
        ),
        cost_per_1k_tokens=0.0004,
    ),
    # ── Google Gemma family ───────────────────────────────────────────────
    ModelSpec(
        id="gemma-27b",
        name="Gemma 2 27B Instruct",
        family="gemma",
        params="27B",
        context_window=8_192,
        license="gemma",
        source="google/gemma-2-27b-it",
        amf_tier="tier-1",
        default_quant="AWQ-4bit",
        min_vram_gb=24,
        supports_openclaw=True,
        supports_nemoclaw=False,
        tags=("chat", "instruct", "research"),
        description=(
            "Google's Gemma 2 27B instruction-tuned model. "
            "Excellent quality-to-size ratio for mid-range deployments."
        ),
        cost_per_1k_tokens=0.0002,
    ),
    ModelSpec(
        id="gemma-9b",
        name="Gemma 2 9B Instruct",
        family="gemma",
        params="9B",
        context_window=8_192,
        license="gemma",
        source="google/gemma-2-9b-it",
        amf_tier="tier-1",
        default_quant="FP16",
        min_vram_gb=18,
        supports_openclaw=True,
        supports_nemoclaw=False,
        tags=("chat", "instruct", "lightweight", "research"),
        description=(
            "Compact Gemma 2 9B model suitable for edge and research use. "
            "Runs comfortably on a single 24 GB GPU."
        ),
        cost_per_1k_tokens=0.0001,
    ),
)

# Fast lookup index: model_id -> ModelSpec
_INDEX: dict[str, ModelSpec] = {m.id: m for m in MODEL_CATALOG}


def get_model(model_id: str) -> Optional[ModelSpec]:
    """Return a model spec by ID, or ``None`` if not found."""
    return _INDEX.get(model_id)


def get_model_or_raise(model_id: str) -> ModelSpec:
    """Return a model spec by ID. Raises ``KeyError`` on miss."""
    spec = _INDEX.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown model: {model_id!r}")
    return spec


def list_models(
    *,
    family: Optional[str] = None,
    supports_openclaw: Optional[bool] = None,
    supports_nemoclaw: Optional[bool] = None,
    tag: Optional[str] = None,
) -> list[ModelSpec]:
    """Return models matching the given filters (all optional)."""
    results: list[ModelSpec] = []
    for m in MODEL_CATALOG:
        if family is not None and m.family != family:
            continue
        if supports_openclaw is not None and m.supports_openclaw != supports_openclaw:
            continue
        if supports_nemoclaw is not None and m.supports_nemoclaw != supports_nemoclaw:
            continue
        if tag is not None and tag not in m.tags:
            continue
        results.append(m)
    return results


def list_families() -> list[str]:
    """Return sorted unique model families."""
    return sorted({m.family for m in MODEL_CATALOG})
