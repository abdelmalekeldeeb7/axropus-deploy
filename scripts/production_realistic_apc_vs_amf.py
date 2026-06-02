#!/usr/bin/env python3
"""Production-realistic synthetic APC vs AMF benchmark — with disruption event model.

This benchmark compares vLLM Automatic Prefix Caching (APC) against AMF
persistent KV materialization on the same synthetic request stream and hardware.

Traffic model
-------------
  * Reusable prefixes follow a Zipfian/power-law distribution.
  * A configurable fraction of prefixes are unique one-offs.
  * Prefix lengths are sampled from short/medium/long buckets.
  * Every request appends a short unique suffix.

Disruption events (new)
-----------------------
Three cache-disruption event types are injected at realistic frequencies
configured in the YAML.  They are reported as a SEPARATE REGIME from
steady-state, never blended into a single inflated headline number.  A
blended view is computed from realistic frequency weights and clearly
labelled as such.

  restart          — engine restart: APC cache → zero; AMF restores from
                     its persistent VRAM tier.
  eviction_to_zero — tenant rotation: a flood of new-tenant tokens fills
                     the KV budget, evicting original hot-prefix blocks via
                     LRU.  APC cold-prefills; AMF restores.
  cold_routing     — request routed to a worker that never saw the prefix:
                     APC empty; AMF pulls from shared persistent tier.

Correctness gate
----------------
For a sampled subset of AMF-restored requests (steady-state AND
post-disruption), the benchmark computes a cold greedy output and compares
token IDs with the AMF-restored output.  Mismatches are written to the JSONL
event log and make the run fail by default.

Pre-flight verification (CRITICAL)
-----------------------------------
Before the main benchmark, the AMF arm runs a prefill-elimination probe:
  1. Cold-prefill a medium prefix and record timing.
  2. AMF save.
  3. Reset the APC cache (simulate restart/eviction).
  4. AMF restore + re-register.
  5. Warm prefill and record timing.
  6. Assert: warm_ms < cold_ms / speedup_threshold  (default 1.5×).
  7. Assert: recomputed_tokens ≤ max_recomputed_fraction × prefix_tokens.
If the probe fails, the run is aborted.  A benchmark on a path that silently
recomputes proves nothing.

All output is labelled synthetic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

from korith_vllm_ext.korith_vllm_server import _call_engine_core


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefixSpec:
    prefix_id: str
    rank: int
    target_tokens: int
    bucket: str
    hot: bool
    unique: bool


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    simulated_time_s: float
    prefix_id: str
    suffix_id: str
    hot: bool
    unique: bool


@dataclass(frozen=True)
class DisruptionEvent:
    """A cache-disruption event injected at a specific request index."""
    event_id: int
    event_type: str          # "restart" | "eviction_to_zero" | "cold_routing"
    request_index: int       # event fires BEFORE this request is processed
    recovery_window: int     # number of subsequent requests tagged post-disruption


@dataclass
class RequestTag:
    """Per-request regime label, assigned during traffic scheduling."""
    regime: str = "steady_state"
    disruption_event_id: int = -1
    disruption_type: str = ""
    distance_from_event: int = -1


@dataclass
class RequestMetric:
    request_id: int
    prefix_id: str
    arm: str
    budget_fraction: float
    unique: bool
    hot: bool
    prefix_tokens: int
    suffix_tokens: int
    source: str
    any_hit: bool
    full_hit: bool
    reused_tokens_before: int
    recomputed_prefix_tokens: int
    latency_ms: float
    generate_ms: float
    restore_ms: float = 0.0
    register_ms: float = 0.0
    save_ms: float = 0.0
    output_match: bool | None = None
    # Regime fields (new)
    regime: str = "steady_state"
    disruption_event_id: int = -1
    disruption_type: str = ""
    distance_from_event: int = -1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is not installed; use a JSON config or install pyyaml")
    return yaml.safe_load(text)


def _arm_kv_budget_mb(cache_cfg: dict[str, Any], arm: str, default_budget_mb: int) -> int:
    """Return the vLLM KV budget for an arm.

    The default benchmark budget is derived from working-set fraction and is
    applied to every arm.  For same-total-GPU-memory comparisons, configs can
    pin each arm's vLLM KV budget explicitly while AMF uses its own VRAM tier.
    """
    arm_budgets = cache_cfg.get("arm_kv_cache_memory_mb", {})
    if not isinstance(arm_budgets, dict) or arm not in arm_budgets:
        return default_budget_mb
    budget = int(arm_budgets[arm])
    if budget <= 0:
        raise ValueError(f"cache.arm_kv_cache_memory_mb.{arm} must be > 0")
    return budget


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _weighted_choice(rng: random.Random, items: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(float(item["weight"]) for item in items)
    pick = rng.random() * total
    acc = 0.0
    for item in items:
        acc += float(item["weight"])
        if pick <= acc:
            return item
    return items[-1]


def _zipf_index(rng: random.Random, n: int, alpha: float) -> int:
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for i, weight in enumerate(weights):
        acc += weight
        if pick <= acc:
            return i
    return n - 1


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[idx]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _cost_gpu_seconds(ms_values: list[float]) -> float:
    return sum(ms_values) / 1000.0


def _sha_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _fit_text_to_tokens(tokenizer: Any, seed: str, target_tokens: int) -> tuple[str, list[int]]:
    text = seed
    while len(tokenizer.encode(text)) < target_tokens + 64:
        text += seed
    ids = tokenizer.encode(text)[:target_tokens]
    text = tokenizer.decode(ids)
    return text, tokenizer.encode(text)


def _make_suffix(tokenizer: Any, request_id: int, target_tokens: int) -> tuple[str, list[int]]:
    seed = (
        f"\n\nUser request {request_id:06d}: answer this unique query, include only "
        f"the next action id {request_id:06d}. "
    )
    return _fit_text_to_tokens(tokenizer, seed, target_tokens)


# ---------------------------------------------------------------------------
# Traffic generation
# ---------------------------------------------------------------------------

def _generate_traffic(cfg: dict[str, Any]) -> tuple[list[PrefixSpec], list[RequestSpec]]:
    rng = random.Random(int(cfg.get("seed", 1337)))
    traffic = cfg["traffic"]
    n_requests = int(cfg.get("requests", 2000))
    reusable_count = int(traffic.get("reusable_prefixes", 256))
    unique_fraction = float(traffic.get("unique_fraction", 0.20))
    zipf_alpha = float(traffic.get("zipf_alpha", 1.15))
    hot_cutoff = int(traffic.get("hot_rank_cutoff", max(1, reusable_count // 16)))
    length_mix = list(traffic["prefix_length_mix"])

    prefixes: list[PrefixSpec] = []
    for rank in range(reusable_count):
        bucket = _weighted_choice(rng, length_mix)
        target = rng.randint(int(bucket["min_tokens"]), int(bucket["max_tokens"]))
        prefixes.append(
            PrefixSpec(
                prefix_id=f"p{rank:05d}",
                rank=rank + 1,
                target_tokens=target,
                bucket=str(bucket["name"]),
                hot=rank < hot_cutoff,
                unique=False,
            )
        )

    requests: list[RequestSpec] = []
    t = 0.0
    for request_id in range(n_requests):
        t += rng.expovariate(1.0)
        is_unique = rng.random() < unique_fraction
        if is_unique:
            bucket = _weighted_choice(rng, length_mix)
            target = rng.randint(int(bucket["min_tokens"]), int(bucket["max_tokens"]))
            pid = f"u{request_id:06d}"
            prefixes.append(
                PrefixSpec(
                    prefix_id=pid,
                    rank=reusable_count + request_id + 1,
                    target_tokens=target,
                    bucket=str(bucket["name"]),
                    hot=False,
                    unique=True,
                )
            )
            hot = False
        else:
            idx = _zipf_index(rng, reusable_count, zipf_alpha)
            pid = prefixes[idx].prefix_id
            hot = prefixes[idx].hot
        requests.append(
            RequestSpec(
                request_id=request_id,
                simulated_time_s=t,
                prefix_id=pid,
                suffix_id=f"s{request_id:06d}",
                hot=hot,
                unique=is_unique,
            )
        )
    return prefixes, requests


# ---------------------------------------------------------------------------
# Disruption scheduling
# ---------------------------------------------------------------------------

def _schedule_disruptions(
    cfg: dict[str, Any],
    n_requests: int,
) -> list[DisruptionEvent]:
    """Build the list of disruption events from config.

    Events are placed at deterministic request indices.  Frequencies are
    derived from realistic defaults that a skeptic can tune down.
    """
    disp_cfg = cfg.get("disruptions", {})
    seed = int(cfg.get("seed", 1337))
    events: list[DisruptionEvent] = []
    event_id = 0

    # --- Restart events (rare) ---
    restart_cfg = disp_cfg.get("restart", {})
    if restart_cfg.get("enabled", True):
        interval = int(restart_cfg.get("interval_requests", 500))
        recovery = int(restart_cfg.get("recovery_window_requests", 20))
        for idx in range(interval, n_requests, interval):
            events.append(DisruptionEvent(event_id, "restart", idx, recovery))
            event_id += 1

    # --- Eviction-to-zero / tenant rotation (occasional) ---
    eviction_cfg = disp_cfg.get("eviction_to_zero", {})
    if eviction_cfg.get("enabled", True):
        interval = int(eviction_cfg.get("interval_requests", 100))
        recovery = int(eviction_cfg.get("recovery_window_requests", 10))
        for idx in range(interval, n_requests, interval):
            # Skip if a restart event is already placed within ±3 requests
            # (restart subsumes eviction for measurement purposes)
            if not any(
                e.event_type == "restart" and abs(e.request_index - idx) <= 3
                for e in events
            ):
                events.append(DisruptionEvent(event_id, "eviction_to_zero", idx, recovery))
                event_id += 1

    # --- Cold-routing events (per-request probability based on fleet_size) ---
    cold_cfg = disp_cfg.get("cold_routing", {})
    if cold_cfg.get("enabled", True):
        fleet_size = int(cold_cfg.get("fleet_size", 8))
        recovery = int(cold_cfg.get("recovery_window_requests", 1))
        rng = random.Random(seed + 42)
        for idx in range(n_requests):
            if rng.random() < 1.0 / max(fleet_size, 1):
                # Skip if another event already fires at this index
                if not any(e.request_index == idx for e in events):
                    events.append(DisruptionEvent(event_id, "cold_routing", idx, recovery))
                    event_id += 1

    events.sort(key=lambda e: (e.request_index, e.event_id))
    return events


def _tag_requests_with_regime(
    n_requests: int,
    events: list[DisruptionEvent],
) -> dict[int, RequestTag]:
    """Assign a regime label to each request index.

    A request may fall inside multiple recovery windows if events overlap.
    The last (latest) event's tag wins, which corresponds to the most recent
    disruption.
    """
    tags: dict[int, RequestTag] = {i: RequestTag("steady_state") for i in range(n_requests)}
    regime_name = {
        "restart": "post_restart",
        "eviction_to_zero": "post_eviction",
        "cold_routing": "post_cold_routing",
    }
    for event in events:
        start = event.request_index
        end = min(n_requests, start + event.recovery_window)
        rname = regime_name[event.event_type]
        for j in range(start, end):
            tags[j] = RequestTag(
                regime=rname,
                disruption_event_id=event.event_id,
                disruption_type=event.event_type,
                distance_from_event=j - start,
            )
    return tags


# ---------------------------------------------------------------------------
# Model / engine helpers
# ---------------------------------------------------------------------------

def _estimate_kv_bytes_per_token(model: str, kv_cache_dtype: str) -> int:
    try:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model, trust_remote_code=True)
        layers = int(getattr(hf, "num_hidden_layers"))
        n_heads = int(getattr(hf, "num_attention_heads"))
        n_kv_heads = int(getattr(hf, "num_key_value_heads", n_heads))
        hidden = int(getattr(hf, "hidden_size"))
        head_dim = int(getattr(hf, "head_dim", hidden // n_heads))
        dtype_bytes = 1 if str(kv_cache_dtype).lower().startswith("fp8") else 2
        return layers * 2 * n_kv_heads * head_dim * dtype_bytes
    except Exception:
        return 28 * 2 * 2 * 128 * 2  # conservative fallback


def _coverage(llm: Any, token_ids: list[int]) -> dict[str, Any]:
    try:
        info = _call_engine_core(llm, "amf_get_cached_block_ids", list(token_ids))
    except Exception:
        info = {}
    info = info if isinstance(info, dict) else {}
    found = int(info.get("n_found", 0) or 0)
    full = int(info.get("n_full_blocks", 0) or 0)
    block_size = int(info.get("block_size", 16) or 16)
    return {
        "found": found,
        "full": full,
        "block_size": block_size,
        "tokens": found * block_size,
        "any_hit": found > 0,
        "full_hit": full > 0 and found == full,
        "block_ids": info.get("block_ids", []) or [],
    }


def _reset(llm: Any) -> None:
    llm.llm_engine.reset_prefix_cache(reset_running_requests=True)


def _generate(llm: LLM, prompt: str, sampling: SamplingParams) -> tuple[float, list[int], int]:
    """Returns (wall_clock_ms, output_token_ids, num_cached_tokens).

    num_cached_tokens is vLLM's actual report of how many prompt tokens were
    served from the KV cache (prefix cache hit), not a formula-based estimate.
    A cold request returns ~0; a warm/restored request returns ~len(prefix_ids).
    """
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    ms = (time.perf_counter() - t0) * 1000.0
    token_ids: list[int] = []
    num_cached = 0
    if outputs:
        if outputs[0].outputs:
            token_ids = list(outputs[0].outputs[0].token_ids)
        num_cached = int(outputs[0].num_cached_tokens or 0)
    return ms, token_ids, num_cached


def _new_llm(cfg: dict[str, Any], kv_cache_memory_mb: int, with_amf: bool) -> LLM:
    runtime = cfg["runtime"]
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "trust_remote_code": True,
        "enforce_eager": bool(runtime.get("enforce_eager", True)),
        "async_scheduling": False,
        "gpu_memory_utilization": float(runtime.get("gpu_memory_utilization", 0.90)),
        "kv_cache_memory_bytes": kv_cache_memory_mb * 1024 * 1024,
        "max_model_len": int(runtime.get("max_model_len", 131072)),
        "kv_cache_dtype": runtime.get("kv_cache_dtype", "auto"),
    }
    if with_amf:
        kwargs["worker_extension_cls"] = "korith_vllm_ext.amf_worker_ext.AmfWorkerExtension"
    return LLM(**kwargs)


def _new_lmcache_llm(
    cfg: dict[str, Any],
    kv_cache_memory_mb: int,
    *,
    with_amf: bool = False,
) -> LLM:
    runtime = cfg["runtime"]
    lmcache_cfg = cfg.get("lmcache", {})
    os.environ.setdefault("LMCACHE_TRACK_USAGE", "false")
    os.environ.setdefault("LMCACHE_LOCAL_CPU", "true")
    os.environ.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", str(lmcache_cfg.get("local_cpu_gb", 2)))
    os.environ.setdefault("LMCACHE_CHUNK_SIZE", str(lmcache_cfg.get("chunk_size", 256)))
    os.environ.setdefault(
        "LMCACHE_SAVE_UNFULL_CHUNK",
        "true" if bool(lmcache_cfg.get("save_unfull_chunk", True)) else "false",
    )
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "trust_remote_code": True,
        "enforce_eager": bool(runtime.get("enforce_eager", True)),
        "async_scheduling": False,
        "gpu_memory_utilization": float(runtime.get("gpu_memory_utilization", 0.90)),
        "kv_cache_memory_bytes": kv_cache_memory_mb * 1024 * 1024,
        "max_model_len": int(runtime.get("max_model_len", 131072)),
        "kv_cache_dtype": runtime.get("kv_cache_dtype", "auto"),
        "enable_prefix_caching": True,
        "kv_transfer_config": KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role=str(lmcache_cfg.get("role", "kv_both")),
            kv_connector_extra_config={
                "discard_partial_chunks": bool(lmcache_cfg.get("discard_partial_chunks", False)),
                "skip_last_n_tokens": int(lmcache_cfg.get("skip_last_n_tokens", 0)),
            },
        ),
    }
    if with_amf:
        kwargs["worker_extension_cls"] = "korith_vllm_ext.amf_worker_ext.AmfWorkerExtension"
    return LLM(**kwargs)


class PromptStore:
    """Lazy deterministic synthetic prompt materialization."""

    def __init__(self, tokenizer: Any, prefix_by_id: dict[str, PrefixSpec], suffix_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.prefix_by_id = prefix_by_id
        self.suffix_tokens = suffix_tokens
        self._prefix_text: dict[str, str] = {}
        self._prefix_ids: dict[str, list[int]] = {}
        self._suffix_text: dict[str, str] = {}
        self._suffix_ids: dict[str, list[int]] = {}

    def prefix(self, prefix_id: str) -> tuple[str, list[int]]:
        if prefix_id not in self._prefix_text:
            spec = self.prefix_by_id[prefix_id]
            seed = (
                f"Prefix {spec.prefix_id} rank={spec.rank} bucket={spec.bucket}. "
                "Synthetic shared context with system instructions, tool schemas, "
                "retrieved documents, memory, logs, and policy text. "
            )
            text, ids = _fit_text_to_tokens(self.tokenizer, seed, spec.target_tokens)
            self._prefix_text[prefix_id] = text
            self._prefix_ids[prefix_id] = ids
        return self._prefix_text[prefix_id], self._prefix_ids[prefix_id]

    def suffix(self, request_id: int, suffix_id: str) -> tuple[str, list[int]]:
        if suffix_id not in self._suffix_text:
            text, ids = _make_suffix(self.tokenizer, request_id, self.suffix_tokens)
            self._suffix_text[suffix_id] = text
            self._suffix_ids[suffix_id] = ids
        return self._suffix_text[suffix_id], self._suffix_ids[suffix_id]


# ---------------------------------------------------------------------------
# Pre-flight: prefill elimination verification
# ---------------------------------------------------------------------------

def _verify_prefill_elimination(
    llm: Any,
    tokenizer: Any,
    sampling: SamplingParams,
    prompt_store: PromptStore,
    prefixes: list[PrefixSpec],
    amf_path: Path,
    cfg: dict[str, Any],
    skip: bool = False,
) -> bool:
    """Verify that AMF actually eliminates prefill (not just re-registers to the same cost).

    Returns True if verification passes (or is skipped).  Returns False and
    prints a VERIFY_FAIL message if prefill elimination is not confirmed.

    The check:
      1. Cold-prefill a medium prefix → measure cold_ms.
      2. AMF save the materialized KV state.
      3. _reset(llm) to evict from APC.
      4. AMF restore + re-register.
      5. Warm prefill the same prompt → measure warm_ms.
      6. Assert warm_ms < cold_ms / speedup_threshold.
      7. Assert recomputed_tokens ≤ max_recomputed_fraction × prefix_tokens.
    """
    verify_cfg = cfg.get("verify", {})
    if skip or not verify_cfg.get("enabled", True):
        print("[VERIFY_SKIP] prefill-elimination verification disabled", flush=True)
        return True

    fail_on_fail = bool(verify_cfg.get("fail_on_verify_fail", True))
    speedup_threshold = float(verify_cfg.get("speedup_threshold", 1.5))
    max_recomputed_frac = float(verify_cfg.get("max_recomputed_fraction", 0.10))

    # Pick a medium-bucket reusable prefix for the probe.
    candidates = [p for p in prefixes if p.bucket == "medium" and not p.unique]
    if not candidates:
        candidates = [p for p in prefixes if not p.unique]
    if not candidates:
        print("[VERIFY_SKIP] no reusable prefix available for verification probe", flush=True)
        return True

    probe = candidates[0]
    ptext, pids = prompt_store.prefix(probe.prefix_id)
    stext, _ = prompt_store.suffix(999_999, "s_verify_probe")
    prompt = ptext + stext

    print(
        f"[VERIFY_START] probe prefix_id={probe.prefix_id} "
        f"prefix_tokens={len(pids)} bucket={probe.bucket}",
        flush=True,
    )

    # 1. Cold prefill — vLLM must report num_cached_tokens ≈ 0.
    _reset(llm)
    cold_ms, cold_tokens, cold_cached = _generate(llm, prompt, sampling)
    total_prompt_tokens = len(pids) + len(tokenizer.encode(stext))
    cold_prefilled = total_prompt_tokens - cold_cached
    print(
        f"[VERIFY] step=cold cold_ms={cold_ms:.1f} "
        f"prompt_tokens={total_prompt_tokens} cached={cold_cached} prefilled={cold_prefilled}",
        flush=True,
    )

    # 2. AMF save.
    cov = _coverage(llm, pids)
    save_results = llm.collective_rpc(
        "amf_save_kv",
        timeout=600,
        args=(str(amf_path), list(pids), 0, "__shared__"),
        kwargs={"physical_block_ids": cov["block_ids"]} if cov["block_ids"] else None,
    )
    saved = bool((save_results[0] if save_results else {}).get("saved", False))
    if not saved:
        msg = "[VERIFY_FAIL] AMF save returned not-saved; cannot confirm restore path. STOPPING."
        print(msg, flush=True)
        if fail_on_fail:
            return False
        print("[VERIFY_WARN] continuing despite save failure (fail_on_verify_fail=false)", flush=True)
        return True

    # 3. Reset APC.
    _reset(llm)

    # 4. AMF restore + re-register.
    t0 = time.perf_counter()
    restored = llm.collective_rpc(
        "amf_restore_kv",
        timeout=600,
        args=(str(amf_path), list(pids), 0, "__shared__"),
    )
    restore_ms = (time.perf_counter() - t0) * 1000.0
    restored_tokens = int(restored[0] if restored else 0)

    block_size = int(cov.get("block_size", 16) or 16)
    restore_block_ids = list(range(max(len(pids) // block_size, 1)))
    _call_engine_core(llm, "amf_register_prefix", list(pids), restore_block_ids)

    # 5. Warm prefill — vLLM must report num_cached_tokens ≈ len(pids).
    warm_ms, warm_tokens, warm_cached = _generate(llm, prompt, sampling)
    warm_prefilled = total_prompt_tokens - warm_cached
    speedup = cold_ms / max(warm_ms, 1.0)

    print(
        f"[VERIFY] step=warm restore_ms={restore_ms:.1f} warm_ms={warm_ms:.1f} "
        f"speedup={speedup:.2f}x restored_tokens={restored_tokens} "
        f"cached={warm_cached}/{total_prompt_tokens} prefilled={warm_prefilled}",
        flush=True,
    )

    # Hard check: warm_prefilled must be at most suffix_tokens (prefix is cached).
    # This is the ground-truth signal that vLLM actually skipped prefix prefill.
    suffix_tokens_approx = total_prompt_tokens - len(pids)
    pass_cached = warm_cached >= len(pids) * (1.0 - max_recomputed_frac)

    # Timing check as secondary signal.
    pass_speedup = speedup >= speedup_threshold

    if not pass_cached:
        print(
            f"[VERIFY_FAIL] vLLM num_cached_tokens={warm_cached} < "
            f"{(1-max_recomputed_frac)*100:.0f}% of prefix_tokens={len(pids)}. "
            "Prefill was NOT eliminated — vLLM recomputed the prefix after AMF restore. "
            "Fix amf_restore_kv + amf_register_prefix so the KV blocks are visible "
            "to vLLM's APC lookup before generate() is called.",
            flush=True,
        )

    if not pass_speedup:
        print(
            f"[VERIFY_FAIL] speedup={speedup:.2f}x < threshold={speedup_threshold:.2f}x "
            "(timing secondary check failed).",
            flush=True,
        )

    if pass_cached and pass_speedup:
        print(
            f"[VERIFY_PASS] prefill elimination confirmed: "
            f"vLLM cached={warm_cached}/{len(pids)} prefix tokens "
            f"warm_prefilled={warm_prefilled} speedup={speedup:.2f}x",
            flush=True,
        )
        return True

    if fail_on_fail:
        print(
            "[VERIFY_FAIL] Aborting. Use verify.fail_on_verify_fail=false to bypass.",
            flush=True,
        )
        return False

    print("[VERIFY_WARN] verification failed but fail_on_verify_fail=false; continuing", flush=True)
    return True


# ---------------------------------------------------------------------------
# Cache disruption helpers
# ---------------------------------------------------------------------------

def _flood_cache(
    llm: Any,
    tokenizer: Any,
    sampling: SamplingParams,
    budget_mb: int,
    kv_bytes_per_token: int,
    max_model_len: int,
    flood_multiplier: float = 2.0,
) -> None:
    """Prefill synthetic flood tokens to evict existing hot-prefix blocks via LRU.

    Flood volume = capacity_tokens × flood_multiplier, split into chunks of
    max_model_len - 64 tokens each.  Each chunk uses a distinct seed to get a
    unique block hash, guaranteeing new block allocation (not re-use of existing).

    Bug fixed: do NOT cap target_tokens at max_model_len.  For small-kv models
    (e.g. Qwen 0.5B with 2 KV heads) capacity >> max_model_len, so a single-chunk
    flood covered only ~13% of cache and left hot-prefix blocks un-evicted.
    """
    capacity_tokens = int(budget_mb * 1024 * 1024 / max(kv_bytes_per_token, 1))
    target_tokens = max(int(capacity_tokens * flood_multiplier), 256)
    chunk_size = max(min(max_model_len - 64, target_tokens), 64)

    flooded = 0
    chunk_idx = 0
    while flooded < target_tokens:
        to_flood = min(chunk_size, target_tokens - flooded)
        # Per-chunk seed → unique token sequence → unique block hash → forces new allocation
        chunk_seed = (
            f"FLOOD_CHUNK_{chunk_idx:05d} EVICTION_TENANT_ROTATION_SYNTHETIC "
            "displace hot-prefix KV blocks from LRU. "
        ) * 200
        chunk_ids = tokenizer.encode(chunk_seed)[:to_flood]
        flood_text = tokenizer.decode(chunk_ids) + " A."
        _generate(llm, flood_text, sampling)  # discard output; only want cache eviction
        flooded += to_flood
        chunk_idx += 1

    print(
        f"[EVICT] eviction_to_zero: flooded {flooded} tokens "
        f"({chunk_idx} chunk(s)) to displace original hot-prefix blocks",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _summarize_metrics(
    metrics: list[RequestMetric],
    *,
    arm: str,
    budget_fraction: float,
    budget_mb: int,
    correctness_checks: int,
    correctness_mismatches: int,
    synthetic_working_set_mb: float,
    regime: str = "all",
) -> dict[str, Any]:
    lat = [m.latency_ms for m in metrics]
    restore = [m.restore_ms for m in metrics if m.restore_ms > 0]
    non_unique = [m for m in metrics if not m.unique]
    hot = [m for m in metrics if m.hot]
    unique = [m for m in metrics if m.unique]

    def rate(rows: list[RequestMetric], attr: str) -> float:
        if not rows:
            return 0.0
        return sum(1 for m in rows if bool(getattr(m, attr))) / len(rows)

    total_reused = sum(m.reused_tokens_before for m in metrics)
    total_recomputed = sum(m.recomputed_prefix_tokens for m in metrics)
    unique_benefit = sum(1 for m in unique if m.any_hit or m.source in {"amf_restore", "apc_live"})

    return {
        "synthetic": True,
        "arm": arm,
        "regime": regime,
        "budget_fraction": budget_fraction,
        "budget_mb": budget_mb,
        "synthetic_working_set_mb": round(synthetic_working_set_mb, 2),
        "requests": len(metrics),
        "any_hit_rate": rate(metrics, "any_hit"),
        "full_hit_rate": rate(metrics, "full_hit"),
        "hot_any_hit_rate": rate(hot, "any_hit"),
        "hot_full_hit_rate": rate(hot, "full_hit"),
        "non_unique_any_hit_rate": rate(non_unique, "any_hit"),
        "non_unique_full_hit_rate": rate(non_unique, "full_hit"),
        "unique_any_hit_rate": rate(unique, "any_hit"),
        "unique_benefit_count": unique_benefit,
        "latency_mean_ms": _mean(lat),
        "latency_p50_ms": _percentile(lat, 50),
        "latency_p95_ms": _percentile(lat, 95),
        "latency_p99_ms": _percentile(lat, 99),
        "restore_mean_ms": _mean(restore),
        "restore_p95_ms": _percentile(restore, 95),
        "prefill_tokens_reused": total_reused,
        "prefill_tokens_recomputed": total_recomputed,
        "gpu_seconds_proxy": _cost_gpu_seconds(lat),
        "correctness_checks": correctness_checks,
        "correctness_mismatches": correctness_mismatches,
        "correctness_pass": correctness_mismatches == 0,
    }


def _summarize_by_regime(
    metrics: list[RequestMetric],
    *,
    arm: str,
    budget_fraction: float,
    budget_mb: int,
    correctness_by_regime: dict[str, dict[str, int]],
    synthetic_working_set_mb: float,
) -> list[dict[str, Any]]:
    """Return one summary dict per regime, plus a blended summary."""
    regime_metrics: dict[str, list[RequestMetric]] = {}
    for m in metrics:
        regime_metrics.setdefault(m.regime, []).append(m)

    summaries: list[dict[str, Any]] = []
    for regime, rm in regime_metrics.items():
        cc = correctness_by_regime.get(regime, {})
        summaries.append(
            _summarize_metrics(
                rm,
                arm=arm,
                budget_fraction=budget_fraction,
                budget_mb=budget_mb,
                correctness_checks=cc.get("checks", 0),
                correctness_mismatches=cc.get("mismatches", 0),
                synthetic_working_set_mb=synthetic_working_set_mb,
                regime=regime,
            )
        )

    # Blended view: weighted by request count (= actual frequency in this run).
    total = len(metrics)
    if total > 0:
        blended_lat = [m.latency_ms for m in metrics]
        blended_restore = [m.restore_ms for m in metrics if m.restore_ms > 0]
        blended_checks = sum(v.get("checks", 0) for v in correctness_by_regime.values())
        blended_mismatches = sum(v.get("mismatches", 0) for v in correctness_by_regime.values())
        regime_breakdown = {
            r: round(len(rm) / total, 4) for r, rm in regime_metrics.items()
        }
        blended = _summarize_metrics(
            metrics,
            arm=arm,
            budget_fraction=budget_fraction,
            budget_mb=budget_mb,
            correctness_checks=blended_checks,
            correctness_mismatches=blended_mismatches,
            synthetic_working_set_mb=synthetic_working_set_mb,
            regime="blended",
        )
        blended["regime_weights"] = json.dumps(regime_breakdown, sort_keys=True)
        blended["blended_note"] = (
            "Weighted by actual request counts in this run. "
            "See per-regime rows for individual breakdown."
        )
        summaries.append(blended)

    return summaries


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Union of all keys (rows may differ if some regimes have extra fields)
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})


# ---------------------------------------------------------------------------
# APC arm
# ---------------------------------------------------------------------------

def _run_apc_arm(
    cfg: dict[str, Any],
    budget_fraction: float,
    budget_mb: int,
    prefixes: list[PrefixSpec],
    requests: list[RequestSpec],
    disruption_events: list[DisruptionEvent],
    request_tags: dict[int, RequestTag],
    kv_bytes_per_token: int,
    synthetic_working_set_mb: float,
    out_dir: Path,
) -> list[dict[str, Any]]:
    print(f"[ARM_START] arm=apc budget_fraction={budget_fraction} budget_mb={budget_mb}", flush=True)
    llm = _new_llm(cfg, budget_mb, with_amf=False)
    tokenizer = llm.get_tokenizer()
    runtime = cfg["runtime"]
    sampling = SamplingParams(
        temperature=float(runtime.get("temperature", 0.0)),
        max_tokens=int(runtime.get("max_tokens", 1)),
    )
    max_model_len = int(runtime.get("max_model_len", 131072))
    prompt_store = PromptStore(
        tokenizer,
        {p.prefix_id: p for p in prefixes},
        int(cfg["traffic"].get("suffix_tokens", 64)),
    )
    disp_cfg = cfg.get("disruptions", {})
    flood_multiplier = float(disp_cfg.get("eviction_to_zero", {}).get("flood_multiplier", 2.0))

    # Build event lookup: request_index → events firing before it
    events_at: dict[int, list[DisruptionEvent]] = {}
    for e in disruption_events:
        events_at.setdefault(e.request_index, []).append(e)

    metrics: list[RequestMetric] = []
    events_log: list[dict[str, Any]] = []

    for req_idx, req in enumerate(requests):
        # --- Handle disruption events that fire before this request ---
        for event in events_at.get(req_idx, []):
            print(
                f"[DISRUPTION] arm=apc event_type={event.event_type} "
                f"event_id={event.event_id} req_idx={req_idx}",
                flush=True,
            )
            if event.event_type == "eviction_to_zero":
                _flood_cache(llm, tokenizer, sampling, budget_mb, kv_bytes_per_token,
                             max_model_len, flood_multiplier)
            elif event.event_type in ("restart", "cold_routing"):
                _reset(llm)

        tag = request_tags.get(req_idx, RequestTag("steady_state"))
        if tag.disruption_type == "cold_routing" and tag.distance_from_event > 0:
            _reset(llm)
        ptext, pids = prompt_store.prefix(req.prefix_id)
        stext, sids = prompt_store.suffix(req.request_id, req.suffix_id)
        prompt = ptext + stext
        gen_ms, _out, num_cached = _generate(llm, prompt, sampling)
        any_hit = num_cached > 0
        full_hit = num_cached >= max(len(pids) - 1, 1)
        # Use vLLM's actual num_cached_tokens for recomputed count (ground truth).
        recomputed = max(len(pids) - num_cached, 0)
        metric = RequestMetric(
            request_id=req.request_id,
            prefix_id=req.prefix_id,
            arm="apc",
            budget_fraction=budget_fraction,
            unique=req.unique,
            hot=req.hot,
            prefix_tokens=len(pids),
            suffix_tokens=len(sids),
            source="apc_live" if any_hit else "cold",
            any_hit=any_hit,
            full_hit=full_hit,
            reused_tokens_before=num_cached,
            recomputed_prefix_tokens=recomputed,
            latency_ms=gen_ms,
            generate_ms=gen_ms,
            regime=tag.regime,
            disruption_event_id=tag.disruption_event_id,
            disruption_type=tag.disruption_type,
            distance_from_event=tag.distance_from_event,
        )
        metrics.append(metric)
        row = asdict(metric)
        if req.unique and metric.any_hit:
            row["bug"] = "unique_prefix_got_cache_benefit"
            print(f"[BUG] arm=apc unique_prefix_hit request_id={req.request_id}", flush=True)
        events_log.append(row)
        if (req_idx + 1) % 100 == 0:
            print(f"[PROGRESS] arm=apc completed={req_idx + 1}/{len(requests)}", flush=True)

    # Per-regime summaries (no correctness checking in APC arm)
    summaries = _summarize_by_regime(
        metrics,
        arm="apc",
        budget_fraction=budget_fraction,
        budget_mb=budget_mb,
        correctness_by_regime={},
        synthetic_working_set_mb=synthetic_working_set_mb,
    )
    _write_jsonl(out_dir / f"events_apc_budget_{budget_fraction:g}.jsonl", events_log)
    del llm
    return summaries


# ---------------------------------------------------------------------------
# LMCache arm
# ---------------------------------------------------------------------------

def _run_lmcache_arm(
    cfg: dict[str, Any],
    budget_fraction: float,
    budget_mb: int,
    prefixes: list[PrefixSpec],
    requests: list[RequestSpec],
    disruption_events: list[DisruptionEvent],
    request_tags: dict[int, RequestTag],
    kv_bytes_per_token: int,
    synthetic_working_set_mb: float,
    out_dir: Path,
) -> list[dict[str, Any]]:
    print(f"[ARM_START] arm=lmcache budget_fraction={budget_fraction} budget_mb={budget_mb}", flush=True)
    runtime = cfg["runtime"]
    llm = _new_lmcache_llm(cfg, budget_mb)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=float(runtime.get("temperature", 0.0)),
        max_tokens=int(runtime.get("max_tokens", 1)),
    )
    max_model_len = int(runtime.get("max_model_len", 131072))
    prompt_store = PromptStore(
        tokenizer,
        {p.prefix_id: p for p in prefixes},
        int(cfg["traffic"].get("suffix_tokens", 64)),
    )

    disp_cfg = cfg.get("disruptions", {})
    flood_multiplier = float(disp_cfg.get("eviction_to_zero", {}).get("flood_multiplier", 2.0))

    events_at: dict[int, list[DisruptionEvent]] = {}
    for e in disruption_events:
        events_at.setdefault(e.request_index, []).append(e)

    metrics: list[RequestMetric] = []
    events_log: list[dict[str, Any]] = []

    for req_idx, req in enumerate(requests):
        for event in events_at.get(req_idx, []):
            print(
                f"[DISRUPTION] arm=lmcache event_type={event.event_type} "
                f"event_id={event.event_id} req_idx={req_idx}",
                flush=True,
            )
            if event.event_type == "eviction_to_zero":
                _flood_cache(llm, tokenizer, sampling, budget_mb, kv_bytes_per_token,
                             max_model_len, flood_multiplier)
            elif event.event_type in ("restart", "cold_routing"):
                # This models vLLM/APC state disruption while the LMCache tier
                # remains available to the connector in the same process.
                _reset(llm)

        tag = request_tags.get(req_idx, RequestTag("steady_state"))
        if tag.disruption_type == "cold_routing" and tag.distance_from_event > 0:
            _reset(llm)

        ptext, pids = prompt_store.prefix(req.prefix_id)
        stext, sids = prompt_store.suffix(req.request_id, req.suffix_id)
        prompt = ptext + stext
        gen_ms, _out, num_cached = _generate(llm, prompt, sampling)
        any_hit = num_cached > 0
        full_hit = num_cached >= max(len(pids) - 1, 1)
        recomputed = max(len(pids) - num_cached, 0)
        metric = RequestMetric(
            request_id=req.request_id,
            prefix_id=req.prefix_id,
            arm="lmcache",
            budget_fraction=budget_fraction,
            unique=req.unique,
            hot=req.hot,
            prefix_tokens=len(pids),
            suffix_tokens=len(sids),
            source="lmcache_or_apc" if any_hit else "cold",
            any_hit=any_hit,
            full_hit=full_hit,
            reused_tokens_before=num_cached,
            recomputed_prefix_tokens=recomputed,
            latency_ms=gen_ms,
            generate_ms=gen_ms,
            regime=tag.regime,
            disruption_event_id=tag.disruption_event_id,
            disruption_type=tag.disruption_type,
            distance_from_event=tag.distance_from_event,
        )
        metrics.append(metric)
        row = asdict(metric)
        if req.unique and metric.any_hit:
            row["bug"] = "unique_prefix_got_cache_benefit"
            print(f"[BUG] arm=lmcache unique_prefix_hit request_id={req.request_id}", flush=True)
        events_log.append(row)
        if (req_idx + 1) % 100 == 0:
            print(f"[PROGRESS] arm=lmcache completed={req_idx + 1}/{len(requests)}", flush=True)

    summaries = _summarize_by_regime(
        metrics,
        arm="lmcache",
        budget_fraction=budget_fraction,
        budget_mb=budget_mb,
        correctness_by_regime={},
        synthetic_working_set_mb=synthetic_working_set_mb,
    )
    _write_jsonl(out_dir / f"events_lmcache_budget_{budget_fraction:g}.jsonl", events_log)
    del llm
    return summaries


# ---------------------------------------------------------------------------
# AMF arm
# ---------------------------------------------------------------------------

def _run_amf_arm(
    cfg: dict[str, Any],
    budget_fraction: float,
    budget_mb: int,
    prefixes: list[PrefixSpec],
    requests: list[RequestSpec],
    disruption_events: list[DisruptionEvent],
    request_tags: dict[int, RequestTag],
    kv_bytes_per_token: int,
    synthetic_working_set_mb: float,
    out_dir: Path,
    skip_verify: bool = False,
) -> list[dict[str, Any]]:
    print(f"[ARM_START] arm=amf budget_fraction={budget_fraction} budget_mb={budget_mb}", flush=True)
    amf_cfg = cfg["amf"]
    amf_path = (
        Path(str(amf_cfg.get("path_root", "/tmp/korith-production-realistic-amf")))
        / f"budget_{budget_fraction:g}"
    )
    shutil.rmtree(amf_path, ignore_errors=True)
    amf_path.mkdir(parents=True, exist_ok=True)

    os.environ["KORITH_ENABLE_AMF"] = "1"
    os.environ["KORITH_AMF_PATH"] = str(amf_path)
    os.environ["KORITH_AMF_VRAM_FIRST"] = "1"
    os.environ["KORITH_AMF_SYNC_SAVE"] = "1" if bool(amf_cfg.get("sync_save", True)) else "0"
    os.environ["KORITH_VRAM_CACHE_GB"] = str(amf_cfg.get("vram_cache_gb", 8))
    os.environ["KORITH_VRAM_CACHE_DEVICE"] = "cuda:0"
    os.environ["KORITH_VRAM_POOL_GB"] = str(amf_cfg.get("compressed_pool_gb", 0))
    os.environ["KORITH_VRAM_POOL_QUANT"] = str(amf_cfg.get("compressed_quant", "int4"))

    runtime = cfg["runtime"]
    llm = _new_llm(cfg, budget_mb, with_amf=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=float(runtime.get("temperature", 0.0)),
        max_tokens=int(runtime.get("max_tokens", 1)),
    )
    max_model_len = int(runtime.get("max_model_len", 131072))
    prompt_store = PromptStore(
        tokenizer,
        {p.prefix_id: p for p in prefixes},
        int(cfg["traffic"].get("suffix_tokens", 64)),
    )

    # --- Pre-flight: prefill elimination verification ---
    verify_ok = _verify_prefill_elimination(
        llm, tokenizer, sampling, prompt_store, prefixes, amf_path, cfg, skip=skip_verify
    )
    if not verify_ok:
        raise RuntimeError(
            "AMF prefill-elimination verification FAILED. "
            "Cannot trust benchmark results on this path. "
            "Fix the restore/register pipeline before re-running."
        )

    disp_cfg = cfg.get("disruptions", {})
    flood_multiplier = float(disp_cfg.get("eviction_to_zero", {}).get("flood_multiplier", 2.0))

    # Build event lookup
    events_at: dict[int, list[DisruptionEvent]] = {}
    for e in disruption_events:
        events_at.setdefault(e.request_index, []).append(e)

    materialized: set[str] = set()
    metrics: list[RequestMetric] = []
    events_log: list[dict[str, Any]] = []
    correctness_rng = random.Random(int(cfg.get("seed", 1337)) + 999)
    correctness_cfg = cfg.get("correctness", {})
    correctness_fraction = float(correctness_cfg.get("sample_fraction", 0.10))
    max_checks = int(correctness_cfg.get("max_checks", 100))
    fail_on_mismatch = bool(correctness_cfg.get("fail_on_mismatch", True))
    checks = 0
    mismatches = 0
    # Per-regime correctness tracking
    correctness_by_regime: dict[str, dict[str, int]] = {}

    for req_idx, req in enumerate(requests):
        # --- Handle disruption events before this request ---
        for event in events_at.get(req_idx, []):
            print(
                f"[DISRUPTION] arm=amf event_type={event.event_type} "
                f"event_id={event.event_id} req_idx={req_idx}",
                flush=True,
            )
            if event.event_type == "eviction_to_zero":
                # Flood the APC cache to evict original hot-prefix blocks.
                # AMF's persistent tier (materialized set) is untouched.
                _flood_cache(llm, tokenizer, sampling, budget_mb, kv_bytes_per_token,
                             max_model_len, flood_multiplier)
            elif event.event_type in ("restart", "cold_routing"):
                # Reset the in-engine APC cache.
                # materialized set is preserved — AMF's persistent VRAM tier survives.
                _reset(llm)

        tag = request_tags.get(req_idx, RequestTag("steady_state"))
        if tag.disruption_type == "cold_routing" and tag.distance_from_event > 0:
            _reset(llm)
        ptext, pids = prompt_store.prefix(req.prefix_id)
        stext, sids = prompt_store.suffix(req.request_id, req.suffix_id)
        prompt = ptext + stext
        source = "cold"
        restore_ms = 0.0
        register_ms = 0.0
        save_ms = 0.0
        output_match: bool | None = None
        amf_primary_decision = "cold_fallback"
        fallback_hit = False
        promoted_to_amf = False
        materialized_before = req.prefix_id in materialized

        cov = _coverage(llm, pids)
        should_restore = (not cov["full_hit"]) and req.prefix_id in materialized
        should_check = (
            should_restore
            and checks < max_checks
            and correctness_rng.random() < correctness_fraction
        )

        cold_reference: list[int] | None = None
        if should_check:
            _reset(llm)
            _cold_ms, cold_reference, _cc = _generate(llm, prompt, sampling)
            _reset(llm)
            cov = _coverage(llm, pids)
            checks += 1

        if cov["full_hit"]:
            source = "apc_live"
        elif cov["any_hit"]:
            source = "apc_partial"
        elif should_restore:
            t_restore = time.perf_counter()
            restored = llm.collective_rpc(
                "amf_restore_kv",
                timeout=600,
                args=(str(amf_path), list(pids), 0, "__shared__"),
            )
            restore_ms = (time.perf_counter() - t_restore) * 1000.0
            restored_tokens = int(restored[0] if restored else 0)
            block_size = int(cov.get("block_size", 16) or 16)
            restore_block_ids = list(range(max(len(pids) // block_size, 1)))
            t_register = time.perf_counter()
            _call_engine_core(llm, "amf_register_prefix", list(pids), restore_block_ids)
            register_ms = (time.perf_counter() - t_register) * 1000.0
            cov = _coverage(llm, pids)
            source = "amf_restore" if restored_tokens > 0 and cov["any_hit"] else "cold_after_restore_miss"

        gen_ms, out_tokens, num_cached = _generate(llm, prompt, sampling)

        if cold_reference is not None:
            output_match = out_tokens == cold_reference
            if not output_match:
                mismatches += 1
                r_cc = correctness_by_regime.setdefault(tag.regime, {"checks": 0, "mismatches": 0})
                r_cc["mismatches"] += 1
                print(
                    f"[CORRECTNESS_FAIL] request_id={req.request_id} "
                    f"prefix_id={req.prefix_id} regime={tag.regime} "
                    f"cold={cold_reference} amf={out_tokens}",
                    flush=True,
                )
            r_cc2 = correctness_by_regime.setdefault(tag.regime, {"checks": 0, "mismatches": 0})
            r_cc2["checks"] += 1

        # Materialize if not yet saved.
        if req.prefix_id not in materialized and (
            not req.unique or bool(amf_cfg.get("save_unique_prefixes", True))
        ):
            post_cov = _coverage(llm, pids)
            block_size = int(post_cov.get("block_size", 16) or 16)
            expected_blocks = max(len(pids) // block_size, 1)
            if len(post_cov["block_ids"]) >= expected_blocks:
                t_save = time.perf_counter()
                results = llm.collective_rpc(
                    "amf_save_kv",
                    timeout=600,
                    args=(str(amf_path), list(pids), 0, "__shared__"),
                    kwargs={"physical_block_ids": post_cov["block_ids"][:expected_blocks]},
                )
                save_ms = (time.perf_counter() - t_save) * 1000.0
                info = results[0] if results else {}
                if bool(info.get("saved", False)):
                    materialized.add(req.prefix_id)

        # Use vLLM's actual num_cached_tokens (ground truth, not formula).
        recomputed = max(len(pids) - num_cached, 0)
        metric = RequestMetric(
            request_id=req.request_id,
            prefix_id=req.prefix_id,
            arm="amf",
            budget_fraction=budget_fraction,
            unique=req.unique,
            hot=req.hot,
            prefix_tokens=len(pids),
            suffix_tokens=len(sids),
            source=source,
            any_hit=bool(cov["any_hit"]),
            full_hit=bool(cov["full_hit"]),
            reused_tokens_before=num_cached,
            recomputed_prefix_tokens=recomputed,
            latency_ms=restore_ms + register_ms + gen_ms + save_ms,
            generate_ms=gen_ms,
            restore_ms=restore_ms,
            register_ms=register_ms,
            save_ms=save_ms,
            output_match=output_match,
            regime=tag.regime,
            disruption_event_id=tag.disruption_event_id,
            disruption_type=tag.disruption_type,
            distance_from_event=tag.distance_from_event,
        )
        metrics.append(metric)
        row = asdict(metric)
        row["materialized_prefixes"] = len(materialized)
        if req.unique and (metric.any_hit or metric.source in {"amf_restore", "apc_live"}):
            row["bug"] = "unique_prefix_got_cache_benefit"
            print(f"[BUG] arm=amf unique_prefix_hit request_id={req.request_id}", flush=True)
        events_log.append(row)
        if (req_idx + 1) % 100 == 0:
            print(f"[PROGRESS] arm=amf completed={req_idx + 1}/{len(requests)}", flush=True)

    summaries = _summarize_by_regime(
        metrics,
        arm="amf",
        budget_fraction=budget_fraction,
        budget_mb=budget_mb,
        correctness_by_regime=correctness_by_regime,
        synthetic_working_set_mb=synthetic_working_set_mb,
    )
    _write_jsonl(out_dir / f"events_amf_budget_{budget_fraction:g}.jsonl", events_log)
    del llm

    if fail_on_mismatch and mismatches:
        raise RuntimeError(f"AMF correctness failed: {mismatches}/{checks} sampled outputs mismatched")
    return summaries


# ---------------------------------------------------------------------------
# AMF + LMCache combined arm
# ---------------------------------------------------------------------------

def _run_amf_lmcache_arm(
    cfg: dict[str, Any],
    budget_fraction: float,
    budget_mb: int,
    prefixes: list[PrefixSpec],
    requests: list[RequestSpec],
    disruption_events: list[DisruptionEvent],
    request_tags: dict[int, RequestTag],
    kv_bytes_per_token: int,
    synthetic_working_set_mb: float,
    out_dir: Path,
    skip_verify: bool = False,
) -> list[dict[str, Any]]:
    print(f"[ARM_START] arm=amf_lmcache budget_fraction={budget_fraction} budget_mb={budget_mb}", flush=True)
    amf_cfg = cfg["amf"]
    amf_path = (
        Path(str(amf_cfg.get("path_root", "/tmp/korith-production-realistic-amf-lmcache")))
        / f"budget_{budget_fraction:g}_amf_lmcache"
    )
    shutil.rmtree(amf_path, ignore_errors=True)
    amf_path.mkdir(parents=True, exist_ok=True)

    os.environ["KORITH_ENABLE_AMF"] = "1"
    os.environ["KORITH_AMF_PATH"] = str(amf_path)
    os.environ["KORITH_AMF_VRAM_FIRST"] = "1"
    os.environ["KORITH_AMF_SYNC_SAVE"] = "1" if bool(amf_cfg.get("sync_save", True)) else "0"
    os.environ["KORITH_VRAM_CACHE_GB"] = str(amf_cfg.get("vram_cache_gb", 8))
    os.environ["KORITH_VRAM_CACHE_DEVICE"] = "cuda:0"
    os.environ["KORITH_VRAM_POOL_GB"] = str(amf_cfg.get("compressed_pool_gb", 0))
    os.environ["KORITH_VRAM_POOL_QUANT"] = str(amf_cfg.get("compressed_quant", "int4"))

    runtime = cfg["runtime"]
    llm = _new_lmcache_llm(cfg, budget_mb, with_amf=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=float(runtime.get("temperature", 0.0)),
        max_tokens=int(runtime.get("max_tokens", 1)),
    )
    max_model_len = int(runtime.get("max_model_len", 131072))
    prompt_store = PromptStore(
        tokenizer,
        {p.prefix_id: p for p in prefixes},
        int(cfg["traffic"].get("suffix_tokens", 64)),
    )

    verify_ok = _verify_prefill_elimination(
        llm, tokenizer, sampling, prompt_store, prefixes, amf_path, cfg, skip=skip_verify
    )
    if not verify_ok:
        raise RuntimeError(
            "AMF+LMCache prefill-elimination verification FAILED. "
            "Cannot trust benchmark results on this path."
        )

    disp_cfg = cfg.get("disruptions", {})
    flood_multiplier = float(disp_cfg.get("eviction_to_zero", {}).get("flood_multiplier", 2.0))

    events_at: dict[int, list[DisruptionEvent]] = {}
    for e in disruption_events:
        events_at.setdefault(e.request_index, []).append(e)

    materialized: set[str] = set()
    metrics: list[RequestMetric] = []
    events_log: list[dict[str, Any]] = []
    correctness_rng = random.Random(int(cfg.get("seed", 1337)) + 1999)
    correctness_cfg = cfg.get("correctness", {})
    correctness_fraction = float(correctness_cfg.get("sample_fraction", 0.10))
    max_checks = int(correctness_cfg.get("max_checks", 100))
    fail_on_mismatch = bool(correctness_cfg.get("fail_on_mismatch", True))
    checks = 0
    mismatches = 0
    correctness_by_regime: dict[str, dict[str, int]] = {}

    for req_idx, req in enumerate(requests):
        for event in events_at.get(req_idx, []):
            print(
                f"[DISRUPTION] arm=amf_lmcache event_type={event.event_type} "
                f"event_id={event.event_id} req_idx={req_idx}",
                flush=True,
            )
            if event.event_type == "eviction_to_zero":
                _flood_cache(llm, tokenizer, sampling, budget_mb, kv_bytes_per_token,
                             max_model_len, flood_multiplier)
            elif event.event_type in ("restart", "cold_routing"):
                _reset(llm)

        tag = request_tags.get(req_idx, RequestTag("steady_state"))
        if tag.disruption_type == "cold_routing" and tag.distance_from_event > 0:
            _reset(llm)

        ptext, pids = prompt_store.prefix(req.prefix_id)
        stext, sids = prompt_store.suffix(req.request_id, req.suffix_id)
        prompt = ptext + stext
        source = "cold"
        restore_ms = 0.0
        register_ms = 0.0
        save_ms = 0.0
        output_match: bool | None = None
        amf_primary_decision = "cold_fallback"
        fallback_hit = False
        promoted_to_amf = False
        materialized_before = req.prefix_id in materialized

        cov = _coverage(llm, pids)
        should_restore = (not cov["full_hit"]) and req.prefix_id in materialized
        should_check = (
            should_restore
            and checks < max_checks
            and correctness_rng.random() < correctness_fraction
        )

        cold_reference: list[int] | None = None
        if should_check:
            _reset(llm)
            _cold_ms, cold_reference, _cc = _generate(llm, prompt, sampling)
            _reset(llm)
            cov = _coverage(llm, pids)
            checks += 1

        if cov["full_hit"]:
            source = "apc_or_lmcache_live"
            amf_primary_decision = "live_prefix_hit"
        elif cov["any_hit"]:
            source = "apc_or_lmcache_partial"
            amf_primary_decision = "live_partial_hit"
        elif should_restore:
            amf_primary_decision = "amf_restore_attempt"
            t_restore = time.perf_counter()
            restored = llm.collective_rpc(
                "amf_restore_kv",
                timeout=600,
                args=(str(amf_path), list(pids), 0, "__shared__"),
            )
            restore_ms = (time.perf_counter() - t_restore) * 1000.0
            restored_tokens = int(restored[0] if restored else 0)
            block_size = int(cov.get("block_size", 16) or 16)
            restore_block_ids = list(range(max(len(pids) // block_size, 1)))
            t_register = time.perf_counter()
            _call_engine_core(llm, "amf_register_prefix", list(pids), restore_block_ids)
            register_ms = (time.perf_counter() - t_register) * 1000.0
            cov = _coverage(llm, pids)
            source = "amf_restore" if restored_tokens > 0 and cov["any_hit"] else "lmcache_fallback_after_amf_miss"
        else:
            amf_primary_decision = "lmcache_fallback_lookup"

        gen_ms, out_tokens, num_cached = _generate(llm, prompt, sampling)

        if source == "cold" and num_cached > 0:
            source = "lmcache_fallback"
            fallback_hit = True
        elif source == "lmcache_fallback_after_amf_miss" and num_cached == 0:
            source = "cold_after_amf_restore_miss"
        elif source == "lmcache_fallback_after_amf_miss" and num_cached > 0:
            fallback_hit = True

        if cold_reference is not None:
            output_match = out_tokens == cold_reference
            if not output_match:
                mismatches += 1
                r_cc = correctness_by_regime.setdefault(tag.regime, {"checks": 0, "mismatches": 0})
                r_cc["mismatches"] += 1
                print(
                    f"[CORRECTNESS_FAIL] arm=amf_lmcache request_id={req.request_id} "
                    f"prefix_id={req.prefix_id} regime={tag.regime} "
                    f"cold={cold_reference} warm={out_tokens}",
                    flush=True,
                )
            r_cc2 = correctness_by_regime.setdefault(tag.regime, {"checks": 0, "mismatches": 0})
            r_cc2["checks"] += 1

        if req.prefix_id not in materialized and (
            not req.unique or bool(amf_cfg.get("save_unique_prefixes", True))
        ):
            post_cov = _coverage(llm, pids)
            block_size = int(post_cov.get("block_size", 16) or 16)
            expected_blocks = max(len(pids) // block_size, 1)
            if len(post_cov["block_ids"]) >= expected_blocks:
                t_save = time.perf_counter()
                results = llm.collective_rpc(
                    "amf_save_kv",
                    timeout=600,
                    args=(str(amf_path), list(pids), 0, "__shared__"),
                    kwargs={"physical_block_ids": post_cov["block_ids"][:expected_blocks]},
                )
                save_ms = (time.perf_counter() - t_save) * 1000.0
                info = results[0] if results else {}
                if bool(info.get("saved", False)):
                    materialized.add(req.prefix_id)
                    promoted_to_amf = not materialized_before

        recomputed = max(len(pids) - num_cached, 0)
        metric = RequestMetric(
            request_id=req.request_id,
            prefix_id=req.prefix_id,
            arm="amf_lmcache",
            budget_fraction=budget_fraction,
            unique=req.unique,
            hot=req.hot,
            prefix_tokens=len(pids),
            suffix_tokens=len(sids),
            source=source,
            any_hit=num_cached > 0,
            full_hit=num_cached >= max(len(pids) - 1, 1),
            reused_tokens_before=num_cached,
            recomputed_prefix_tokens=recomputed,
            latency_ms=restore_ms + register_ms + gen_ms + save_ms,
            generate_ms=gen_ms,
            restore_ms=restore_ms,
            register_ms=register_ms,
            save_ms=save_ms,
            output_match=output_match,
            regime=tag.regime,
            disruption_event_id=tag.disruption_event_id,
            disruption_type=tag.disruption_type,
            distance_from_event=tag.distance_from_event,
        )
        metrics.append(metric)
        row = asdict(metric)
        row["materialized_prefixes"] = len(materialized)
        row["amf_primary_decision"] = amf_primary_decision
        row["fallback_hit"] = fallback_hit
        row["promoted_to_amf"] = promoted_to_amf
        if req.unique and metric.any_hit:
            row["bug"] = "unique_prefix_got_cache_benefit"
            print(f"[BUG] arm=amf_lmcache unique_prefix_hit request_id={req.request_id}", flush=True)
        events_log.append(row)
        if (req_idx + 1) % 100 == 0:
            print(f"[PROGRESS] arm=amf_lmcache completed={req_idx + 1}/{len(requests)}", flush=True)

    summaries = _summarize_by_regime(
        metrics,
        arm="amf_lmcache",
        budget_fraction=budget_fraction,
        budget_mb=budget_mb,
        correctness_by_regime=correctness_by_regime,
        synthetic_working_set_mb=synthetic_working_set_mb,
    )
    _write_jsonl(out_dir / f"events_amf_lmcache_budget_{budget_fraction:g}.jsonl", events_log)
    del llm

    if fail_on_mismatch and mismatches:
        raise RuntimeError(f"AMF+LMCache correctness failed: {mismatches}/{checks} sampled outputs mismatched")
    return summaries


# ---------------------------------------------------------------------------
# Output / printing
# ---------------------------------------------------------------------------

_REGIME_ORDER = ["steady_state", "post_restart", "post_eviction", "post_cold_routing", "blended"]


def _regime_sort_key(row: dict[str, Any]) -> tuple[int, str, float]:
    regime = str(row.get("regime", "all"))
    try:
        r_idx = _REGIME_ORDER.index(regime)
    except ValueError:
        r_idx = len(_REGIME_ORDER)
    return (r_idx, str(row.get("arm", "")), float(row.get("budget_fraction", 0)))


def _print_table(rows: list[dict[str, Any]], label: str = "SYNTHETIC_RESULTS_TABLE") -> None:
    cols = [
        "budget_fraction",
        "arm",
        "regime",
        "requests",
        "any_hit_rate",
        "full_hit_rate",
        "hot_any_hit_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "restore_mean_ms",
        "gpu_seconds_proxy",
        "correctness_pass",
    ]
    sorted_rows = sorted(rows, key=_regime_sort_key)
    print(f"\n[{label}]", flush=True)
    print(" | ".join(cols), flush=True)
    for row in sorted_rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        print(" | ".join(vals), flush=True)


def _print_comparison(
    summaries: list[dict[str, Any]],
    budget_fraction: float,
) -> None:
    """Print per-regime APC vs LMCache vs AMF comparison for a given budget fraction."""
    by_regime: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summaries:
        if float(row.get("budget_fraction", -1)) != budget_fraction:
            continue
        regime = str(row.get("regime", "all"))
        arm = str(row.get("arm", ""))
        by_regime.setdefault(regime, {})[arm] = row

    for regime in _REGIME_ORDER:
        arms = by_regime.get(regime)
        if not arms:
            continue
        apc = arms.get("apc")
        lmcache = arms.get("lmcache")
        amf = arms.get("amf")
        amf_lmcache = arms.get("amf_lmcache")
        if apc and amf:
            speedup = apc["gpu_seconds_proxy"] / max(amf["gpu_seconds_proxy"], 1e-9)
            blended_note = " [BLENDED — see per-regime rows]" if regime == "blended" else ""
            print(
                f"[SYNTHETIC_COMPARISON] "
                f"budget_fraction={budget_fraction:g} regime={regime}{blended_note} "
                f"apc_gpu_s={apc['gpu_seconds_proxy']:.4f} "
                f"amf_gpu_s={amf['gpu_seconds_proxy']:.4f} "
                f"cost_ratio={speedup:.3f}x "
                f"apc_full_hit={apc['full_hit_rate']:.4f} "
                f"amf_full_hit={amf['full_hit_rate']:.4f} "
                f"amf_restore_p95ms={amf.get('restore_p95_ms', 0.0):.1f} "
                f"apc_correct={apc.get('correctness_pass', True)} "
                f"amf_correct={amf.get('correctness_pass', True)}",
                flush=True,
            )
        if lmcache and amf:
            speedup = lmcache["gpu_seconds_proxy"] / max(amf["gpu_seconds_proxy"], 1e-9)
            print(
                f"[SYNTHETIC_COMPARISON] "
                f"budget_fraction={budget_fraction:g} regime={regime} "
                f"lmcache_gpu_s={lmcache['gpu_seconds_proxy']:.4f} "
                f"amf_gpu_s={amf['gpu_seconds_proxy']:.4f} "
                f"cost_ratio_lmcache_over_amf={speedup:.3f}x "
                f"lmcache_full_hit={lmcache['full_hit_rate']:.4f} "
                f"amf_full_hit={amf['full_hit_rate']:.4f} "
                f"lmcache_correct={lmcache.get('correctness_pass', True)} "
                f"amf_correct={amf.get('correctness_pass', True)}",
                flush=True,
            )
        if lmcache and amf_lmcache:
            speedup = lmcache["gpu_seconds_proxy"] / max(amf_lmcache["gpu_seconds_proxy"], 1e-9)
            print(
                f"[SYNTHETIC_COMPARISON] "
                f"budget_fraction={budget_fraction:g} regime={regime} "
                f"lmcache_gpu_s={lmcache['gpu_seconds_proxy']:.4f} "
                f"amf_lmcache_gpu_s={amf_lmcache['gpu_seconds_proxy']:.4f} "
                f"cost_ratio_lmcache_over_amf_lmcache={speedup:.3f}x "
                f"lmcache_full_hit={lmcache['full_hit_rate']:.4f} "
                f"amf_lmcache_full_hit={amf_lmcache['full_hit_rate']:.4f} "
                f"amf_lmcache_correct={amf_lmcache.get('correctness_pass', True)}",
                flush=True,
            )
        if amf and amf_lmcache:
            speedup = amf["gpu_seconds_proxy"] / max(amf_lmcache["gpu_seconds_proxy"], 1e-9)
            print(
                f"[SYNTHETIC_COMPARISON] "
                f"budget_fraction={budget_fraction:g} regime={regime} "
                f"amf_gpu_s={amf['gpu_seconds_proxy']:.4f} "
                f"amf_lmcache_gpu_s={amf_lmcache['gpu_seconds_proxy']:.4f} "
                f"cost_ratio_amf_over_amf_lmcache={speedup:.3f}x "
                f"amf_full_hit={amf['full_hit_rate']:.4f} "
                f"amf_lmcache_full_hit={amf_lmcache['full_hit_rate']:.4f} "
                f"amf_lmcache_correct={amf_lmcache.get('correctness_pass', True)}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/production_realistic_apc_vs_amf.yaml")
    parser.add_argument("--only-budget", type=float, default=None)
    parser.add_argument("--only-arm", choices=["apc", "lmcache", "amf", "amf_lmcache"], default=None)
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help="Skip AMF prefill-elimination verification (not recommended; use only for debugging).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    prefixes, requests = _generate_traffic(cfg)
    prefix_by_id = {p.prefix_id: p for p in prefixes}
    runtime = cfg["runtime"]
    cache_cfg = cfg["cache"]

    kv_bytes_per_token = cache_cfg.get("kv_bytes_per_token")
    if kv_bytes_per_token is None:
        kv_bytes_per_token = _estimate_kv_bytes_per_token(
            cfg["model"], runtime.get("kv_cache_dtype", "auto")
        )
    kv_bytes_per_token = int(kv_bytes_per_token)

    reusable_seen = {req.prefix_id for req in requests if not req.unique}
    working_set_tokens = sum(prefix_by_id[pid].target_tokens for pid in reusable_seen)
    working_set_bytes = working_set_tokens * kv_bytes_per_token
    working_set_mb = working_set_bytes / (1024 * 1024)
    fractions = [float(x) for x in cache_cfg.get("budget_fractions", [0.25, 0.50, 1.0, 1.5])]
    if args.only_budget is not None:
        fractions = [float(args.only_budget)]

    out_dir = Path(str(cfg.get("output_dir", "benchmark_results/production_realistic")))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Schedule disruption events (same for both arms; computed once).
    n_requests = len(requests)
    disruption_events = _schedule_disruptions(cfg, n_requests)
    request_tags = _tag_requests_with_regime(n_requests, disruption_events)

    regime_counts: dict[str, int] = {}
    for tag in request_tags.values():
        regime_counts[tag.regime] = regime_counts.get(tag.regime, 0) + 1

    print(
        "[SYNTHETIC_SETUP] "
        f"model={cfg['model']} requests={len(requests)} "
        f"reusable_prefixes={cfg['traffic'].get('reusable_prefixes')} "
        f"unique_fraction={cfg['traffic'].get('unique_fraction')} "
        f"working_set_mb={working_set_mb:.2f} "
        f"kv_bytes_per_token={kv_bytes_per_token} "
        f"disruption_events={len(disruption_events)} "
        f"regime_counts={json.dumps(regime_counts)} "
        f"output_dir={out_dir}",
        flush=True,
    )

    (out_dir / "traffic.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "config": cfg,
                "kv_bytes_per_token": kv_bytes_per_token,
                "working_set_tokens": working_set_tokens,
                "working_set_mb": working_set_mb,
                "disruption_events": [asdict(e) for e in disruption_events],
                "regime_counts": regime_counts,
                "prefixes": [asdict(p) for p in prefixes],
                "requests": [asdict(r) for r in requests],
                "request_tags": {
                    str(i): asdict(t) for i, t in request_tags.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    all_summaries: list[dict[str, Any]] = []
    min_budget_mb = int(cache_cfg.get("min_kv_cache_memory_mb", 512))

    for fraction in fractions:
        budget_mb = max(min_budget_mb, int(math.ceil(working_set_mb * fraction)))

        if args.only_arm in (None, "apc"):
            arm_budget_mb = _arm_kv_budget_mb(cache_cfg, "apc", budget_mb)
            apc_summaries = _run_apc_arm(
                cfg, fraction, arm_budget_mb, prefixes, requests,
                disruption_events, request_tags,
                kv_bytes_per_token, working_set_mb, out_dir,
            )
            all_summaries.extend(apc_summaries)

        if args.only_arm in (None, "lmcache"):
            arm_budget_mb = _arm_kv_budget_mb(cache_cfg, "lmcache", budget_mb)
            lmcache_summaries = _run_lmcache_arm(
                cfg, fraction, arm_budget_mb, prefixes, requests,
                disruption_events, request_tags,
                kv_bytes_per_token, working_set_mb, out_dir,
            )
            all_summaries.extend(lmcache_summaries)

        if args.only_arm in (None, "amf"):
            arm_budget_mb = _arm_kv_budget_mb(cache_cfg, "amf", budget_mb)
            amf_summaries = _run_amf_arm(
                cfg, fraction, arm_budget_mb, prefixes, requests,
                disruption_events, request_tags,
                kv_bytes_per_token, working_set_mb, out_dir,
                skip_verify=args.skip_verify,
            )
            all_summaries.extend(amf_summaries)

        if args.only_arm in (None, "amf_lmcache"):
            arm_budget_mb = _arm_kv_budget_mb(cache_cfg, "amf_lmcache", budget_mb)
            amf_lmcache_summaries = _run_amf_lmcache_arm(
                cfg, fraction, arm_budget_mb, prefixes, requests,
                disruption_events, request_tags,
                kv_bytes_per_token, working_set_mb, out_dir,
                skip_verify=args.skip_verify,
            )
            all_summaries.extend(amf_lmcache_summaries)

    _write_jsonl(out_dir / "summary.jsonl", all_summaries)
    _write_csv(out_dir / "summary.csv", all_summaries)
    _print_table(all_summaries)

    print("\n[SYNTHETIC_COMPARISONS_BY_REGIME]", flush=True)
    for fraction in fractions:
        _print_comparison(all_summaries, fraction)

    print(
        "\n[NOTE] The 'blended' regime is weighted by actual request counts in this run. "
        "The steady_state and post_* rows show the per-regime breakdown. "
        "Do not cite the blended number without showing the per-regime breakdown alongside it. "
        "All data is synthetic.",
        flush=True,
    )


if __name__ == "__main__":
    main()
