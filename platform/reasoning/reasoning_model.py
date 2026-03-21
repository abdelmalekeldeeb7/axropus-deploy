"""
Component 2: Reasoning Model Interface
========================================
Wraps Llama 3.1 8B (4-bit NF4) to produce AMF decision vectors from the
64-element metrics tensor produced by MetricsCollector.

Tensor layout: slots 0-31 = prefill/cache signals, slots 32-63 = decode signals.
The model outputs a two-section JSON with "prefill" and "decode" keys, returning
a UnifiedDecisionVector that controls both cache policy and speculative decode.

Latency reality
---------------
5ms per cycle is not achievable with an 8B model. Realistic on a dedicated H100
with 4-bit quantization and system-prompt KV pre-caching:
  - System prompt prefill : 0 ms (pre-cached at load time — AMF's own trick)
  - User input prefill    : 5–15 ms (~130 tokens with both sections)
  - JSON decode (greedy)  : 20–40 ms (~150 output tokens for two-section JSON)
  - Total                 : 25–55 ms

`max_inference_ms` (default 50) enforces a hard deadline. If inference exceeds
it or produces invalid JSON, `decide()` returns a safe no-op fallback that
leaves every AMF parameter unchanged. The fallback sets `adaptive_depth_override=False`
so the Rust scheduler retains full decode control. Component 3 (DecisionExecutor)
applies further safety gating on top.

Usage
-----
    model = ReasoningModel(device="cuda:1", max_inference_ms=50.0)
    model.load()                              # ~30s on first call (downloads weights)

    tensor = collector.get_metrics_tensor()  # numpy float32 (64,)
    dv     = model.decide(tensor)            # UnifiedDecisionVector

    if not dv.is_fallback:
        # act on dv
        pass

    model.unload()

Dependencies
------------
    pip install transformers bitsandbytes torch accelerate

Note: if bitsandbytes is unavailable the model loads in float16 instead of 4-bit.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics_collector import Slot

log = logging.getLogger(__name__)

# ── required top-level JSON keys in model output ─────────────────────────────
_REQUIRED_KEYS = {"prefill", "decode"}
_REQUIRED_PREFILL_KEYS = {"cache_admission", "admission_priority", "throttle_back_pressure"}
_REQUIRED_DECODE_KEYS  = {"spec_decode_enable", "spec_decode_depth", "spec_mode", "adaptive_depth_override"}

_SYSTEM_PROMPT = """\
You are an LLM inference controller for AMF (Adaptive Memory Field) with NVIDIA Dynamo fleet awareness.
Given prefill, Dynamo fleet, and decode metrics, output a JSON object with EXACTLY two sections.
No extra text, no markdown, no explanation:

{"prefill":{"cache_admission":true,"admission_priority":0.5,"eviction_target":null,\
"pre_warm_predictions":[],"route_decision":null,"batch_group":null,"throttle_back_pressure":0.0,\
"prefer_local_restore":true,"cross_node_restore_target":null,"persist_priority":0.5,\
"replicate_to_nodes":[],"evict_hint_nodes":[]},\
"decode":{"spec_decode_enable":true,"spec_decode_depth":4,"spec_mode":"v2_clean",\
"spec_v3_block_size":8,"draft_temperature":0.8,"kv_compression_enable":false,\
"kv_compression_ratio":1.0,"decode_batch_priority":2,"early_stop_confidence":0.0,\
"cuda_graph_hint":true,"adaptive_depth_override":false}}

Prefill field rules:
- cache_admission: bool. Admit prefix to cache?
- admission_priority: float 0-1. Higher = admit first under storage pressure.
- eviction_target: string prefix_hash to evict, or null.
- pre_warm_predictions: list of up to 5 prefix_hash strings.
- route_decision: node_id string, or null.
- batch_group: group_id string, or null.
- throttle_back_pressure: float 0-1. 0=no throttle.
- prefer_local_restore: bool. True=restore from local AMF node before cross-node.
- cross_node_restore_target: node_id string to restore from cross-node, or null.
- persist_priority: float 0-1. Higher=persist to NIXL storage with higher priority.
- replicate_to_nodes: list of up to 3 node_id strings to replicate this prefix to.
- evict_hint_nodes: list of up to 3 node_id strings to hint eviction on (cold nodes).

Dynamo fleet field guidance (from dyn_* metrics in input):
- High dyn_max_load (>0.8): prefer_local_restore=false, cross_node_restore_target to a less loaded node.
- High dyn_restore_ratio (>0.5): increase persist_priority, replicate_to popular nodes.
- High dyn_xfers: reduce cross_node_restore_target usage to cut bandwidth.
- Low dyn_g1_pct + high dyn_g3g4_pct: data in slow tiers, increase persist_priority.

Decode field rules:
- spec_decode_enable: bool. Enable speculative decode?
- spec_decode_depth: int 1-16. Draft token depth.
- spec_mode: "v2_clean" | "v3_block" | "disabled".
- spec_v3_block_size: int 4-32. Block size for v3_block mode.
- draft_temperature: float 0-1. Draft sampling temperature.
- kv_compression_enable: bool. Compress KV cache?
- kv_compression_ratio: float 0.1-1.0. Target ratio.
- decode_batch_priority: int 1-4. Lane priority hint.
- early_stop_confidence: float 0-1. 0=disabled.
- cuda_graph_hint: bool. Use CUDA graph capture?
- adaptive_depth_override: bool. False=Rust scheduler controls depth.

Output ONLY the JSON object. No other text."""

# Prefill slot → compact prompt key
_PREFILL_SLOT_KEYS: Dict[int, str] = {
    Slot.HIT_RATE:           "hit_rate",
    Slot.STORAGE_UTIL:       "storage_util",
    Slot.EVICTION_PRESSURE:  "evict_pressure",
    Slot.HOT_ENTRY_RATIO:    "hot_ratio",
    Slot.COLD_ENTRY_RATIO:   "cold_ratio",
    Slot.MISS_STREAK_NORM:   "miss_streak",
    Slot.HIT_STREAK_NORM:    "hit_streak",
    Slot.REQUEST_RATE:       "req_rate",
    Slot.EVICTION_RATE:      "evict_rate",
    Slot.ADMISSION_RATE:     "admit_rate",
    Slot.REJECTION_RATE:     "reject_rate",
    Slot.FAILURE_RATE:       "failure_rate",
    Slot.RESTORE_MS_AVG:     "restore_ms",
    Slot.DECODE_LATENCY_NORM:"decode_lat",
    Slot.AVG_NODE_HIT_RATE:  "node_hit_rate",
    Slot.AVG_NODE_WARMTH:    "node_warmth",
    Slot.NODE_COUNT:         "nodes",
    Slot.TOTAL_CACHE_GB:     "cache_gb",
    Slot.ROI_EMA_NORM:       "roi_ema",
    Slot.MEMORY_PRESSURE:    "mem_pressure",
    Slot.REUSE_RATIO:        "reuse_ratio",
    Slot.MISS_QUEUE_NORM:    "miss_queue",
    # Dynamo fleet slots
    Slot.DYNAMO_FLEET_HIT_RATE:      "dyn_fleet_hit",
    Slot.DYNAMO_FLEET_MISS_RATE:     "dyn_fleet_miss",
    Slot.DYNAMO_EVICTION_RATE:       "dyn_evict_rate",
    Slot.DYNAMO_BLOCK_COUNT_NORM:    "dyn_blocks",
    Slot.DYNAMO_WORKER_COUNT:        "dyn_workers",
    Slot.DYNAMO_AVG_WORKER_LOAD:     "dyn_avg_load",
    Slot.DYNAMO_MAX_WORKER_LOAD:     "dyn_max_load",
    Slot.DYNAMO_KV_OVERLAP_AVG:      "dyn_kv_overlap",
    Slot.DYNAMO_AMF_ROUTED_RATIO:    "dyn_amf_ratio",
    Slot.DYNAMO_RESTORE_RATIO:       "dyn_restore_ratio",
    Slot.DYNAMO_CROSS_NODE_XFERS:    "dyn_xfers",
    Slot.DYNAMO_TIER_G1_PCT:         "dyn_g1_pct",
    Slot.DYNAMO_TIER_G2_PCT:         "dyn_g2_pct",
    Slot.DYNAMO_TIER_G3G4_PCT:       "dyn_g3g4_pct",
}

# Decode slot → compact prompt key
_DECODE_SLOT_KEYS: Dict[int, str] = {
    Slot.ACCEPTANCE_EMA:        "accept_ema",
    Slot.ACCEPTANCE_VARIANCE:   "accept_var",
    Slot.CURRENT_SPEC_DEPTH:    "spec_depth",
    Slot.ENTROPY_EMA:           "entropy",
    Slot.TPS_CURRENT:           "tps",
    Slot.TPS_DELTA:             "tps_delta",
    Slot.SPEC_PROFIT_RATIO:     "spec_profit",
    Slot.SPEC_BAD_STREAK:       "bad_streak",
    Slot.LANE_CURRENT:          "lane",
    Slot.LANE_HOLD_REMAINING:   "lane_hold",
    Slot.STARVATION_COUNTER:    "starve",
    Slot.KV_CACHE_UTILIZATION:  "kv_util",
    Slot.CUDA_GRAPH_ACTIVE:     "cuda_graph",
    Slot.DECODE_LATENCY_PER_TOKEN: "dec_lat_tok",
    Slot.V3_BLOCK_DEPTH:        "v3_depth",
    Slot.HOLO_FUTURE_SAFE:      "holo_safe",
    Slot.POWER_WATTS_NORM:      "power",
}

# Legacy alias — used by tests and existing code that references _SLOT_KEYS
_SLOT_KEYS: Dict[int, str] = {**_PREFILL_SLOT_KEYS, **_DECODE_SLOT_KEYS}

# ── DecisionVector (kept for backward compatibility) ─────────────────────────

@dataclass
class DecisionVector:
    """Structured output from one reasoning model inference cycle."""
    cache_admission: bool
    admission_priority: float        # [0, 1]
    eviction_target: Optional[str]   # prefix_hash or None
    pre_warm_predictions: List[str]  # up to 5 prefix hashes
    route_decision: Optional[str]    # node_id or None
    batch_group: Optional[str]       # group_id or None
    spec_decode_enable: bool
    spec_decode_k: int               # [1, 8]
    throttle_back_pressure: float    # [0, 1]
    inference_ms: float              # wall-clock time for this decision
    is_fallback: bool                # True → model timed out or errored

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_admission":       self.cache_admission,
            "admission_priority":    round(self.admission_priority, 4),
            "eviction_target":       self.eviction_target,
            "pre_warm_predictions":  self.pre_warm_predictions,
            "route_decision":        self.route_decision,
            "batch_group":           self.batch_group,
            "spec_decode_enable":    self.spec_decode_enable,
            "spec_decode_k":         self.spec_decode_k,
            "throttle_back_pressure": round(self.throttle_back_pressure, 4),
            "inference_ms":          round(self.inference_ms, 2),
            "is_fallback":           self.is_fallback,
        }


# ── UnifiedDecisionVector ─────────────────────────────────────────────────────

@dataclass
class UnifiedDecisionVector:
    """Full decision vector controlling both prefill/cache policy and decode."""

    # --- prefill / cache fields ---
    cache_admission: bool
    admission_priority: float        # [0, 1]
    eviction_target: Optional[str]   # prefix_hash or None
    pre_warm_predictions: List[str]  # up to 5 prefix hashes
    route_decision: Optional[str]    # node_id or None
    batch_group: Optional[str]       # group_id or None
    throttle_back_pressure: float    # [0, 1]
    inference_ms: float
    is_fallback: bool
    domain: str                      # "unified"

    # --- Dynamo fleet fields ---
    prefer_local_restore: bool            # restore from local AMF before cross-node
    cross_node_restore_target: Optional[str]  # node_id to restore from cross-node, or None
    persist_priority: float               # [0, 1] priority to persist to NIXL
    replicate_to_nodes: List[str]         # up to 3 node_ids to replicate to
    evict_hint_nodes: List[str]           # up to 3 node_ids to hint eviction on

    # --- decode / speculative fields ---
    spec_decode_enable: bool
    spec_decode_depth: int           # [1, 16]
    spec_mode: str                   # "v2_clean" | "v3_block" | "disabled"
    spec_v3_block_size: int          # [4, 32]
    draft_temperature: float         # [0, 1]
    kv_compression_enable: bool
    kv_compression_ratio: float      # [0.1, 1.0]
    decode_batch_priority: int       # [1, 4]
    early_stop_confidence: float     # [0, 1] — 0 = disabled
    cuda_graph_hint: bool
    adaptive_depth_override: bool    # False = Rust scheduler controls depth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prefill": {
                "cache_admission":           self.cache_admission,
                "admission_priority":        round(self.admission_priority, 4),
                "eviction_target":           self.eviction_target,
                "pre_warm_predictions":      self.pre_warm_predictions,
                "route_decision":            self.route_decision,
                "batch_group":               self.batch_group,
                "throttle_back_pressure":    round(self.throttle_back_pressure, 4),
                "prefer_local_restore":      self.prefer_local_restore,
                "cross_node_restore_target": self.cross_node_restore_target,
                "persist_priority":          round(self.persist_priority, 4),
                "replicate_to_nodes":        self.replicate_to_nodes,
                "evict_hint_nodes":          self.evict_hint_nodes,
            },
            "decode": {
                "spec_decode_enable":     self.spec_decode_enable,
                "spec_decode_depth":      self.spec_decode_depth,
                "spec_mode":              self.spec_mode,
                "spec_v3_block_size":     self.spec_v3_block_size,
                "draft_temperature":      round(self.draft_temperature, 4),
                "kv_compression_enable":  self.kv_compression_enable,
                "kv_compression_ratio":   round(self.kv_compression_ratio, 4),
                "decode_batch_priority":  self.decode_batch_priority,
                "early_stop_confidence":  round(self.early_stop_confidence, 4),
                "cuda_graph_hint":        self.cuda_graph_hint,
                "adaptive_depth_override": self.adaptive_depth_override,
            },
            "inference_ms": round(self.inference_ms, 2),
            "is_fallback":  self.is_fallback,
            "domain":       self.domain,
        }


def _safe_fallback(inference_ms: float = 0.0) -> UnifiedDecisionVector:
    """
    Neutral no-op decision that leaves every AMF and decode parameter at defaults.
    Returned whenever the model is unavailable, times out, or produces bad JSON.

    Critical: adaptive_depth_override=False ensures the Rust scheduler retains
    full control of speculative decode depth on any fallback path.
    """
    return UnifiedDecisionVector(
        cache_admission=True,
        admission_priority=0.5,
        eviction_target=None,
        pre_warm_predictions=[],
        route_decision=None,
        batch_group=None,
        throttle_back_pressure=0.0,
        prefer_local_restore=True,
        cross_node_restore_target=None,
        persist_priority=0.5,
        replicate_to_nodes=[],
        evict_hint_nodes=[],
        inference_ms=inference_ms,
        is_fallback=True,
        domain="unified",
        spec_decode_enable=True,
        spec_decode_depth=4,
        spec_mode="v2_clean",
        spec_v3_block_size=8,
        draft_temperature=0.8,
        kv_compression_enable=False,
        kv_compression_ratio=1.0,
        decode_batch_priority=2,
        early_stop_confidence=0.0,
        cuda_graph_hint=True,
        adaptive_depth_override=False,
    )


# ── ReasoningModel ───────────────────────────────────────────────────────────

class ReasoningModel:
    """
    Llama 3.1 8B (4-bit NF4) wrapper that converts a metrics tensor into a
    DecisionVector for the AMF reasoning layer.

    The system prompt's KV state is pre-computed once at load() time and reused
    on every call — the same principle AMF uses for prefix caching.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "cuda:0",
        max_inference_ms: float = 50.0,
        load_in_4bit: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_inference_ms = float(max_inference_ms)
        self._load_in_4bit = load_in_4bit

        self._model = None
        self._tokenizer = None
        self._system_input_ids = None   # tokenized system prompt
        self._system_kv_cache = None    # pre-computed KV states for system prompt
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the model and tokenizer, then pre-compute the system prompt KV
        cache. Call once at startup. Blocks until complete (~30s on first run
        when downloading weights).
        """
        import torch

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            _have_bnb = True
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            BitsAndBytesConfig = None  # type: ignore[assignment]
            _have_bnb = False

        log.info("[reasoning] loading tokenizer: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            use_fast=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        log.info("[reasoning] loading model (4-bit=%s) on %s", self._load_in_4bit, self._device)

        if self._load_in_4bit and _have_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                quantization_config=bnb_config,
                device_map={"": self._device},
                torch_dtype=torch.float16,
            )
        else:
            if self._load_in_4bit and not _have_bnb:
                log.warning(
                    "[reasoning] bitsandbytes not available; loading in float16 instead"
                )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                device_map={"": self._device},
                torch_dtype=torch.float16,
            )

        self._model.eval()
        self._prewarm_system_kv()
        self._loaded = True
        log.info("[reasoning] model loaded and system prompt KV pre-cached")

    def unload(self) -> None:
        """Free GPU memory. After this, decide() returns safe fallbacks."""
        import gc
        import torch
        self._loaded = False
        self._system_kv_cache = None
        self._system_input_ids = None
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("[reasoning] model unloaded")

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ── main API ─────────────────────────────────────────────────────────────

    def decide(self, tensor: np.ndarray) -> UnifiedDecisionVector:
        """
        Convert a 64-element metrics tensor into a UnifiedDecisionVector.

        If the model is not loaded, inference exceeds max_inference_ms, or the
        output cannot be parsed, returns a safe no-op fallback with
        adaptive_depth_override=False so the Rust scheduler stays in control.

        Parameters
        ----------
        tensor : np.ndarray
            float32 array of shape (64,) from MetricsCollector.get_metrics_tensor()

        Returns
        -------
        UnifiedDecisionVector
        """
        if not self._loaded:
            return _safe_fallback(0.0)

        import torch

        prefill_json, decode_json = self._tensor_to_prompt(tensor)

        # Build full prompt using chat template if available, else plain concat
        full_prompt = self._build_full_prompt(prefill_json, decode_json)

        # Tokenize user portion only (system KV is pre-cached)
        user_ids = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self._device)

        t0 = time.perf_counter()

        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids=user_ids,
                    past_key_values=self._system_kv_cache,
                    max_new_tokens=120,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self._tokenizer.pad_token_id,
                    use_cache=True,
                )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            log.warning("[reasoning] generate() raised: %s", exc)
            return _safe_fallback(elapsed)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if elapsed_ms > self._max_inference_ms:
            log.debug("[reasoning] timeout: %.1f ms > %.1f ms limit", elapsed_ms, self._max_inference_ms)
            return _safe_fallback(elapsed_ms)

        # Decode only the newly generated tokens
        new_ids = output_ids[0][user_ids.shape[1]:]
        raw_text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        parsed = self._parse_output(raw_text)
        if parsed is None:
            log.debug("[reasoning] JSON parse failed; text: %.120s", raw_text)
            return _safe_fallback(elapsed_ms)

        try:
            dv = self._dict_to_decision_vector(parsed, elapsed_ms)
        except Exception as exc:
            log.debug("[reasoning] dict_to_dv failed: %s", exc)
            return _safe_fallback(elapsed_ms)

        return dv

    # ── internal helpers ─────────────────────────────────────────────────────

    def _prewarm_system_kv(self) -> None:
        """
        Run a forward pass over the system prompt to warm its KV cache.
        This is stored and reused on every decide() call — the same principle
        as AMF prefix caching applied to the reasoning model itself.
        """
        import torch

        sys_text = _SYSTEM_PROMPT + "\nPrefill metrics: {}\nDecode metrics: {}\nDecision:"
        ids = self._tokenizer(
            sys_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(self._device)

        self._system_input_ids = ids

        with torch.no_grad():
            out = self._model(ids, use_cache=True)

        # Detach KV cache from grad graph and keep on GPU
        self._system_kv_cache = tuple(
            tuple(kv.detach() for kv in layer_kv)
            for layer_kv in out.past_key_values
        )
        log.debug(
            "[reasoning] system KV pre-cached: %d tokens, %d layers",
            ids.shape[1],
            len(self._system_kv_cache),
        )

    def _build_full_prompt(self, prefill_json: str, decode_json: str) -> str:
        """
        Build the user-side text that follows the pre-cached system prompt.
        Two sections: prefill metrics and decode metrics.
        """
        return f"Prefill metrics: {prefill_json}\nDecode metrics: {decode_json}\nDecision:"

    def _tensor_to_prompt(self, tensor: np.ndarray) -> tuple:
        """
        Split the 64-element tensor into (prefill_json, decode_json) strings.
        Omits slots whose absolute value is below 0.001 to minimise token count.
        """
        prefill: Dict[str, float] = {}
        for slot_idx, key in _PREFILL_SLOT_KEYS.items():
            if slot_idx < len(tensor):
                val = float(tensor[slot_idx])
                if abs(val) >= 0.001:
                    prefill[key] = round(val, 4)

        decode: Dict[str, float] = {}
        for slot_idx, key in _DECODE_SLOT_KEYS.items():
            if slot_idx < len(tensor):
                val = float(tensor[slot_idx])
                if abs(val) >= 0.001:
                    decode[key] = round(val, 4)

        return (
            json.dumps(prefill, separators=(",", ":")),
            json.dumps(decode,  separators=(",", ":")),
        )

    @staticmethod
    def _parse_output(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse the first complete JSON object from model output.
        Requires both "prefill" and "decode" top-level keys with their required
        sub-keys. Returns None on any failure.
        """
        start = text.find("{")
        if start == -1:
            return None
        end = text.rfind("}")
        if end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        if not _REQUIRED_KEYS.issubset(obj.keys()):
            return None
        prefill = obj.get("prefill", {})
        decode  = obj.get("decode", {})
        if not isinstance(prefill, dict) or not isinstance(decode, dict):
            return None
        if not _REQUIRED_PREFILL_KEYS.issubset(prefill.keys()):
            return None
        if not _REQUIRED_DECODE_KEYS.issubset(decode.keys()):
            return None
        return obj

    @staticmethod
    def _dict_to_decision_vector(d: Dict[str, Any], inference_ms: float) -> UnifiedDecisionVector:
        """Convert a validated two-section dict into a UnifiedDecisionVector."""

        def clamp01(v: Any, default: float = 0.0) -> float:
            try:
                return max(0.0, min(1.0, float(v)))
            except Exception:
                return default

        def clamp_depth(v: Any) -> int:
            try:
                return max(1, min(16, int(v)))
            except Exception:
                return 4

        def clamp_block_size(v: Any) -> int:
            try:
                return max(4, min(32, int(v)))
            except Exception:
                return 8

        def clamp_priority(v: Any) -> int:
            try:
                return max(1, min(4, int(v)))
            except Exception:
                return 2

        def valid_spec_mode(v: Any) -> str:
            s = str(v) if v else "v2_clean"
            return s if s in ("v2_clean", "v3_block", "disabled") else "v2_clean"

        p = d.get("prefill", {})
        dc = d.get("decode", {})

        pre_warm = p.get("pre_warm_predictions", [])
        if not isinstance(pre_warm, list):
            pre_warm = []
        pre_warm = [str(x) for x in pre_warm[:5]]

        eviction_target = p.get("eviction_target")
        eviction_target = str(eviction_target) if eviction_target else None

        route_decision = p.get("route_decision")
        route_decision = str(route_decision) if route_decision else None

        batch_group = p.get("batch_group")
        batch_group = str(batch_group) if batch_group else None

        # Fleet fields
        cross_node_target = p.get("cross_node_restore_target")
        cross_node_target = str(cross_node_target) if cross_node_target else None

        replicate_to = p.get("replicate_to_nodes", [])
        if not isinstance(replicate_to, list):
            replicate_to = []
        replicate_to = [str(x) for x in replicate_to[:3]]

        evict_hints = p.get("evict_hint_nodes", [])
        if not isinstance(evict_hints, list):
            evict_hints = []
        evict_hints = [str(x) for x in evict_hints[:3]]

        return UnifiedDecisionVector(
            cache_admission=bool(p.get("cache_admission", True)),
            admission_priority=clamp01(p.get("admission_priority", 0.5), 0.5),
            eviction_target=eviction_target,
            pre_warm_predictions=pre_warm,
            route_decision=route_decision,
            batch_group=batch_group,
            throttle_back_pressure=clamp01(p.get("throttle_back_pressure", 0.0)),
            prefer_local_restore=bool(p.get("prefer_local_restore", True)),
            cross_node_restore_target=cross_node_target,
            persist_priority=clamp01(p.get("persist_priority", 0.5), 0.5),
            replicate_to_nodes=replicate_to,
            evict_hint_nodes=evict_hints,
            inference_ms=inference_ms,
            is_fallback=False,
            domain="unified",
            spec_decode_enable=bool(dc.get("spec_decode_enable", True)),
            spec_decode_depth=clamp_depth(dc.get("spec_decode_depth", 4)),
            spec_mode=valid_spec_mode(dc.get("spec_mode", "v2_clean")),
            spec_v3_block_size=clamp_block_size(dc.get("spec_v3_block_size", 8)),
            draft_temperature=clamp01(dc.get("draft_temperature", 0.8), 0.8),
            kv_compression_enable=bool(dc.get("kv_compression_enable", False)),
            kv_compression_ratio=max(0.1, min(1.0, float(dc.get("kv_compression_ratio", 1.0) or 1.0))),
            decode_batch_priority=clamp_priority(dc.get("decode_batch_priority", 2)),
            early_stop_confidence=clamp01(dc.get("early_stop_confidence", 0.0)),
            cuda_graph_hint=bool(dc.get("cuda_graph_hint", True)),
            adaptive_depth_override=bool(dc.get("adaptive_depth_override", False)),
        )
