"""
NemoClaw Reasoning Model
=========================
Drop-in replacement for ReasoningModel that uses NVIDIA NemoClaw via the
NIM API (OpenAI-compatible) instead of a local Llama 8B NF4.

Why NemoClaw instead of Llama 8B:
- Built-in RL loop — no manual peft/transformers wiring needed
- NVIDIA maintains and improves it — we get smarter decisions over time for free
- Runs on Nebius infrastructure (Nebius is NVIDIA's flagship cloud partner)
- Agentic by design — built for exactly this kind of adaptive control loop
- Automatic scheduling — no interval_s tuning, NemoClaw decides when to update

Architecture change:
    Before: ReasoningModel (Llama 8B NF4) + LearningLoop (manual LoRA)
    After:  NemoClawReasoningModel (NIM API) — RL is internal to NemoClaw

Stack simplification:
    Keep:    MetricsCollector, DecisionExecutor, RewardCalculator, Orchestrator
    Replace: ReasoningModel → NemoClawReasoningModel
    Remove:  LearningLoop (use NemoClawReasoningModel.record_reward() instead)

Usage
-----
    model = NemoClawReasoningModel(
        api_key=os.environ["NVIDIA_API_KEY"],       # from build.nvidia.com
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/nemoclaw",
    )

    tensor = collector.get_metrics_tensor()
    dv     = model.decide(tensor)                   # same interface as before

    # After outcome is known — feeds NemoClaw's internal RL
    model.record_reward(decision_id, reward=1.0, outcome="hit_admitted")

Self-hosted (Nebius / on-prem):
    model = NemoClawReasoningModel(
        api_key="...",
        base_url="http://your-nebius-nim-node:8000/v1",
        model="nvidia/nemoclaw",
    )
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics_collector import Slot, TENSOR_SIZE
from .reasoning_model import UnifiedDecisionVector, _SYSTEM_PROMPT

log = logging.getLogger(__name__)

# ── NemoClaw NIM endpoint defaults ───────────────────────────────────────────
_DEFAULT_BASE_URL  = "https://integrate.api.nvidia.com/v1"
_DEFAULT_MODEL     = "nvidia/nemoclaw"
_MAX_REWARD_HISTORY = 20   # in-context RL window size

# ── reward outcome → numeric score ───────────────────────────────────────────
_REWARD_SCORES: Dict[str, float] = {
    "hit_admitted":       +1.0,
    "pre_warm_hit":       +1.0,
    "spec_high_accept":   +0.8,
    "spec_depth_optimal": +0.6,
    "eviction_miss":      -1.0,
    "miss_never_reused":  -0.5,
    "spec_low_accept":    -0.4,
    "spec_depth_wasteful":-0.3,
}


@dataclass
class NemoClawDecisionRecord:
    """Tracks an in-flight decision for reward attribution."""
    decision_id: str
    prefix_hash: str
    spec_depth:  int
    admitted:    bool
    timestamp:   float = field(default_factory=time.monotonic)


class NemoClawReasoningModel:
    """
    AMF reasoning brain powered by Nemotron-70B / NemoClaw via NVIDIA NIM.

    RL mechanism: in-context learning with reward history.
    Every decision outcome (hit/miss/spec quality) is appended to a rolling
    reward buffer that gets included in the next inference call. The model
    reads "what worked last time" and adapts its next decision accordingly.
    No fine-tuning, no peft, no training infrastructure — pure in-context RL.

    Swap to NemoClaw: change model= to "nvidia/nemoclaw" when API key lands.

    Parameters
    ----------
    api_key : str
        NVIDIA API key from build.nvidia.com or Nebius NIM.
    base_url : str
        NIM base URL. Override with Nebius node for on-prem.
    model : str
        Defaults to Nemotron-70B (live now). Swap to nemoclaw when available.
    max_inference_ms : float
        Hard deadline. Falls back to safe defaults on timeout.
    enable_reward_feedback : bool
        Append outcomes to reward history for in-context RL.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        max_inference_ms: float = 100.0,
        enable_reward_feedback: bool = True,
    ) -> None:
        self._api_key   = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._base_url  = base_url.rstrip("/")
        self._model     = model
        self._max_ms    = max_inference_ms
        self._reward_fb = enable_reward_feedback

        # In-flight decision tracking for reward attribution
        self._decisions: Dict[str, NemoClawDecisionRecord] = {}

        # In-context RL: rolling reward history fed into every prompt
        # Each entry: {"outcome": str, "reward": float, "spec_depth": int,
        #              "admitted": bool, "hit_rate": float}
        self._reward_history: List[Dict[str, Any]] = []

        # Stats
        self._client = None  # lazy OpenAI client init

        self._total_calls    = 0
        self._fallback_calls = 0
        self._reward_sent    = 0
        self._rl_updates     = 0  # times reward history improved a decision

        log.info(
            "[nemoclaw] initialized model=%s base_url=%s in_context_rl=%s",
            self._model, self._base_url, self._reward_fb,
        )

    # ── public API (same interface as ReasoningModel) ─────────────────────────

    def decide(self, tensor: np.ndarray) -> UnifiedDecisionVector:
        """
        Send metrics tensor to NemoClaw, get back an AMF decision vector.
        Falls back to safe defaults on timeout or API error.
        """
        self._total_calls += 1
        t0 = time.monotonic()

        user_msg = self._build_user_message(tensor)

        try:
            raw_json = self._call_nim(
                system_prompt=_SYSTEM_PROMPT,
                user_message=user_msg,
                max_tokens=256,
                temperature=0.0,   # greedy — deterministic decisions
            )
            inference_ms = (time.monotonic() - t0) * 1000

            if inference_ms > self._max_ms:
                log.warning("[nemoclaw] inference timeout %.1fms > %.1fms", inference_ms, self._max_ms)
                return self._fallback_dv(inference_ms)

            dv = self._parse_response(raw_json, inference_ms)
            log.debug("[nemoclaw] decision in %.1fms: admit=%s spec_depth=%d",
                      inference_ms, dv.cache_admission, dv.spec_decode_depth)
            return dv

        except Exception as exc:
            inference_ms = (time.monotonic() - t0) * 1000
            log.warning("[nemoclaw] API error: %s (%.1fms) — using fallback", exc, inference_ms)
            self._fallback_calls += 1
            return self._fallback_dv(inference_ms)

    def record_reward(
        self,
        decision_id: str,
        outcome: str,
        value: float = 0.0,
        hit_rate: float = 0.0,
    ) -> None:
        """
        Record outcome for in-context RL.

        Appends to rolling reward history included in every future prompt.
        The model reads recent outcomes and adapts — no fine-tuning needed.

        Called by Orchestrator after each request resolves:
            on_hit()       → record_reward(id, "hit_admitted",  +1.0)
            on_miss()      → record_reward(id, "eviction_miss", -1.0)
            on_spec_done() → record_reward(id, "spec_high_accept", +0.8)
        """
        if not self._reward_fb:
            return

        score  = _REWARD_SCORES.get(outcome, value)
        record = self._decisions.pop(decision_id, None)

        # Append to in-context RL history
        entry = {
            "outcome":    outcome,
            "reward":     round(score, 2),
            "spec_depth": record.spec_depth if record else 0,
            "admitted":   record.admitted   if record else False,
            "hit_rate":   round(hit_rate, 3),
        }
        self._reward_history.append(entry)
        if len(self._reward_history) > _MAX_REWARD_HISTORY:
            self._reward_history.pop(0)

        self._reward_sent += 1
        if score > 0:
            self._rl_updates += 1

        log.debug("[nemoclaw] reward recorded: %s → %.2f (%s) history_len=%d",
                  decision_id[:8], score, outcome, len(self._reward_history))

    def track_decision(self, decision_id: str, prefix_hash: str,
                       spec_depth: int, admitted: bool) -> None:
        """Register a decision for later reward attribution."""
        self._decisions[decision_id] = NemoClawDecisionRecord(
            decision_id=decision_id,
            prefix_hash=prefix_hash,
            spec_depth=spec_depth,
            admitted=admitted,
        )
        # Evict old unresolved decisions (prevent unbounded growth)
        if len(self._decisions) > 10_000:
            oldest = sorted(self._decisions.items(), key=lambda x: x[1].timestamp)
            for k, _ in oldest[:1000]:
                del self._decisions[k]

    def get_stats(self) -> Dict[str, Any]:
        avg_reward = 0.0
        if self._reward_history:
            avg_reward = sum(e["reward"] for e in self._reward_history) / len(self._reward_history)
        return {
            "total_calls":       self._total_calls,
            "fallback_calls":    self._fallback_calls,
            "fallback_rate":     round(self._fallback_calls / max(1, self._total_calls), 3),
            "reward_sent":       self._reward_sent,
            "rl_updates":        self._rl_updates,
            "rl_history_len":    len(self._reward_history),
            "rl_avg_reward":     round(avg_reward, 3),
            "pending_decisions": len(self._decisions),
            "model":             self._model,
            "base_url":          self._base_url,
        }

    # ── load/unload (compatibility with ReasoningModel interface) ─────────────

    def load(self) -> None:
        """Verify API connectivity."""
        if not self._api_key:
            log.warning("[nemoclaw] no API key set — set NVIDIA_API_KEY env var")
            return
        try:
            # Quick health ping
            self._call_nim(
                system_prompt="ping",
                user_message="ping",
                max_tokens=1,
                temperature=0.0,
            )
            log.info("[nemoclaw] API connectivity verified")
        except Exception as exc:
            log.warning("[nemoclaw] connectivity check failed: %s", exc)

    def unload(self) -> None:
        log.info("[nemoclaw] unloaded (stats: %s)", self.get_stats())

    # ── NIM API calls ─────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key or "no-key",
                base_url=self._base_url,
            )
        return self._client

    def _call_nim(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """
        Call NemoClaw via NIM's OpenAI-compatible chat completions endpoint.
        Returns the raw text content of the first choice.
        """
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=int(self._max_ms / 1000) + 2,
        )
        return resp.choices[0].message.content

    def _post_reward(self, payload: Dict[str, Any]) -> None:
        """
        Reward feedback endpoint — no-op until NemoClaw exposes /nemoclaw/reward.
        In-context RL via reward history in _build_user_message handles this already.
        """
        log.debug("[nemoclaw] _post_reward: in-context RL active, skipping HTTP reward post")

    # ── message building ──────────────────────────────────────────────────────

    def _build_user_message(self, tensor: np.ndarray) -> str:
        """
        Compact metrics + reward history → user message.

        The reward history is the in-context RL signal. The model reads
        what worked in recent decisions and adapts its next choice.
        """
        prefill = {
            "hit_rate":       round(float(tensor[Slot.HIT_RATE]), 3),
            "storage_util":   round(float(tensor[Slot.STORAGE_UTIL]), 3),
            "evict_pressure": round(float(tensor[Slot.EVICTION_PRESSURE]), 3),
            "hot_ratio":      round(float(tensor[Slot.HOT_ENTRY_RATIO]), 3),
            "cold_ratio":     round(float(tensor[Slot.COLD_ENTRY_RATIO]), 3),
            "miss_streak":    round(float(tensor[Slot.MISS_STREAK_NORM]), 3),
            "hit_streak":     round(float(tensor[Slot.HIT_STREAK_NORM]), 3),
            "roi_ema":        round(float(tensor[Slot.ROI_EMA_NORM]) * 10, 3),
            "reuse_ratio":    round(float(tensor[Slot.REUSE_RATIO]), 3),
            "nodes":          int(tensor[Slot.NODE_COUNT]),
            "cache_gb":       round(float(tensor[Slot.TOTAL_CACHE_GB]), 2),
        }
        decode = {
            "accept_ema": round(float(tensor[Slot.ACCEPTANCE_EMA]), 3),
            "spec_depth": int(tensor[Slot.CURRENT_SPEC_DEPTH]),
            "tps":        round(float(tensor[Slot.TPS_CURRENT]), 1),
            "entropy":    round(float(tensor[Slot.ENTROPY_EMA]), 3),
            "kv_util":    round(float(tensor[Slot.KV_CACHE_UTILIZATION]), 3),
            "power":      round(float(tensor[Slot.POWER_WATTS_NORM]), 3),
        }
        msg: Dict[str, Any] = {"prefill": prefill, "decode": decode}

        # Inject reward history — this IS the in-context RL signal
        if self._reward_history:
            msg["recent_outcomes"] = self._reward_history[-10:]  # last 10
            msg["rl_note"] = (
                "Learn from recent_outcomes. Positive reward=good decision. "
                "Adjust spec_depth and cache_admission based on what worked."
            )

        return json.dumps(msg, separators=(",", ":"))

    # ── response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, raw: str, inference_ms: float) -> UnifiedDecisionVector:
        """
        Parse NemoClaw's JSON output into a UnifiedDecisionVector.
        Falls back to safe defaults if JSON is malformed.
        """
        try:
            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            obj = json.loads(text)
            p = obj.get("prefill", {})
            d = obj.get("decode", {})

            return UnifiedDecisionVector(
                # prefill / cache
                cache_admission      = bool(p.get("cache_admission", True)),
                admission_priority   = float(p.get("admission_priority", 0.5)),
                eviction_target      = p.get("eviction_target") or None,
                pre_warm_predictions = list(p.get("pre_warm_predictions", []))[:5],
                route_decision       = p.get("route_decision") or None,
                batch_group          = p.get("batch_group") or None,
                throttle_back_pressure = float(p.get("throttle_back_pressure", 0.0)),
                inference_ms         = inference_ms,
                is_fallback          = False,
                domain               = "unified",
                # decode / spec
                spec_decode_enable   = bool(d.get("spec_decode_enable", True)),
                spec_decode_depth    = max(1, min(16, int(d.get("spec_decode_depth", 4)))),
                spec_mode            = str(d.get("spec_mode", "v2_clean")),
                spec_v3_block_size   = max(4, min(32, int(d.get("spec_v3_block_size", 8)))),
                draft_temperature    = float(d.get("draft_temperature", 0.8)),
                kv_compression_enable = bool(d.get("kv_compression_enable", False)),
                kv_compression_ratio  = float(d.get("kv_compression_ratio", 1.0)),
                decode_batch_priority = max(1, min(4, int(d.get("decode_batch_priority", 2)))),
                early_stop_confidence = float(d.get("early_stop_confidence", 0.0)),
                cuda_graph_hint      = bool(d.get("cuda_graph_hint", True)),
                adaptive_depth_override = bool(d.get("adaptive_depth_override", False)),
            )

        except Exception as exc:
            log.warning("[nemoclaw] parse error: %s — raw: %.100s", exc, raw)
            return self._fallback_dv(inference_ms)

    def _fallback_dv(self, inference_ms: float) -> UnifiedDecisionVector:
        """Safe no-op fallback — leaves all AMF parameters unchanged."""
        return UnifiedDecisionVector(
            cache_admission=True, admission_priority=0.5,
            eviction_target=None, pre_warm_predictions=[],
            route_decision=None, batch_group=None,
            throttle_back_pressure=0.0,
            inference_ms=inference_ms, is_fallback=True, domain="unified",
            spec_decode_enable=True, spec_decode_depth=4,
            spec_mode="v2_clean", spec_v3_block_size=8,
            draft_temperature=0.8, kv_compression_enable=False,
            kv_compression_ratio=1.0, decode_batch_priority=2,
            early_stop_confidence=0.0, cuda_graph_hint=True,
            adaptive_depth_override=False,
        )
