"""OpenClaw Proxy Bridge — Route OpenClaw agent LLM calls through AMF.

OpenClaw agents make standard OpenAI-compatible API calls. Axropus IS
that endpoint. This bridge adds AMF-specific optimizations:

1. Automatic prefix pinning for system prompts
2. Cross-step prefix sharing within a task
3. Step-level metrics collection for claw monitoring
4. NemoClaw compatibility mode

Usage:
    bridge = OpenClawBridge()
    bridge.register_claw(claw_config)

    # On each LLM call from OpenClaw:
    annotated_req = bridge.on_request(request, claw_id)
    # ... route through AMF inference ...
    annotated_resp = bridge.on_response(response, claw_id, step_num)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClawSession:
    """Active session for a running claw."""
    claw_id: str
    task_id: str
    model_id: str
    system_prompt_hash: str
    step_count: int = 0
    total_tokens: int = 0
    tokens_saved: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)


class OpenClawBridge:
    """Bridge between OpenClaw agent calls and Axropus AMF inference.

    Intercepts LLM requests from OpenClaw agents, adds AMF optimization
    hints, and collects step-level metrics for monitoring.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ClawSession] = {}
        self._claw_configs: Dict[str, Dict] = {}
        self._prefix_hashes: Dict[str, str] = {}  # claw_id → system prompt hash

    def register_claw(self, claw_config: Dict[str, Any]) -> str:
        """Register a claw configuration. Returns claw_id."""
        claw_id = claw_config.get("id", hashlib.md5(
            claw_config.get("name", "").encode()
        ).hexdigest()[:12])

        self._claw_configs[claw_id] = claw_config

        # Pre-compute system prompt hash for prefix sharing
        system_prompt = claw_config.get("system_prompt", "")
        if system_prompt:
            self._prefix_hashes[claw_id] = hashlib.sha256(
                system_prompt.encode()
            ).hexdigest()[:16]

        logger.info("[OPENCLAW] Registered claw %s (model=%s, tools=%d)",
                     claw_id, claw_config.get("model_id"),
                     len(claw_config.get("tools", [])))
        return claw_id

    def start_task(self, claw_id: str, task_id: str) -> ClawSession:
        """Start a new task session for a claw."""
        config = self._claw_configs.get(claw_id, {})
        session = ClawSession(
            claw_id=claw_id,
            task_id=task_id,
            model_id=config.get("model_id", ""),
            system_prompt_hash=self._prefix_hashes.get(claw_id, ""),
        )
        self._sessions[task_id] = session
        logger.info("[OPENCLAW] Task %s started for claw %s", task_id, claw_id)
        return session

    def on_request(
        self, request: Dict[str, Any], claw_id: str, task_id: str
    ) -> Dict[str, Any]:
        """Annotate an incoming LLM request with AMF optimization hints.

        Adds:
        - AMF prefix sharing hint (same system prompt = cache hit)
        - Step tracking metadata
        - Prefix pinning for system prompts
        """
        session = self._sessions.get(task_id)
        if session:
            session.step_count += 1

        config = self._claw_configs.get(claw_id, {})
        amf_config = config.get("amf_config", {})

        # Add AMF hints to request
        request.setdefault("extra_body", {})
        request["extra_body"]["amf"] = {
            "enabled": True,
            "prefix_cache": True,
            "quant_mode": amf_config.get("quant_mode", "int4"),
            "pin_system_prompt": amf_config.get("pin_system_prompt", True),
            "claw_id": claw_id,
            "task_id": task_id,
            "step": session.step_count if session else 0,
            "prefix_hash": self._prefix_hashes.get(claw_id),
        }

        return request

    def on_response(
        self,
        response: Dict[str, Any],
        claw_id: str,
        task_id: str,
        amf_metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Process response and collect step metrics."""
        session = self._sessions.get(task_id)
        if not session:
            return response

        # Extract token usage
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total = input_tokens + output_tokens

        session.total_tokens += total

        # Track AMF metrics for this step
        step_data = {
            "step_number": session.step_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "amf_hit": False,
            "tokens_saved": 0,
            "restore_ms": 0,
            "timestamp": time.monotonic(),
        }

        if amf_metrics:
            step_data["amf_hit"] = amf_metrics.get("hit", False)
            step_data["tokens_saved"] = amf_metrics.get("tokens_saved", 0)
            step_data["restore_ms"] = amf_metrics.get("restore_ms", 0)

            if step_data["amf_hit"]:
                session.prefix_hits += 1
                session.tokens_saved += step_data["tokens_saved"]
            else:
                session.prefix_misses += 1

        session.steps.append(step_data)

        # Add AMF metrics to response
        response["amf_metrics"] = step_data

        return response

    def end_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """End a task and return summary metrics."""
        session = self._sessions.pop(task_id, None)
        if not session:
            return None

        elapsed_ms = (time.monotonic() - session.started_at) * 1000.0
        total_steps = session.prefix_hits + session.prefix_misses
        reuse_rate = session.prefix_hits / max(total_steps, 1)

        summary = {
            "task_id": task_id,
            "claw_id": session.claw_id,
            "total_steps": session.step_count,
            "total_tokens": session.total_tokens,
            "tokens_saved": session.tokens_saved,
            "prefix_reuse_rate": reuse_rate,
            "duration_ms": elapsed_ms,
            "steps": session.steps,
        }

        logger.info(
            "[OPENCLAW] Task %s completed: %d steps, %d tokens saved, %.0f%% reuse",
            task_id, session.step_count, session.tokens_saved, reuse_rate * 100,
        )
        return summary

    def get_claw_metrics(self, claw_id: str) -> Dict[str, Any]:
        """Get aggregate metrics for a claw across all active sessions."""
        active_sessions = [
            s for s in self._sessions.values() if s.claw_id == claw_id
        ]

        total_tokens = sum(s.total_tokens for s in active_sessions)
        total_saved = sum(s.tokens_saved for s in active_sessions)
        total_steps = sum(s.step_count for s in active_sessions)
        total_hits = sum(s.prefix_hits for s in active_sessions)
        total_total = sum(s.prefix_hits + s.prefix_misses for s in active_sessions)

        return {
            "claw_id": claw_id,
            "active_tasks": len(active_sessions),
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "tokens_saved": total_saved,
            "prefix_reuse_rate": total_hits / max(total_total, 1),
            "savings_rate": total_saved / max(total_tokens, 1),
        }
