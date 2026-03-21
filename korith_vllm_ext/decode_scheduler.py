from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)
_TRUTHY = ("1", "true", "yes", "on")

# ---------------------------------------------------------------------------
# Dynamo integration helpers — all guarded by _DYNAMO_WORKER flag
# ---------------------------------------------------------------------------

_DYNAMO_WORKER: bool = any(
    os.environ.get(k, "").strip().lower() in _TRUTHY
    for k in ("DYNAMO_WORKER", "DYNAMO_ENABLED", "DYNAMO_WORKER_MODE")
)

if _DYNAMO_WORKER:
    logger.info("KorithDecodeScheduler running in Dynamo worker mode")


def _get_amf_coordinator_client() -> Any:
    """Lazily import and return AmfCoordinatorClient (avoids hard dep when Dynamo absent)."""
    try:
        from platform.runtime.amf_coordinator_client import AmfCoordinatorClient  # type: ignore[import]
        url = os.environ.get("AMF_COORDINATOR_URL", "")
        if url:
            return AmfCoordinatorClient(url)
    except ImportError:
        pass
    return None


_AMF_CLIENT: Any = _get_amf_coordinator_client() if _DYNAMO_WORKER else None


def _get_nats_client() -> Any:
    """Return a lazy wrapper for NATS publish (best-effort, non-critical)."""
    if not _DYNAMO_WORKER:
        return None
    try:
        import nats  # type: ignore[import]
        return nats
    except ImportError:
        return None


_NATS_LIB: Any = _get_nats_client()

# Async NATS connection is managed per-process; we keep a module-level handle.
_nats_conn: Any = None
_nats_lock_imported = False


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in _TRUTHY


class KorithDecodeScheduler(Scheduler):
    """
    Decode-oriented vLLM scheduler extension.

    Goals:
    - Prefer active decode requests over prefill-heavy requests.
    - Keep decode batches shape- and priority-aware by reordering running queue.
    - Apply adaptive token budget to reduce decode-step variance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._korith_decode_first = _env_truthy("KORITH_VLLM_DECODE_FIRST", default=True)
        self._korith_adaptive_budget = _env_truthy("KORITH_VLLM_ADAPTIVE_BUDGET", default=True)
        self._korith_budget_scale_decode = max(
            0.25,
            min(1.0, float(os.environ.get("KORITH_VLLM_BUDGET_SCALE_DECODE", "0.75") or 0.75)),
        )
        self._korith_decode_ratio_threshold = max(
            0.0,
            min(1.0, float(os.environ.get("KORITH_VLLM_DECODE_RATIO_THRESHOLD", "0.6") or 0.6)),
        )
        self._korith_min_scheduled_tokens = max(
            1,
            int(os.environ.get("KORITH_VLLM_MIN_SCHEDULED_TOKENS", "128") or 128),
        )
        self._korith_shape_boost = max(
            0.0,
            min(5.0, float(os.environ.get("KORITH_VLLM_SHAPE_BOOST", "0.15") or 0.15)),
        )
        self._korith_starvation_s = max(
            0.0,
            float(os.environ.get("KORITH_VLLM_STARVATION_S", "0.75") or 0.75),
        )
        self._korith_lane_bias = {
            "SPEC_HIT": max(0.1, float(os.environ.get("KORITH_VLLM_LANE_BIAS_SPEC_HIT", "1.25") or 1.25)),
            "HIT": max(0.1, float(os.environ.get("KORITH_VLLM_LANE_BIAS_HIT", "1.1") or 1.1)),
            "SPEC_MISS": max(0.1, float(os.environ.get("KORITH_VLLM_LANE_BIAS_SPEC_MISS", "0.95") or 0.95)),
            "MISS": max(0.1, float(os.environ.get("KORITH_VLLM_LANE_BIAS_MISS", "1.0") or 1.0)),
        }
        self._korith_waiting_pressure_threshold = max(
            0,
            int(os.environ.get("KORITH_VLLM_WAITING_PRESSURE_THRESHOLD", "8") or 8),
        )
        self._korith_waiting_retune = _env_truthy("KORITH_VLLM_WAITING_RETUNE", default=True)
        self._korith_waiting_retune_max = max(
            1,
            int(os.environ.get("KORITH_VLLM_WAITING_RETUNE_MAX", "128") or 128),
        )
        self._korith_prefill_penalty = max(
            0.0,
            min(3.0, float(os.environ.get("KORITH_VLLM_PREFILL_PENALTY", "0.2") or 0.2)),
        )
        self._korith_short_decode_bonus = max(
            0.0,
            min(3.0, float(os.environ.get("KORITH_VLLM_SHORT_DECODE_BONUS", "0.15") or 0.15)),
        )
        self._korith_spec_score_per_k = max(
            0.0,
            min(0.10, float(os.environ.get("KORITH_VLLM_SPEC_SCORE_PER_K", "0.02") or 0.02)),
        )
        self._korith_spec_score_cap = max(
            0.0,
            min(0.50, float(os.environ.get("KORITH_VLLM_SPEC_SCORE_CAP", "0.15") or 0.15)),
        )
        self._korith_decode_full_budget_on_hit = _env_truthy("KORITH_VLLM_DECODE_FULL_BUDGET_ON_HIT", default=True)
        self._korith_sched_trace_path = str(os.environ.get("KORITH_VLLM_SCHED_TRACE_PATH", "")).strip()
        self._korith_sched_trace_every = max(
            1,
            int(os.environ.get("KORITH_VLLM_SCHED_TRACE_EVERY", "32") or 32),
        )
        self._korith_sched_calls = 0
        self._korith_sched_total_ms = 0.0
        self._korith_sched_max_ms = 0.0
        self._korith_base_max_num_scheduled_tokens = int(self.max_num_scheduled_tokens)
        logger.info(
            "KorithDecodeScheduler enabled decode_first=%s adaptive_budget=%s base_tokens=%d full_budget_on_hit=%s",
            self._korith_decode_first,
            self._korith_adaptive_budget,
            self._korith_base_max_num_scheduled_tokens,
            self._korith_decode_full_budget_on_hit,
        )

    @staticmethod
    def _request_extra_args(req: Any) -> dict[str, Any]:
        sampling = getattr(req, "sampling_params", None)
        if sampling is None:
            return {}
        extra = getattr(sampling, "extra_args", None)
        return extra if isinstance(extra, dict) else {}

    @staticmethod
    def _xarg_str(req: Any, key: str, default: str = "") -> str:
        value = KorithDecodeScheduler._request_extra_args(req).get(key, default)
        if value is None:
            return str(default)
        return str(value)

    @staticmethod
    def _xarg_int(req: Any, key: str, default: int = 0) -> int:
        value = KorithDecodeScheduler._request_extra_args(req).get(key, default)
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return int(default)

    @staticmethod
    def _remaining_decode_tokens(req: Any) -> int:
        max_tokens = int(getattr(req, "max_tokens", 0) or 0)
        out_tokens = int(getattr(req, "num_output_tokens", 0) or 0)
        return max(0, max_tokens - out_tokens)

    @staticmethod
    def _cached_tokens(req: Any) -> int:
        try:
            return max(0, int(getattr(req, "num_cached_tokens", 0) or 0))
        except Exception:
            return 0

    def _shape_key(self, req: Any) -> str:
        xarg_key = self._xarg_str(req, "korith_shape_key", "").strip()
        if xarg_key:
            return xarg_key
        # Fallback: coarse shape bucket based on prompt/output lengths.
        prompt_len = int(getattr(req, "num_prompt_tokens", 0) or 0)
        out_tokens = int(getattr(req, "num_output_tokens", 0) or 0)
        return f"{(prompt_len // 256) * 256}:{(out_tokens // 64) * 64}"

    def _lane_key(self, req: Any) -> str:
        lane = self._xarg_str(req, "korith_lane", "").strip().upper()
        if lane:
            return lane
        # Infer lane from priority if explicit lane metadata is missing.
        priority = int(getattr(req, "priority", 0) or 0)
        if priority <= 0:
            return "SPEC_HIT"
        if priority == 1:
            return "HIT"
        if priority == 2:
            return "SPEC_MISS"
        return "MISS"

    def _decode_budget_hint(self, req: Any) -> int:
        hinted = self._xarg_int(req, "korith_decode_budget_tokens", 0)
        if hinted > 0:
            return hinted
        target = self._xarg_int(req, "korith_target_tokens", 0)
        if target > 0:
            return target
        return int(getattr(req, "max_tokens", 0) or 0)

    def _request_replay_state(self, req: Any) -> str:
        return self._xarg_str(req, "korith_replay_state", "").strip().lower()

    def _request_replay_local(self, req: Any) -> bool:
        val = self._xarg_str(req, "korith_replay_local", "").strip().lower()
        return val in ("1", "true", "yes", "on")

    def _request_prompt_tokens(self, req: Any) -> int:
        hinted = self._xarg_int(req, "korith_prompt_tokens", 0)
        if hinted > 0:
            return hinted
        return int(getattr(req, "num_prompt_tokens", 0) or 0)

    def _dominant_decode_shape(self) -> str:
        counts: dict[str, int] = {}
        for req in self.running:
            if int(getattr(req, "num_output_tokens", 0) or 0) <= 0:
                continue
            key = self._shape_key(req)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _request_score(self, req: Any, *, dominant_shape: str) -> tuple[float, float, float, float, float, float]:
        now = time.time()
        arrival = float(getattr(req, "arrival_time", now) or now)
        wait_s = max(0.0, now - arrival)
        starvation_bonus = 1.0 if wait_s >= self._korith_starvation_s else 0.0

        cached_prefix = self._cached_tokens(req)
        in_decode = 1 if (int(getattr(req, "num_output_tokens", 0) or 0) > 0 or cached_prefix > 0) else 0
        remaining = self._remaining_decode_tokens(req)
        priority = int(getattr(req, "priority", 0) or 0)
        lane = self._lane_key(req)
        lane_bias = float(self._korith_lane_bias.get(lane, 1.0) or 1.0)

        shape_key_local = self._shape_key(req)
        shape_bonus = 1.0
        if dominant_shape and shape_key_local == dominant_shape:
            shape_bonus += self._korith_shape_boost

        replay_state = self._request_replay_state(req)
        replay_bonus = 1.0
        if replay_state == "restore":
            replay_bonus = 1.06
        elif replay_state == "hit":
            replay_bonus = 1.03
        if self._request_replay_local(req):
            replay_bonus *= 1.06

        spec_enabled = self._xarg_int(req, "korith_spec_enabled", 0) > 0
        spec_k = max(0, self._xarg_int(req, "korith_spec_k", 0))
        spec_bonus = 1.0
        if spec_enabled and in_decode > 0:
            spec_bonus += min(self._korith_spec_score_cap, float(spec_k) * self._korith_spec_score_per_k)

        prompt_tokens = self._request_prompt_tokens(req)
        prefill_penalty = 1.0
        if in_decode <= 0 and prompt_tokens > 1024 and cached_prefix <= 0:
            prefill_penalty = max(0.5, 1.0 - self._korith_prefill_penalty)

        short_decode_bonus = 1.0
        if in_decode > 0 and remaining <= 128:
            short_decode_bonus += self._korith_short_decode_bonus

        decode_score = (
            (100000 - remaining)
            * shape_bonus
            * lane_bias
            * replay_bonus
            * spec_bonus
            * prefill_penalty
            * short_decode_bonus
        )
        return (starvation_bonus, float(in_decode), float(-priority), lane_bias, replay_bonus, decode_score)

    def _retune_waiting_priorities(self, *, dominant_shape: str) -> None:
        if not self._korith_waiting_retune:
            return
        waiting_depth = len(self.waiting) if self.waiting is not None else 0
        if waiting_depth <= 1 or waiting_depth > self._korith_waiting_retune_max:
            return
        waiting_reqs = list(self.waiting)
        if not waiting_reqs:
            return

        def dyn_priority(req: Any) -> int:
            lane = self._lane_key(req)
            lane_map = {"SPEC_HIT": 0, "HIT": 1, "SPEC_MISS": 2, "MISS": 3}
            p = int(lane_map.get(lane, 3))
            if self._request_replay_state(req) in ("restore", "hit"):
                p -= 1
            if self._request_replay_local(req):
                p -= 1
            if int(getattr(req, "num_output_tokens", 0) or 0) > 0:
                p -= 1
            if dominant_shape and self._shape_key(req) == dominant_shape:
                p -= 1
            now = time.time()
            arrival = float(getattr(req, "arrival_time", now) or now)
            if (now - arrival) >= self._korith_starvation_s:
                p -= 2
            return max(0, min(100, int(p)))

        for req in waiting_reqs:
            try:
                req.priority = dyn_priority(req)
            except Exception:
                continue

        try:
            self.waiting.remove_requests(waiting_reqs)
            for req in waiting_reqs:
                self.waiting.add_request(req)
        except Exception:
            # If queue internals change across vLLM versions, fail open.
            return

    def _korith_reorder_running(self) -> None:
        if not self._korith_decode_first or len(self.running) <= 1:
            return

        dominant_shape = self._dominant_decode_shape()
        self._retune_waiting_priorities(dominant_shape=dominant_shape)

        def score(req: Any) -> tuple[float, float, float, float, float, float]:
            return self._request_score(req, dominant_shape=dominant_shape)

        try:
            sorted_running = sorted(self.running, key=score, reverse=True)
            self.running[:] = sorted_running
        except Exception:
            # Fail open: if vLLM's running queue type does not support slice
            # assignment or sorting (e.g. after an upstream API change), leave
            # the order unchanged rather than raising inside the scheduler hot path.
            pass

    def _korith_effective_budget(self) -> int:
        base = max(1, int(self._korith_base_max_num_scheduled_tokens))
        if not self._korith_adaptive_budget:
            return base
        running = list(self.running)
        if not running:
            return base
        if self._korith_decode_full_budget_on_hit:
            cached_hit = sum(1 for req in running if self._cached_tokens(req) > 0)
            if cached_hit > 0:
                return base
        decode_count = sum(1 for req in running if int(getattr(req, "num_output_tokens", 0) or 0) > 0)
        decode_ratio = float(decode_count) / float(max(1, len(running)))
        hints = [self._decode_budget_hint(req) for req in running]
        hints = [h for h in hints if h > 0]
        if decode_ratio < self._korith_decode_ratio_threshold:
            budget = base
        else:
            budget = int(base * self._korith_budget_scale_decode)
            budget = max(self._korith_min_scheduled_tokens, min(base, budget))
        if hints:
            hints.sort()
            median_hint = int(hints[len(hints) // 2])
            median_hint = max(self._korith_min_scheduled_tokens, min(base, median_hint))
            budget = min(int(budget), int(median_hint))
        waiting_depth = len(self.waiting) if self.waiting is not None else 0
        if waiting_depth >= self._korith_waiting_pressure_threshold and budget > self._korith_min_scheduled_tokens:
            budget = int(max(self._korith_min_scheduled_tokens, round(float(budget) * 0.9)))
        return int(max(self._korith_min_scheduled_tokens, min(base, budget)))

    def _emit_sched_trace(self, elapsed_ms: float) -> None:
        self._korith_sched_calls += 1
        self._korith_sched_total_ms += float(elapsed_ms)
        self._korith_sched_max_ms = max(self._korith_sched_max_ms, float(elapsed_ms))
        if not self._korith_sched_trace_path:
            return
        if (self._korith_sched_calls % self._korith_sched_trace_every) != 0:
            return
        avg = self._korith_sched_total_ms / float(max(1, self._korith_sched_calls))
        waiting_depth = 0
        try:
            waiting_depth = int(len(self.waiting) if self.waiting is not None else 0)
        except Exception:
            try:
                waiting_depth = int(self.waiting.qsize() if self.waiting is not None else 0)  # type: ignore[attr-defined]
            except Exception:
                waiting_depth = 0

        payload = {
            "calls": int(self._korith_sched_calls),
            "avg_ms": float(avg),
            "max_ms": float(self._korith_sched_max_ms),
            "last_ms": float(elapsed_ms),
            "running": int(len(self.running)),
            "waiting": int(waiting_depth),
            "timestamp_s": float(time.time()),
        }
        path = Path(self._korith_sched_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Dynamo integration — only active when DYNAMO_WORKER=true
    # ------------------------------------------------------------------

    def _dynamo_check_amf_hit(self, req: Any) -> bool:
        """Return True if AMF coordinator reports a snapshot for this request.

        When True, elevates the request to SPEC_HIT lane (highest priority)
        because prefill will be skipped via AMF restore.
        """
        if not _DYNAMO_WORKER or _AMF_CLIENT is None:
            return False
        prefix_hash = self._xarg_str(req, "korith_prefix_hash", "").strip()
        tenant_id = self._xarg_str(req, "korith_tenant_id", "default").strip() or "default"
        if not prefix_hash:
            return False
        try:
            rows = _AMF_CLIENT.lookup(prefix_hash=prefix_hash, tenant_id=tenant_id)
            return bool(rows)
        except Exception:
            return False

    def _dynamo_apply_amf_priority(self, req: Any) -> bool:
        """Elevate request to SPEC_HIT lane if AMF has a snapshot.

        Returns True if AMF hit was detected and priority was updated.
        """
        if not _DYNAMO_WORKER:
            return False
        if self._xarg_int(req, "korith_amf_hit", 0) > 0:
            return True  # Already marked.
        if not self._dynamo_check_amf_hit(req):
            return False

        # Mark AMF hit so subsequent scoring uses SPEC_HIT lane.
        try:
            extra = self._request_extra_args(req)
            extra["korith_amf_hit"] = 1
            extra["korith_lane"] = "SPEC_HIT"
            extra["korith_replay_state"] = "restore"
        except Exception:
            pass
        return True

    def _dynamo_emit_metadata(self, req: Any, amf_hit: bool) -> None:
        """Attach Dynamo-compatible metadata to request extra_args.

        Emits korith_lane, korith_replay_state, and korith_amf_hit so that
        Dynamo's frontend can consume them for routing feedback.
        """
        if not _DYNAMO_WORKER:
            return
        try:
            extra = self._request_extra_args(req)
            extra.setdefault("korith_lane", self._lane_key(req))
            extra.setdefault("korith_replay_state", self._request_replay_state(req) or "miss")
            extra["korith_amf_hit"] = 1 if amf_hit else 0
        except Exception:
            pass

    def _dynamo_emit_restore_event(
        self,
        req: Any,
        restore_ms: float,
        saved_ms: float,
        tokens_skipped: int,
    ) -> None:
        """Emit amf.restore.completed event to NATS (best-effort, non-blocking).

        This feeds back into AMF Coordinator and the Dynamo router so they can
        learn from actual restore outcomes.
        """
        if not _DYNAMO_WORKER or _NATS_LIB is None:
            return
        try:
            import asyncio

            worker_id = os.environ.get("KORITH_WORKER_ID", "")
            prefix_hash = self._xarg_str(req, "korith_prefix_hash", "")
            payload = json.dumps({
                "worker_id": worker_id,
                "prefix_hash": prefix_hash,
                "restore_ms": restore_ms,
                "saved_ms": saved_ms,
                "tokens_skipped": tokens_skipped,
                "timestamp_ms": time.time() * 1000,
            }, ensure_ascii=False).encode("utf-8")

            nats_url = os.environ.get("DYNAMO_NATS_URL", "nats://localhost:4222")

            async def _publish() -> None:
                global _nats_conn
                try:
                    if _nats_conn is None or _nats_conn.is_closed:
                        _nats_conn = await _NATS_LIB.connect(nats_url)
                    await _nats_conn.publish("amf.restore.completed", payload)
                except Exception as exc:
                    logger.debug("NATS publish amf.restore.completed failed: %s", exc)

            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_publish())
            else:
                loop.run_until_complete(_publish())
        except Exception as exc:
            logger.debug("_dynamo_emit_restore_event error: %s", exc)

    # ------------------------------------------------------------------
    # schedule() — main hot path
    # ------------------------------------------------------------------

    def schedule(self):
        t0 = time.perf_counter()
        self._korith_reorder_running()

        # Dynamo: check AMF hits for waiting prefill-heavy requests.
        if _DYNAMO_WORKER:
            for req in list(self.waiting) if self.waiting is not None else []:
                try:
                    prompt_tokens = self._request_prompt_tokens(req)
                    if prompt_tokens >= 1024:
                        self._dynamo_apply_amf_priority(req)
                except Exception:
                    pass

        old_budget = int(self.max_num_scheduled_tokens)
        self.max_num_scheduled_tokens = int(self._korith_effective_budget())
        try:
            result = super().schedule()

            # Dynamo: emit metadata and restore events on scheduled requests.
            if _DYNAMO_WORKER and result is not None:
                try:
                    scheduled_reqs = getattr(result, "scheduled_requests", None) or []
                    for req in scheduled_reqs:
                        amf_hit = self._xarg_int(req, "korith_amf_hit", 0) > 0
                        self._dynamo_emit_metadata(req, amf_hit=amf_hit)
                        if amf_hit and self._xarg_str(req, "korith_replay_state", "") == "restore":
                            tokens_skipped = self._request_prompt_tokens(req)
                            self._dynamo_emit_restore_event(
                                req,
                                restore_ms=float(self._xarg_int(req, "korith_restore_ms", 0)),
                                saved_ms=0.0,
                                tokens_skipped=tokens_skipped,
                            )
                except Exception:
                    pass

            return result
        finally:
            self.max_num_scheduled_tokens = old_budget
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._emit_sched_trace(elapsed_ms)
