"""axropus.server — OpenAI-compatible HTTP frontend for AMF.

This is the server customers run alongside their existing vLLM engine.
It does three things:

    1. Exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint
       that proxies to vLLM (or falls back to an echo stub when vLLM is
       unavailable) but first runs every request through the AMF hook.

    2. Exposes ``/v1/completions`` as a thin wrapper around the chat
       endpoint so existing integrations can reuse the same cache.

    3. Exposes ``/metrics`` in Prometheus text format and ``/stats`` as
       a JSON snapshot of the router, pool, and LMCache state.

The endpoints are intentionally minimal — AMF is infrastructure, not a
consumer product. See §7 of the design doc for the CLI contract.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import AxropusConfig
from .metrics import MetricRegistry

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, Response
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    FastAPI = None        # type: ignore
    Header = None         # type: ignore
    HTTPException = None  # type: ignore
    Request = None        # type: ignore
    JSONResponse = None   # type: ignore
    Response = None       # type: ignore
    _HAS_FASTAPI = False

logger = logging.getLogger("axropus.server")


# ── Server state ────────────────────────────────────────────────────────────


@dataclass
class ServerState:
    config:   AxropusConfig
    hook:     Any   # AMFvLLMHook
    metrics:  MetricRegistry
    vllm_engine: Any = None


def _build_hook(cfg: AxropusConfig):
    """Construct the compressed pool + router + hook from a config."""
    from korith_vllm_ext.amf_vllm_hook import AMFvLLMHook
    from korith_vllm_ext.compressed_vram_pool import CompressedVRAMPool
    from korith_vllm_ext.lmcache_adapter import LMCacheAdapter
    from korith_vllm_ext.tiered_router import TieredCacheRouter, WritePolicy

    pool = CompressedVRAMPool(
        num_layers=cfg.num_layers,
        bytes_per_layer=cfg.bytes_per_layer,
        block_bytes=cfg.block_bytes,
        default_format=cfg.default_format,
        device=cfg.device,
    )
    lmc = LMCacheAdapter(
        enabled=cfg.lmcache_enabled,
        backend=cfg.lmcache_backend,
        path=cfg.lmcache_path,
        url=cfg.lmcache_url,
    )
    write_policy_map = {
        "always":       WritePolicy.ALWAYS,
        "large_only":   WritePolicy.LARGE_ONLY,
        "on_eviction":  WritePolicy.ON_EVICTION,
        "never":        WritePolicy.NEVER,
    }
    router = TieredCacheRouter(
        pool=pool,
        lmcache=lmc,
        min_prefix_tokens=cfg.min_prefix_tokens,
        write_policy=write_policy_map.get(cfg.write_policy, WritePolicy.LARGE_ONLY),
        large_write_threshold=cfg.large_write_threshold,
    )
    hook = AMFvLLMHook(
        pool=pool,
        router=router,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        min_prefix=cfg.min_prefix_tokens,
    )
    return hook


def _try_build_vllm(cfg: AxropusConfig):
    """Best-effort vLLM engine construction.

    Returns the engine on success, ``None`` when vLLM is not installed
    (in which case the server runs in stub mode for local smoke tests).
    """
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        engine = LLM(model=cfg.model, tensor_parallel_size=1, gpu_memory_utilization=0.80, max_model_len=4096)
        logger.info("vLLM engine created for model %s", cfg.model)
        return engine
    except Exception as exc:
        logger.info("vLLM unavailable (%s); running in stub mode", exc)
        return None


# ── App factory ─────────────────────────────────────────────────────────────


def create_app(cfg: AxropusConfig):
    """Return a FastAPI application bound to the given config."""
    if not _HAS_FASTAPI:
        raise RuntimeError("fastapi not installed — install axropus[server]")

    hook = _build_hook(cfg)
    metrics = MetricRegistry()
    state = ServerState(config=cfg, hook=hook, metrics=metrics, vllm_engine=_try_build_vllm(cfg))

    app = FastAPI(title="Axropus AMF", version="0.1.0")
    app.state.axropus = state

    # ── Auth dependency ───────────────────────────────────────────────────
    def _check_auth(auth_header: Optional[str]) -> None:
        if not cfg.api_key:
            return
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if auth_header.split(" ", 1)[1] != cfg.api_key:
            raise HTTPException(status_code=401, detail="bad bearer token")

    # ── Health ────────────────────────────────────────────────────────────
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": cfg.model, "vllm": state.vllm_engine is not None}

    # ── Metrics ───────────────────────────────────────────────────────────
    @app.get("/metrics")
    def prometheus():
        # Refresh pool gauges before rendering.
        pool_stats = state.hook.pool.stats()
        metrics.record_pool_snapshot(
            bytes_used=pool_stats["used_bytes"],
            prefixes=pool_stats["num_prefixes"],
            evictions=pool_stats["evictions"],
        )
        return Response(
            content=metrics.render(),
            media_type="text/plain; version=0.0.4",
        )

    # ── Stats ─────────────────────────────────────────────────────────────
    @app.get("/stats")
    def stats():
        return state.hook.stats()

    # ── Chat completion (the important endpoint) ──────────────────────────
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        _check_auth(authorization)
        body = await request.json()
        return _handle_chat(state, body)

    @app.post("/v1/completions")
    async def completions(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        _check_auth(authorization)
        body = await request.json()
        # Wrap plain prompt in an OpenAI-style chat message.
        prompt = body.get("prompt", "")
        body["messages"] = [{"role": "user", "content": prompt}]
        return _handle_chat(state, body)

    return app


# ── Chat request handling ───────────────────────────────────────────────────


def _handle_chat(state: ServerState, body: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.monotonic()
    messages: List[Dict[str, Any]] = body.get("messages") or []
    prompt_text = _render_messages(messages)

    # Trivial tokenization fallback — the real server uses vLLM's tokenizer.
    tokens = _tokenize(state, prompt_text)
    seq_id = int(t0 * 1e6) & 0xFFFFFFFF

    decision = state.hook.on_request_arrival(seq_id, tokens)
    state.metrics.record_hit(decision.tier.value, decision.latency_ms)
    if decision.miss_reason_name:
        state.metrics.record_miss(decision.miss_reason_name)

    if decision.action.value == "skip_to_decode":
        # Warm hit: on a real deployment we would jump into decode. Here we
        # return the cached prefix metadata.
        state.metrics.record_tokens_saved(
            len(tokens),
            compute_seconds=decision.pool_entry.avg_savings_ms / 1000.0 if decision.pool_entry else 0,
        )

    # Proxy to vLLM when available.
    if state.vllm_engine is not None:
        out_text = _vllm_generate(state, prompt_text, body)
    else:
        out_text = f"[stub] received {len(tokens)} tokens, tier={decision.tier.value}"

    # Store-after-prefill simulation. In production this hook is called by
    # vLLM's worker with the actual KV tensor; we don't have one here so we
    # emit a synthetic small tensor for stub mode.
    if state.vllm_engine is None and decision.action.value == "cold_prefill":
        try:
            import torch

            stub_kv = torch.zeros(
                state.config.num_layers,
                2,
                len(tokens),
                state.config.num_kv_heads,
                state.config.head_dim,
                dtype=torch.float16,
            )
            state.hook.on_prefill_complete(seq_id, stub_kv, saved_ms=0.0)
        except Exception:
            pass

    state.hook.on_request_complete(seq_id)

    latency_ms = (time.monotonic() - t0) * 1000.0
    return {
        "id":      f"chatcmpl-{seq_id:08x}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   state.config.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": out_text},
                "finish_reason": "stop",
            }
        ],
        "axropus": {
            "tier":      decision.tier.value,
            "hit":       decision.action.value == "skip_to_decode",
            "latency_ms": round(latency_ms, 3),
            "prefix_hash": decision.prefix_hash,
        },
    }


def _render_messages(messages: List[Dict[str, Any]]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out.append(f"<|{role}|>{content}<|/{role}|>")
    return "".join(out)


def _tokenize(state: ServerState, text: str) -> List[int]:
    engine = state.vllm_engine
    if engine is not None:
        try:
            tok = engine.get_tokenizer()  # type: ignore[attr-defined]
            return list(tok.encode(text))
        except Exception:
            pass
    # Deterministic byte-level fallback so two identical prompts map to the
    # same prefix hash.
    return [b for b in text.encode("utf-8", "replace")]


def _vllm_generate(state: ServerState, prompt_text: str, body: Dict[str, Any]) -> str:
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(
        temperature=float(body.get("temperature", 0.7)),
        top_p=float(body.get("top_p", 1.0)),
        max_tokens=int(body.get("max_tokens", 256)),
    )
    outputs = state.vllm_engine.generate([prompt_text], params)
    return outputs[0].outputs[0].text if outputs else ""


# ── Patch HookDecision.miss_reason_name (convenience) ───────────────────────
# The dataclass defined in amf_vllm_hook.HookDecision does not carry the
# string form of the miss reason. We add it here so we do not have to reach
# into private attributes on every call.

def _add_miss_reason_attr() -> None:
    from korith_vllm_ext.amf_vllm_hook import HookDecision

    def _miss_reason_name(self):
        return None
    HookDecision.miss_reason_name = property(_miss_reason_name)  # type: ignore[attr-defined]


_add_miss_reason_attr()


__all__ = ["create_app", "ServerState"]
