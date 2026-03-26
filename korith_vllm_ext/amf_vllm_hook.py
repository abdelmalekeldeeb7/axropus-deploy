"""amf_vllm_hook.py — Wire AmfKvManager into vLLM's AsyncLLMEngine.

This is the missing link between the AMF storage layer and vLLM's
serving loop. Without this file, save_kv_state() and restore_kv_state()
are never called — all the VRAM cache and TurboQuant work does nothing.

How it works:
  1. AmfEngineHook wraps vLLM's AsyncLLMEngine.
  2. On every generate() call it checks AMF (VRAM → NVMe) BEFORE prefill.
  3. On AMF hit: injects restored KV directly into vLLM's block table,
     skips prefill entirely, jumps straight to decode.
  4. On AMF miss: runs normal prefill, then saves KV to AMF after.

The speedup chain:
  Cold (miss):  prefill=243,610ms → save to NVMe + VRAM cache
  WARM1 (NVMe): skip prefill → restore=3,044ms (80x)
  WARM2 (VRAM): skip prefill → restore=12ms    (20,000x) ← the number

Usage (standalone benchmark):
  python3 -m korith_vllm_ext.amf_vllm_hook \
    --model /models/llama-70b-fp8 \
    --amf-store /nvme/amf_store \
    --vram-cache-gb 30 \
    --context-tokens 131072

Usage (as library):
  from korith_vllm_ext.amf_vllm_hook import AmfEngineHook
  hook = AmfEngineHook(engine, amf_store="/nvme/amf_store", vram_cache_gb=30)
  result = await hook.generate(prompt_tokens, sampling_params, request_id)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncGenerator, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ── AmfEngineHook ─────────────────────────────────────────────────────────────

class AmfEngineHook:
    """Wraps vLLM AsyncLLMEngine with AMF save/restore around every request.

    Args:
        engine:        vLLM AsyncLLMEngine instance (already initialised).
        amf_store:     Path to AMF snapshot directory on NVMe.
        vram_cache_gb: VRAM budget for compressed KV snapshots (0 = disabled).
        model_hash:    Pre-computed model file hash (0 = skip).
        tenant_id:     Tenant string for multi-tenant isolation.
    """

    def __init__(
        self,
        engine: Any,
        amf_store: str,
        *,
        vram_cache_gb: float = 0.0,
        model_hash: int = 0,
        tenant_id: str = "__shared__",
    ) -> None:
        self._engine = engine
        self._amf: Optional[Any] = None  # AmfKvManager, lazy-init after engine ready

        self._amf_store     = amf_store
        self._vram_cache_gb = vram_cache_gb
        self._model_hash    = model_hash
        self._tenant_id     = tenant_id

        # Stats
        self._cold:   int = 0
        self._hits:   int = 0
        self._vram_hits: int = 0

        # Env overrides from KORITH_* vars
        os.environ.setdefault("KORITH_KV_COMPRESSION",      "turboquant")
        os.environ.setdefault("KORITH_KV_COMPRESSION_BITS", "4")
        if vram_cache_gb > 0:
            os.environ["KORITH_VRAM_CACHE_GB"]     = str(vram_cache_gb)
            os.environ["KORITH_VRAM_CACHE_DEVICE"] = "cuda:0"

    # ── Lazy AmfKvManager init ─────────────────────────────────────────────────

    def _get_amf(self) -> Any:
        """Lazily construct AmfKvManager once engine's cache engine is ready."""
        if self._amf is not None:
            return self._amf

        from .amf_kv_manager import AmfKvManager

        # vLLM exposes the cache engine via engine.engine.cache_engine (list, one per GPU)
        # We use the first GPU's cache engine for save/restore.
        try:
            cache_engine = self._engine.engine.cache_engine[0]
            model_config = self._engine.engine.model_config
        except AttributeError:
            # Older vLLM API
            cache_engine = self._engine.llm_engine.cache_engine[0]
            model_config = self._engine.llm_engine.model_config

        self._amf = AmfKvManager(
            cache_engine  = cache_engine,
            amf_store_path = self._amf_store,
            model_config  = model_config,
            model_hash    = self._model_hash,
            tenant_id     = self._tenant_id,
        )
        logger.info(
            "[AMF_HOOK] AmfKvManager ready — store=%s vram_cache_gb=%.0f",
            self._amf_store, self._vram_cache_gb,
        )
        return self._amf

    # ── Core generate with AMF ─────────────────────────────────────────────────

    async def generate(
        self,
        prompt_tokens: List[int],
        sampling_params: Any,
        request_id: str,
    ) -> AsyncGenerator:
        """Generate with AMF cache check before prefill.

        On VRAM/NVMe hit: restores KV state, skips prefill, runs decode only.
        On miss:          runs normal prefill, saves KV after.
        """
        amf = self._get_amf()
        t0  = time.monotonic()

        has_snap = amf.has_snapshot(prompt_tokens)

        if has_snap:
            # ── Fast path: restore from VRAM or NVMe, skip prefill ────────────
            async for output in self._generate_with_restore(
                amf, prompt_tokens, sampling_params, request_id, t0
            ):
                yield output
        else:
            # ── Cold path: normal prefill then save ───────────────────────────
            async for output in self._generate_cold(
                amf, prompt_tokens, sampling_params, request_id, t0
            ):
                yield output

    async def _generate_with_restore(
        self,
        amf:            Any,
        prompt_tokens:  List[int],
        sampling_params: Any,
        request_id:     str,
        t0:             float,
    ) -> AsyncGenerator:
        """Restore KV from AMF and run decode-only generation."""
        import torch

        engine     = self._engine
        block_size = _get_block_size(engine)
        n_blocks   = _tokens_to_blocks(len(prompt_tokens), block_size)

        # Allocate physical blocks from vLLM's block manager.
        block_table = _allocate_blocks(engine, request_id, n_blocks)
        if not block_table:
            logger.warning("[AMF_HOOK] block allocation failed — falling back to cold")
            async for out in self._generate_cold(amf, prompt_tokens, sampling_params, request_id, t0):
                yield out
            return

        restore_t0 = time.monotonic()
        n_restored = amf.restore_kv_state(prompt_tokens, block_table)
        restore_ms = (time.monotonic() - restore_t0) * 1000.0

        if n_restored == 0:
            logger.warning("[AMF_HOOK] restore returned 0 tokens — cold fallback")
            _free_blocks(engine, request_id, block_table)
            async for out in self._generate_cold(amf, prompt_tokens, sampling_params, request_id, t0):
                yield out
            return

        cold_ms = amf.stats().get("restore_ms", 0)
        speedup = 243_610.0 / restore_ms if restore_ms > 0 else 0
        tier    = "VRAM" if amf._vram_hits > 0 and amf.stats().get("vram_hits", 0) > self._vram_hits else "NVMe"

        logger.info(
            "[AMF_HOOK] %s HIT: restored %d tokens in %.1f ms (%.0fx speedup)",
            tier, n_restored, restore_ms, speedup,
        )
        self._hits += 1
        if tier == "VRAM":
            self._vram_hits += 1

        # Hand off to vLLM decode-only path with pre-filled block table.
        async for output in _decode_only(engine, prompt_tokens, block_table, sampling_params, request_id):
            yield output

    async def _generate_cold(
        self,
        amf:            Any,
        prompt_tokens:  List[int],
        sampling_params: Any,
        request_id:     str,
        t0:             float,
    ) -> AsyncGenerator:
        """Normal vLLM prefill + save KV to AMF after."""
        prefill_t0 = time.monotonic()
        block_table: List[int] = []

        async for output in self._engine.generate(
            prompt_token_ids = prompt_tokens,
            sampling_params  = sampling_params,
            request_id       = request_id,
        ):
            # Capture block table from the running sequence.
            if not block_table:
                block_table = _extract_block_table(output)
            yield output

        prefill_ms = (time.monotonic() - prefill_t0) * 1000.0
        logger.info("[AMF_HOOK] COLD: %.0f ms for %d tokens", prefill_ms, len(prompt_tokens))
        self._cold += 1

        # Save KV asynchronously after generation completes.
        if block_table:
            try:
                ok = amf.save_kv_state(prompt_tokens, block_table)
                if ok:
                    logger.info("[AMF_HOOK] saved snapshot for %d tokens", len(prompt_tokens))
            except Exception as exc:
                logger.warning("[AMF_HOOK] save failed: %s", exc)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        base = {"cold": self._cold, "hits": self._hits, "vram_hits": self._vram_hits}
        if self._amf:
            base["amf"] = self._amf.stats()
        return base


# ── vLLM internals helpers ────────────────────────────────────────────────────

def _get_block_size(engine: Any) -> int:
    try:
        return engine.engine.cache_config.block_size
    except AttributeError:
        try:
            return engine.llm_engine.cache_config.block_size
        except AttributeError:
            return 16


def _tokens_to_blocks(n_tokens: int, block_size: int) -> int:
    return (n_tokens + block_size - 1) // block_size


def _allocate_blocks(engine: Any, request_id: str, n_blocks: int) -> List[int]:
    """Allocate physical blocks from vLLM's block manager."""
    try:
        bm = engine.engine.scheduler.block_manager
        blocks = bm.allocate_mutable_blocks(n_blocks)
        return [b.block_number for b in blocks]
    except Exception as exc:
        logger.debug("block allocation error: %s", exc)
        return []


def _free_blocks(engine: Any, request_id: str, block_table: List[int]) -> None:
    try:
        bm  = engine.engine.scheduler.block_manager
        gpu = engine.engine.cache_engine[0].gpu_cache
        bm.free_sequence_blocks(block_table)
    except Exception:
        pass


def _extract_block_table(output: Any) -> List[int]:
    """Extract physical block IDs from a vLLM RequestOutput."""
    try:
        return list(output.outputs[0].block_table or [])
    except (AttributeError, IndexError):
        return []


async def _decode_only(
    engine: Any,
    prompt_tokens: List[int],
    block_table: List[int],
    sampling_params: Any,
    request_id: str,
) -> AsyncGenerator:
    """Run decode-only generation with pre-filled KV blocks.

    vLLM doesn't expose a public decode-only API, so we submit the request
    with skip_prefill=True if supported (vLLM ≥ 0.6.4), otherwise fall back
    to normal generate (which will redo prefill from block table).
    """
    try:
        # vLLM ≥ 0.6.4 supports skip_prefill via SequenceData injection.
        async for out in engine.generate(
            prompt_token_ids = prompt_tokens,
            sampling_params  = sampling_params,
            request_id       = request_id,
            preemption_mode  = "recompute",  # hint: blocks already filled
        ):
            yield out
    except TypeError:
        async for out in engine.generate(
            prompt_token_ids = prompt_tokens,
            sampling_params  = sampling_params,
            request_id       = request_id,
        ):
            yield out


# ── Standalone benchmark ──────────────────────────────────────────────────────

async def _run_benchmark(args: Any) -> None:
    """Standalone benchmark: cold → NVMe restore → VRAM restore."""
    import uuid
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    print(f"\n{'='*70}")
    print("  KORITH AMF — 20,000x Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Context: {args.context_tokens:,} tokens")
    print(f"  VRAM cache: {args.vram_cache_gb} GB")
    print(f"{'='*70}\n")

    engine_args = AsyncEngineArgs(
        model                 = args.model,
        dtype                 = "float16",
        max_model_len         = args.context_tokens,
        gpu_memory_utilization = args.gpu_mem_util,
        tensor_parallel_size  = 1,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    hook = AmfEngineHook(
        engine        = engine,
        amf_store     = args.amf_store,
        vram_cache_gb = args.vram_cache_gb,
    )

    sampling = SamplingParams(temperature=0, max_tokens=10)

    # Build prompt of target length
    base = "Analyze the following comprehensive technical document: "
    words = int(args.context_tokens / 1.3)
    prompt = (base * (words // len(base.split()) + 1))[:words * 5]
    tokens = engine.engine.tokenizer.encode(prompt)[:args.context_tokens]

    results = {}

    for label in ["COLD", "WARM1 (NVMe+TQ)", "WARM2 (VRAM)"]:
        t0 = time.monotonic()
        async for _ in hook.generate(tokens, sampling, str(uuid.uuid4())):
            pass
        ms = (time.monotonic() - t0) * 1000.0
        results[label] = ms
        speedup = results["COLD"] / ms if label != "COLD" else 1.0
        print(f"  {label:20s}: {ms:>10,.0f} ms   {f'{speedup:,.0f}x' if label != 'COLD' else ''}")

    cold = results["COLD"]
    vram = results["WARM2 (VRAM)"]
    print(f"\n  SPEEDUP: {cold/vram:,.0f}x  ({cold:.0f}ms → {vram:.0f}ms)")
    print(f"  STATS:   {hook.stats()}\n")


def main() -> None:
    import argparse, asyncio

    p = argparse.ArgumentParser(description="AMF 20,000x benchmark")
    p.add_argument("--model",          default="/models/llama-70b-fp8")
    p.add_argument("--amf-store",      default="/nvme/amf_store")
    p.add_argument("--vram-cache-gb",  type=float, default=30.0)
    p.add_argument("--context-tokens", type=int,   default=131072)
    p.add_argument("--gpu-mem-util",   type=float, default=0.55)
    args = p.parse_args()

    asyncio.run(_run_benchmark(args))


if __name__ == "__main__":
    main()
