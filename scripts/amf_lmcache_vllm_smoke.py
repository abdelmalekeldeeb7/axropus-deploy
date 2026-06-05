#!/usr/bin/env python3
"""Smoke-test vLLM + LMCache + AMF on one repeated prefix.

This is intentionally small: it proves the current installed vLLM can load
LMCache's KV connector while AMF's worker extension is installed, then exercises
both cache paths:

1. Cold generate and let LMCache store KV.
2. AMF-save the prefix KV.
3. Reset vLLM/APC.
4. AMF-restore and register the prefix.
5. Generate again and print vLLM's reported cached token count.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

from korith_vllm_ext.korith_vllm_server import _call_engine_core


def _set_default_env(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def _generate(llm: LLM, prompt: str, sampling: SamplingParams) -> tuple[float, int, list[int]]:
    t0 = time.perf_counter()
    out = llm.generate([prompt], sampling, use_tqdm=False)[0]
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, int(out.num_cached_tokens or 0), list(out.outputs[0].token_ids)


def _coverage(llm: LLM, token_ids: list[int]) -> dict[str, Any]:
    info = _call_engine_core(llm, "amf_get_cached_block_ids", list(token_ids))
    return info if isinstance(info, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/korith/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--kv-cache-mb", type=int, default=384)
    parser.add_argument("--lmcache-cpu-gb", type=float, default=1.0)
    parser.add_argument(
        "--lmcache-role",
        choices=("kv_both", "kv_producer", "kv_consumer"),
        default="kv_both",
    )
    parser.add_argument("--amf-path", default="/tmp/korith-amf-lmcache-smoke")
    parser.add_argument("--prefix-repeat", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    _set_default_env("PYTHONHASHSEED", "0")
    _set_default_env("LMCACHE_TRACK_USAGE", "false")
    _set_default_env("LMCACHE_LOCAL_CPU", "true")
    _set_default_env("LMCACHE_MAX_LOCAL_CPU_SIZE", str(args.lmcache_cpu_gb))
    _set_default_env("LMCACHE_CHUNK_SIZE", "256")
    _set_default_env("LMCACHE_SAVE_UNFULL_CHUNK", "true")
    _set_default_env("KORITH_ENABLE_AMF", "1")
    _set_default_env("KORITH_AMF_PATH", args.amf_path)
    _set_default_env("KORITH_AMF_VRAM_FIRST", "1")
    _set_default_env("KORITH_VRAM_CACHE_GB", "1")
    _set_default_env("KORITH_VRAM_CACHE_DEVICE", "cuda:0")
    _set_default_env("KORITH_VRAM_POOL_GB", "1")
    _set_default_env("KORITH_VRAM_POOL_QUANT", "int4")

    shutil.rmtree(args.amf_path, ignore_errors=True)
    Path(args.amf_path).mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.70,
        max_model_len=args.max_model_len,
        kv_cache_memory_bytes=args.kv_cache_mb * 1024 * 1024,
        enable_prefix_caching=True,
        worker_extension_cls="korith_vllm_ext.amf_worker_ext.AmfWorkerExtension",
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role=args.lmcache_role,
            kv_connector_extra_config={
                "discard_partial_chunks": False,
                "skip_last_n_tokens": 0,
            },
        ),
    )

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    prefix = "System: cache smoke. " * args.prefix_repeat
    suffix = "\nUser: reply one token."
    prompt = prefix + suffix
    prefix_ids = llm.get_tokenizer().encode(prefix)
    prompt_ids = llm.get_tokenizer().encode(prompt)

    print(
        f"[SMOKE] model={args.model} prompt_tokens={len(prompt_ids)} "
        f"prefix_tokens={len(prefix_ids)} vllm_lmcache_amf=enabled "
        f"lmcache_role={args.lmcache_role}",
        flush=True,
    )

    llm.llm_engine.reset_prefix_cache(reset_running_requests=True)
    cold_ms, cold_cached, cold_ids = _generate(llm, prompt, sampling)
    cov = _coverage(llm, prefix_ids)
    save_results = llm.collective_rpc(
        "amf_save_kv",
        timeout=600,
        args=(args.amf_path, list(prefix_ids), 0, "__shared__"),
        kwargs={"physical_block_ids": cov.get("block_ids", [])} if cov.get("block_ids") else None,
    )
    saved = bool((save_results[0] if save_results else {}).get("saved", False))
    print(
        f"[SMOKE] cold_ms={cold_ms:.2f} cold_cached={cold_cached} "
        f"amf_save={saved} lmcache_store_expected=true ids={cold_ids}",
        flush=True,
    )

    llm.llm_engine.reset_prefix_cache(reset_running_requests=True)
    t0 = time.perf_counter()
    restored = llm.collective_rpc(
        "amf_restore_kv",
        timeout=600,
        args=(args.amf_path, list(prefix_ids), 0, "__shared__"),
    )
    restore_ms = (time.perf_counter() - t0) * 1000.0
    restored_tokens = int(restored[0] if restored else 0)
    block_size = int(cov.get("block_size", 16) or 16)
    restore_block_ids = list(range(max(len(prefix_ids) // block_size, 1)))
    t0 = time.perf_counter()
    _call_engine_core(llm, "amf_register_prefix", list(prefix_ids), restore_block_ids)
    register_ms = (time.perf_counter() - t0) * 1000.0
    warm_ms, warm_cached, warm_ids = _generate(llm, prompt, sampling)

    print(
        f"[SMOKE] restore_ms={restore_ms:.2f} register_ms={register_ms:.2f} "
        f"restored_tokens={restored_tokens} warm_ms={warm_ms:.2f} "
        f"warm_cached={warm_cached} output_match={warm_ids == cold_ids} ids={warm_ids}",
        flush=True,
    )
    return 0 if saved and warm_ids == cold_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
