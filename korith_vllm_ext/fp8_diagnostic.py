"""fp8_diagnostic.py — Diagnose why FP8 KV cache produces garbage after restore.

Runs a minimal FP8 save → reset_prefix_cache → restore cycle and logs:
  1. Attention layer scales BEFORE save (after first prefill)
  2. A sample of the raw FP8 bytes BEFORE save
  3. Attention layer scales AFTER reset_prefix_cache
  4. Attention layer scales AFTER restore
  5. A sample of the raw FP8 bytes AFTER restore
  6. The decoded output text

The hypothesis: scales drift between save and warm run because either
``calculate_kv_scales`` stays True and vLLM recomputes on the warm batch's
suffix, or some other vLLM internal resets them.

Usage:
    KORITH_ENABLE_AMF=1 KORITH_AMF_PATH=/tmp/amf_fp8_diag \\
        python3 -m korith_vllm_ext.fp8_diagnostic \\
        --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \\
        --prefix-file /tmp/long_prompt.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import torch


def _dump_scales(model_runner: Any, label: str, limit: int = 4) -> dict:
    """Walk attention layers and collect scale values. Prints the first N."""
    scales = {}
    layer_idx = 0
    for name, module in model_runner.model.named_modules():
        if not any(hasattr(module, a) for a in ("_k_scale", "k_scale")):
            continue
        entry = {}
        for attr in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale",
                     "k_scale", "v_scale", "q_scale", "prob_scale",
                     "_k_scale_float", "_v_scale_float",
                     "calculate_kv_scales"):
            if hasattr(module, attr):
                val = getattr(module, attr)
                if isinstance(val, torch.Tensor):
                    try:
                        entry[attr] = float(val.item())
                    except Exception:
                        entry[attr] = "tensor"
                elif isinstance(val, bool):
                    entry[attr] = val
                else:
                    try:
                        entry[attr] = float(val)
                    except Exception:
                        entry[attr] = str(val)
        scales[f"L{layer_idx}"] = {"name": name, **entry}
        layer_idx += 1

    # Print summary for the first `limit` layers
    print(f"\n[FP8_DIAG] {label}: {layer_idx} attention layers", flush=True)
    shown = 0
    for key, val in scales.items():
        if shown >= limit:
            break
        print(f"[FP8_DIAG]   {key} ({val.get('name', '?')}):", flush=True)
        for k, v in val.items():
            if k != "name":
                print(f"[FP8_DIAG]     {k} = {v}", flush=True)
        shown += 1
    return scales


def _dump_kv_bytes(kv_cache_tensor: torch.Tensor, label: str, n_samples: int = 16) -> list:
    """Sample a few raw bytes from the KV cache tensor for comparison."""
    try:
        flat = kv_cache_tensor.flatten().contiguous()
        # Use view to uint8 for FP8 cache (element_size == 1)
        if flat.element_size() == 1:
            buf = flat.view(torch.uint8)
        else:
            buf = flat.view(torch.uint8) if hasattr(flat, 'view') else flat
        n = min(n_samples, buf.numel())
        sample = buf[:n].cpu().tolist()
        print(f"[FP8_DIAG] {label}: first {n} bytes = {sample}", flush=True)
        return sample
    except Exception as exc:
        print(f"[FP8_DIAG] {label}: dump failed: {exc}", flush=True)
        return []


def run(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    model_path = args.model
    amf_path = os.environ.get("KORITH_AMF_PATH", "/tmp/amf_fp8_diag")
    os.makedirs(amf_path, exist_ok=True)

    print(f"[FP8_DIAG] model={model_path}", flush=True)
    print(f"[FP8_DIAG] amf_path={amf_path}", flush=True)

    # Read prefix
    if args.prefix_file and os.path.exists(args.prefix_file):
        with open(args.prefix_file, "r") as f:
            prefix = f.read()
    else:
        prefix = "Explain quantum computing in great detail. " * 200

    prompts = [
        prefix + "\n\nQuestion 1: What is the first prime number?",
        prefix + "\n\nQuestion 2: What is the second prime number?",
    ]

    # Load model with FP8 KV + AMF worker extension
    ext_cls = "korith_vllm_ext.amf_worker_ext.AmfWorkerExtension"
    print("[FP8_DIAG] loading model with FP8 KV cache...", flush=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        kv_cache_dtype="fp8",
        worker_extension_cls=ext_cls,
    )
    print("[FP8_DIAG] model loaded", flush=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=16)

    # ── Phase 1: Cold prefill ──────────────────────────────────────────
    print("\n[FP8_DIAG] phase 1: cold prefill (populates scales)", flush=True)
    t0 = time.monotonic()
    cold_out = llm.generate([prompts[0]], sampling)
    cold_ms = (time.monotonic() - t0) * 1000.0
    cold_text = cold_out[0].outputs[0].text
    print(f"[FP8_DIAG] cold_ms={cold_ms:.0f} text='{cold_text[:80]}'", flush=True)

    # Dump scales after cold prefill via collective_rpc
    def _snap(label):
        return llm.collective_rpc(
            "amf_diag_snapshot",
            args=(label,),
        )

    # If the worker doesn't have amf_diag_snapshot, fall back to direct reflection
    try:
        scales_post_cold = _snap("post_cold")[0]
    except Exception as exc:
        print(f"[FP8_DIAG] rpc snapshot not available: {exc}", flush=True)
        scales_post_cold = None

    # ── Phase 2: Save KV ───────────────────────────────────────────────
    print("\n[FP8_DIAG] phase 2: save KV to AMF", flush=True)
    from korith_vllm_ext.korith_vllm_server import _call_engine_core
    tokenizer = llm.get_tokenizer()
    token_ids = tokenizer.encode(prefix)

    try:
        cached = _call_engine_core(llm, "amf_get_cached_block_ids", list(token_ids))
        block_ids = cached.get("block_ids", []) if isinstance(cached, dict) else []
    except Exception as exc:
        print(f"[FP8_DIAG] get_cached_block_ids failed: {exc}", flush=True)
        block_ids = []

    save_results = llm.collective_rpc(
        "amf_save_kv",
        timeout=300,
        args=(amf_path, list(token_ids), 0, "__fp8diag__"),
        kwargs={"physical_block_ids": block_ids} if block_ids else None,
    )
    save_info = save_results[0] if save_results else {}
    print(f"[FP8_DIAG] save result: {save_info}", flush=True)

    # ── Phase 3: Reset prefix cache ────────────────────────────────────
    print("\n[FP8_DIAG] phase 3: reset prefix cache", flush=True)
    try:
        llm.llm_engine.reset_prefix_cache()
        print("[FP8_DIAG] prefix cache reset OK", flush=True)
    except Exception as exc:
        print(f"[FP8_DIAG] reset_prefix_cache failed: {exc}", flush=True)

    try:
        scales_post_reset = _snap("post_reset")[0]
    except Exception:
        scales_post_reset = None

    # ── Phase 4: Restore KV ────────────────────────────────────────────
    print("\n[FP8_DIAG] phase 4: restore KV from AMF", flush=True)
    t_r = time.monotonic()
    restore_results = llm.collective_rpc(
        "amf_restore_kv",
        timeout=300,
        args=(amf_path, list(token_ids), 0, "__fp8diag__"),
    )
    restore_ms = (time.monotonic() - t_r) * 1000.0
    n_restored = restore_results[0] if restore_results else 0
    print(f"[FP8_DIAG] restored {n_restored} tokens in {restore_ms:.0f}ms", flush=True)

    # Register in prefix cache
    try:
        block_size = save_info.get("block_size", 16)
        n_blocks = len(token_ids) // block_size
        reg_blocks = list(range(n_blocks))
        reg = _call_engine_core(llm, "amf_register_prefix",
                                 list(token_ids), reg_blocks)
        print(f"[FP8_DIAG] register: {reg}", flush=True)
    except Exception as exc:
        print(f"[FP8_DIAG] register failed: {exc}", flush=True)

    try:
        scales_post_restore = _snap("post_restore")[0]
    except Exception:
        scales_post_restore = None

    # ── Phase 5: Warm run ──────────────────────────────────────────────
    print("\n[FP8_DIAG] phase 5: warm run", flush=True)
    t_w = time.monotonic()
    warm_out = llm.generate(prompts, sampling)
    warm_ms = (time.monotonic() - t_w) * 1000.0
    print(f"[FP8_DIAG] warm_ms={warm_ms:.0f}", flush=True)
    for i, out in enumerate(warm_out):
        text = out.outputs[0].text
        print(f"[FP8_DIAG] warm[{i}] = '{text[:80]}'", flush=True)

    try:
        scales_post_warm = _snap("post_warm")[0]
    except Exception:
        scales_post_warm = None

    # ── Summary ────────────────────────────────────────────────────────
    print("\n[FP8_DIAG] SUMMARY", flush=True)
    print(f"  cold_ms      = {cold_ms:.0f}", flush=True)
    print(f"  save_saved   = {save_info.get('saved', False)}", flush=True)
    print(f"  n_restored   = {n_restored}", flush=True)
    print(f"  restore_ms   = {restore_ms:.0f}", flush=True)
    print(f"  warm_ms      = {warm_ms:.0f}", flush=True)
    print(f"  cold_text    = '{cold_text[:60]}'", flush=True)

    # Compare scales across phases
    def _diff_scales(a, b, label_a, label_b):
        if not isinstance(a, dict) or not isinstance(b, dict):
            return
        print(f"\n[FP8_DIAG] SCALE DIFF {label_a} -> {label_b}", flush=True)
        keys = list(a.keys())[:3]
        for k in keys:
            va = a.get(k, {})
            vb = b.get(k, {})
            for attr in ("_k_scale", "_v_scale", "calculate_kv_scales"):
                if attr in va or attr in vb:
                    print(f"  {k}.{attr}: {va.get(attr)} -> {vb.get(attr)}", flush=True)

    _diff_scales(scales_post_cold,    scales_post_reset,   "post_cold",   "post_reset")
    _diff_scales(scales_post_reset,   scales_post_restore, "post_reset",  "post_restore")
    _diff_scales(scales_post_restore, scales_post_warm,    "post_restore", "post_warm")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prefix-file", type=str, default="")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
