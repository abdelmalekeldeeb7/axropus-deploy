#!/usr/bin/env python3
"""
AMF H100 Benchmark — 1M Token Context
======================================
Measures cold vs warm TTFT on H100 with a large model at long context.

Usage (on H100 node):
    python3 scripts/h100_1m_benchmark.py \
        --model models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
        --context 1000000 \
        --runs 50

Outputs:
    platform_data/h100_bench/<timestamp>/
        summary.json          — cold/warm stats, speedup, savings %
        raw_metrics.csv       — one row per run (matches 817-run CSV format)
        logs/                 — per-run stdout/stderr

Environment:
    KORITH_H100_MODEL       — model path (overrides --model)
    KORITH_H100_CONTEXT     — context tokens (overrides --context)
    KORITH_H100_RUNS        — total runs (overrides --runs)
    KORITH_H100_AMF_DIR     — AMF store directory (default: auto under output dir)
    KORITH_H100_DRAFT_MODEL — draft model path for speculative decode
    HF_TOKEN                — Hugging Face token (for download if needed)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# H100 GPU profiles — indexed by context size
# ---------------------------------------------------------------------------

@dataclass
class H100Profile:
    n_ctx: int
    n_batch: int
    cuda_dtype: str
    amf_max_bytes: int          # GPU-side AMF budget
    amf_storage_budget_gb: int  # Disk AMF budget
    bench_timeout_s: int        # Per-run timeout
    tensor_split: str = ""      # Comma-separated GPU split ratios, e.g. "0.125,0.125,..."
    n_gpus: int = 1             # Number of GPUs to use
    kv_fp8: bool = False        # Enable FP8 KV cache (halves KV memory)
    h100_cost_per_hr: float = 3.50  # $/hr for cost estimate


H100_PROFILES: Dict[int, H100Profile] = {
    # 1M context — full AMF disk-based KV store (70B model, single H100)
    1_048_576: H100Profile(
        n_ctx=1_048_576,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,        # 64 GB GPU AMF
        amf_storage_budget_gb=2048,           # 2 TB disk (KV for 70B at 1M ≈ 320 GB)
        bench_timeout_s=7200,                 # 2 h (cold run can be long)
    ),
    # 512K context
    524_288: H100Profile(
        n_ctx=524_288,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,
        amf_storage_budget_gb=1024,
        bench_timeout_s=3600,
    ),
    # 256K context
    262_144: H100Profile(
        n_ctx=262_144,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,
        amf_storage_budget_gb=512,
        bench_timeout_s=1800,
    ),
    # 128K context — matches existing 817-run baseline
    131_072: H100Profile(
        n_ctx=131_072,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,
        amf_storage_budget_gb=512,
        bench_timeout_s=1200,
    ),
}

# Named profiles for large models (selected via --profile flag)
NAMED_PROFILES: Dict[str, H100Profile] = {
    # 405B @ 1M — 8x H100 NVL8, FP8 KV
    # Memory: 405B Q4 (~230GB) + FP8 KV at 1M (~252GB) = 482GB < 640GB ✓
    "405b_1m": H100Profile(
        n_ctx=1_048_576,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,
        amf_storage_budget_gb=512,            # KV in VRAM via tensor_split — disk is overflow
        bench_timeout_s=10800,                # 3 h cold can be 10-30 min for 405B at 1M
        tensor_split=",".join(["0.125"] * 8),
        n_gpus=8,
        kv_fp8=True,
        h100_cost_per_hr=19.92,              # 8x H100 @ ~$2.49/hr each
    ),
    # 405B @ 128K — 8x H100 NVL8, BF16 KV (fits easily at 128K)
    "405b_128k": H100Profile(
        n_ctx=131_072,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,
        amf_storage_budget_gb=256,
        bench_timeout_s=1800,
        tensor_split=",".join(["0.125"] * 8),
        n_gpus=8,
        kv_fp8=False,
        h100_cost_per_hr=19.92,
    ),
    # 70B @ 1M — single H100, FP8 KV (disk-based)
    "70b_1m": H100Profile(
        n_ctx=1_048_576,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,
        amf_storage_budget_gb=512,
        bench_timeout_s=7200,
        tensor_split="",
        n_gpus=1,
        kv_fp8=True,
        h100_cost_per_hr=3.50,
    ),
    # 120B class @ 128K — 2x H100 SXM (Nemotron-70B GGUF, same arch family)
    # Cold prefill expected: 8-15 min at 128K. Warm: ~12-20s. Speedup: ~30-60x
    "120b_128k": H100Profile(
        n_ctx=131_072,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,
        amf_storage_budget_gb=256,
        bench_timeout_s=1800,
        tensor_split="0.5,0.5",
        n_gpus=2,
        kv_fp8=False,
        h100_cost_per_hr=7.00,   # 2x H100 @ $3.50 each
    ),
    # 120B class @ 1M — 2x H100 SXM, FP8 KV
    # Cold prefill expected: 60-120 min at 1M. Warm: ~12-20s. Speedup: 200-600x
    "120b_1m": H100Profile(
        n_ctx=1_048_576,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024 ** 3,
        amf_storage_budget_gb=1024,
        bench_timeout_s=14400,   # 4h cold can be long at 120B+1M
        tensor_split="0.5,0.5",
        n_gpus=2,
        kv_fp8=True,
        h100_cost_per_hr=7.00,
    ),
}

# Cascade order: try largest first
CONTEXT_CASCADE = [1_048_576, 524_288, 262_144, 131_072]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

PROMPT_PREAMBLE = (
    "SYSTEM CONTEXT:\n"
    "You are processing a long technical document for an AI inference optimization benchmark. "
    "The document contains detailed system performance data, CUDA kernel statistics, memory access "
    "patterns, KV cache utilization metrics, and speculative decode acceptance rates across multiple "
    "hardware configurations. After reading this document, provide a concise summary.\n\n"
    "DOCUMENT BEGIN:\n"
)

# A paragraph that is rich in technical tokens (tokenizes at roughly 1 token per word)
_PARA_TEMPLATE = (
    "Benchmark iteration {i}: prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f} "
    "context_tokens={ctx} hit_rate={hr:.4f} cache_savings_pct={savings:.2f} "
    "kv_restore_ms={restore:.2f} speculative_acceptance={accept:.4f} "
    "amf_decision={decision} skip_ratio={skip:.4f} "
    "gpu_utilization_pct={gpu:.1f} hbm_bandwidth_gbs={bw:.1f} "
    "layer_compute_ms={layer:.4f} attention_flops={flops:.2e} "
    "rope_theta={rope} n_heads={heads} head_dim={hdim} gqa_groups={gqa} "
    "tensor_parallel_rank={tp_rank} pipeline_parallel_rank={pp_rank}\n"
)


def generate_prompt(target_tokens: int, output_path: Path) -> int:
    """
    Write a synthetic prompt to output_path that tokenizes to approximately
    target_tokens. Returns estimated token count.

    Uses a technical paragraph template so the prefix is semantically coherent
    and produces a stable AMF hash key across runs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rough estimate: each rendered paragraph ≈ 90-100 tokens
    tokens_per_para = 95
    n_para = max(1, target_tokens // tokens_per_para)

    import math, random
    rng = random.Random(42)  # fixed seed → same prompt every time → same AMF key

    with output_path.open("w", encoding="utf-8") as f:
        f.write(PROMPT_PREAMBLE)
        written = len(PROMPT_PREAMBLE.split())
        for i in range(n_para):
            para = _PARA_TEMPLATE.format(
                i=i,
                prefill_ms=rng.uniform(100, 300_000),
                decode_ms=rng.uniform(100, 15_000),
                ctx=target_tokens,
                hr=rng.uniform(0.95, 0.9999),
                savings=rng.uniform(80, 99.9),
                restore=rng.uniform(50, 5000),
                accept=rng.uniform(0.5, 0.99),
                decision="hit" if i > 0 else "miss",
                skip=1.0 if i > 0 else 0.0,
                gpu=rng.uniform(60, 99),
                bw=rng.uniform(1000, 22000),
                layer=rng.uniform(0.001, 10.0),
                flops=10 ** rng.uniform(12, 16),
                rope=rng.choice([10000, 500000, 1000000]),
                heads=rng.choice([8, 16, 32, 64]),
                hdim=128,
                gqa=rng.choice([1, 4, 8]),
                tp_rank=0,
                pp_rank=0,
            )
            f.write(para)
            written += len(para.split())

        f.write("\nDOCUMENT END.\nSummary: ")

    estimated = written
    return estimated


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpu() -> Tuple[str, int]:
    """Returns (gpu_name, vram_mb)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        ).strip()
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            name = parts[0] if parts else "unknown"
            mem = int(parts[1]) if len(parts) > 1 else 0
            return name, mem
    except Exception:
        pass
    return "unknown", 0


def detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=15,
        ).strip()
        return max(1, len(out.splitlines()))
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_index: int
    rc: int
    decision: str
    skip_ratio: float
    total_ms: float
    decode_ms: float
    prefill_ms: float
    restore_ms: float
    context_tokens: int
    output_tokens: int
    tps: float
    error: str


def run_once(
    *,
    binary: str,
    model_path: str,
    prompt_file: Path,
    run_index: int,
    run_dir: Path,
    profile: H100Profile,
    amf_path: Path,
    enable_amf: bool,
    seed: int = 42,
    max_tokens: int = 128,
    draft_model: str = "",
) -> RunResult:
    tag = f"run_{run_index:04d}"
    metrics_path = run_dir / "metrics" / f"{tag}.json"
    log_out = run_dir / "logs" / f"{tag}.out"
    log_err = run_dir / "logs" / f"{tag}.err"

    env = os.environ.copy()
    env["KORITH_PROMPT_FILE"] = str(prompt_file)
    env["KORITH_MAX_TOKENS"] = str(max_tokens)
    env["KORITH_N_CTX"] = str(profile.n_ctx)
    env["KORITH_N_BATCH"] = str(profile.n_batch)
    env["KORITH_CUDA_DTYPE"] = profile.cuda_dtype
    env["KORITH_SEED"] = str(seed)
    env["KORITH_TEMPERATURE"] = "0.0"

    # Multi-GPU: tensor_split spreads model weights across all available GPUs
    if profile.tensor_split and profile.n_gpus > 1:
        env["KORITH_TENSOR_SPLIT"] = profile.tensor_split
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(profile.n_gpus))
        env["KORITH_MAIN_GPU"] = "0"
    else:
        env["KORITH_CUDA_DEVICE"] = "0"

    # FP8 KV cache — halves KV memory, required for 405B at 1M to fit in VRAM
    if profile.kv_fp8:
        env["KORITH_KV_FP8"] = "1"

    # AMF
    env["KORITH_ENABLE_AMF"] = "1" if enable_amf else "0"
    env["KORITH_ENABLE_MF"] = "1" if enable_amf else "0"
    env["KORITH_AMF_PATH"] = str(amf_path)
    env["KORITH_AMF_MAX_BYTES"] = str(profile.amf_max_bytes)
    env["KORITH_AMF_STORAGE_BUDGET_GB"] = str(profile.amf_storage_budget_gb)
    env["KORITH_AMF_DISK_GUARD"] = "1"

    # Acceleration
    env["KORITH_ACCEL_ENABLED"] = "1"
    env["KORITH_KERNELS"] = "1"
    env["KORITH_KERNEL_BACKEND"] = "cuda"
    env["KORITH_DECODE_CACHE_ENABLED"] = "1"
    env["KORITH_DECODE_SCHEDULER_ENABLED"] = "1"

    # Speculative decode (if draft model provided)
    if draft_model and Path(draft_model).exists():
        env["KORITH_SPEC_ENABLED"] = "1"
        env["KORITH_SPEC_K"] = "6"
        env["KORITH_DRAFT_MODEL"] = draft_model
        env["KORITH_DRAFT_MODEL_PATH"] = draft_model
        env["KORITH_SPEC_MISS_ONLY"] = "1"
    else:
        env["KORITH_SPEC_ENABLED"] = "0"

    # Metrics output
    env["KORITH_ENGINE_METRICS_PATH"] = str(metrics_path)
    env["KORITH_ENGINE_EVENTS_PATH"] = str(run_dir / "metrics" / f"{tag}.events.jsonl")
    env["KORITH_JOB_ID"] = tag
    env["KORITH_TENANT_ID"] = "h100_bench"

    t0 = time.perf_counter()
    proc = subprocess.run(
        [binary, model_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=profile.bench_timeout_s,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    log_out.write_text(proc.stdout, encoding="utf-8")
    log_err.write_text(proc.stderr, encoding="utf-8")

    decision = "miss"
    skip_ratio = 0.0
    decode_ms = 0.0
    prefill_ms = 0.0
    restore_ms = 0.0
    output_tokens = max_tokens
    tps = 0.0
    error = ""

    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            amf = m.get("amf", {}) if isinstance(m.get("amf"), dict) else {}
            perf = m.get("perf", {}) if isinstance(m.get("perf"), dict) else {}
            decision = str(amf.get("decision", "miss"))
            skip_ratio = float(amf.get("skip_ratio", 0.0) or 0.0)
            restore_ms = float(amf.get("restore_ms", 0.0) or 0.0)
            prefill_ms = float(perf.get("prefill_ms", 0.0) or 0.0)
            decode_ms = float(perf.get("decode_ms", elapsed_ms) or elapsed_ms)
            output_tokens = int(perf.get("tokens_out", max_tokens) or max_tokens)
            tps = float(perf.get("avg_tps", 0.0) or 0.0)
        except Exception as exc:
            error = f"metrics_parse_error:{exc}"
    else:
        error = f"metrics_missing rc={proc.returncode}"

    if proc.returncode != 0 and not error:
        # Extract last stderr line for the error summary
        last_err = (proc.stderr or "").strip().splitlines()
        error = f"rc={proc.returncode} {last_err[-1][:120] if last_err else ''}"

    total_ms = elapsed_ms  # wall-clock (most honest)

    return RunResult(
        run_index=run_index,
        rc=proc.returncode,
        decision=decision,
        skip_ratio=skip_ratio,
        total_ms=total_ms,
        decode_ms=decode_ms,
        prefill_ms=prefill_ms,
        restore_ms=restore_ms,
        context_tokens=profile.n_ctx,
        output_tokens=output_tokens,
        tps=tps,
        error=error,
    )


# ---------------------------------------------------------------------------
# Find korith_dynamic binary
# ---------------------------------------------------------------------------

def find_binary(root: Path) -> Optional[str]:
    candidates = [
        root / "build" / "korith_dynamic",
        root / "korith_dynamic",
        Path("./build/korith_dynamic"),
        Path("korith_dynamic"),
    ]
    for c in candidates:
        if c.exists() and os.access(str(c), os.X_OK):
            return str(c)
    # Try PATH
    found = shutil.which("korith_dynamic")
    return found


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    binary: str,
    model_path: str,
    context_target: int,
    runs: int,
    output_root: Path,
    draft_model: str = "",
    max_tokens: int = 128,
    seed: int = 42,
    force_context: bool = False,
    orchestrator=None,
) -> Dict:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    gpu_name, vram_mb = detect_gpu()
    gpu_count = detect_gpu_count()

    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    log(f"GPU: {gpu_name} ({vram_mb} MB VRAM) x{gpu_count}")
    log(f"Model: {model_path}")
    log(f"Target context: {context_target:,} tokens")
    log(f"Runs: {runs} (1 cold + {runs-1} warm)")

    # Pick profile — cascade down from target if not found
    if force_context:
        cascade = [context_target]
    else:
        cascade = [c for c in CONTEXT_CASCADE if c <= context_target] or CONTEXT_CASCADE[-1:]
        # Add target if not in our predefined list
        if context_target not in CONTEXT_CASCADE:
            cascade = [context_target] + cascade

    selected_ctx = None
    profile = None

    for ctx_size in cascade:
        if ctx_size in H100_PROFILES:
            selected_ctx = ctx_size
            profile = H100_PROFILES[ctx_size]
        else:
            # Build a profile for this custom size
            selected_ctx = ctx_size
            profile = H100Profile(
                n_ctx=ctx_size,
                n_batch=2048,
                cuda_dtype="bf16",
                amf_max_bytes=64 * 1024 ** 3,
                amf_storage_budget_gb=max(512, ctx_size // 1000),  # rough estimate
                bench_timeout_s=max(1200, ctx_size // 500),
            )
        log(f"Attempting context size: {selected_ctx:,} tokens")
        break  # try the largest first; fallback happens in run loop

    run_dir = output_root / f"h100_amf_{selected_ctx}_{stamp}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # AMF store — persistent across runs (that's the point)
    amf_path = run_dir / "amf_store"
    amf_path.mkdir(parents=True, exist_ok=True)

    # Generate prompt file
    log(f"Generating {selected_ctx:,}-token prompt file...")
    prompt_file = run_dir / f"prompt_{selected_ctx}.txt"
    estimated_tokens = generate_prompt(selected_ctx, prompt_file)
    log(f"Prompt generated: ~{estimated_tokens:,} tokens ({prompt_file.stat().st_size / 1024 / 1024:.1f} MB)")

    rows: List[RunResult] = []
    actual_ctx = selected_ctx
    fallback_used = False

    for run_i in range(1, runs + 1):
        # On the first failed run, try smaller context
        if rows and rows[-1].rc != 0 and not fallback_used:
            cascade_remaining = [c for c in CONTEXT_CASCADE if c < actual_ctx]
            if cascade_remaining and not force_context:
                actual_ctx = cascade_remaining[0]
                profile = H100_PROFILES[actual_ctx]
                log(f"Falling back to {actual_ctx:,} context due to previous failure")
                # Regenerate prompt for new context
                prompt_file = run_dir / f"prompt_{actual_ctx}.txt"
                generate_prompt(actual_ctx, prompt_file)
                # Clear AMF store so we get a clean cold run
                shutil.rmtree(str(amf_path), ignore_errors=True)
                amf_path.mkdir(parents=True, exist_ok=True)
                fallback_used = True
                # Reset rows for clean cold run
                rows = []
                run_i = 1

        is_cold = (len(rows) == 0)
        log(f"Run {run_i}/{runs} ({'COLD' if is_cold else 'warm'}) ctx={actual_ctx:,}...")

        # AI reasoning: get decision before run, may override spec_k
        reasoning_outcome = None
        if orchestrator is not None:
            try:
                import hashlib
                prefix_hash = hashlib.sha256(
                    f"h100_bench:{actual_ctx}:{seed}".encode()
                ).hexdigest()[:16]
                reasoning_outcome = orchestrator.on_request(
                    prefix_hash=prefix_hash,
                    tenant_id="h100_bench",
                    node_id="node-0",
                    worker_id="worker-0",
                )
            except Exception as exc:
                log(f"  [reasoning] on_request error: {exc}")

        try:
            result = run_once(
                binary=binary,
                model_path=model_path,
                prompt_file=prompt_file,
                run_index=run_i,
                run_dir=run_dir,
                profile=profile,
                amf_path=amf_path,
                enable_amf=True,
                seed=seed,
                max_tokens=max_tokens,
                draft_model=draft_model,
            )
        except subprocess.TimeoutExpired:
            log(f"  TIMEOUT after {profile.bench_timeout_s}s")
            result = RunResult(
                run_index=run_i, rc=-1,
                decision="timeout", skip_ratio=0.0,
                total_ms=profile.bench_timeout_s * 1000.0,
                decode_ms=0.0, prefill_ms=0.0, restore_ms=0.0,
                context_tokens=actual_ctx, output_tokens=0,
                tps=0.0, error="timeout",
            )

        rows.append(result)

        # AI reasoning: record outcome for RL reward calculation
        if orchestrator is not None and reasoning_outcome is not None:
            try:
                if result.decision == "hit":
                    orchestrator.on_hit(prefix_hash)
                else:
                    orchestrator.on_miss(prefix_hash)
            except Exception as exc:
                log(f"  [reasoning] outcome hook error: {exc}")

        status = "OK" if result.rc == 0 else f"FAIL rc={result.rc}"
        log(
            f"  {status} | {result.decision} | total={result.total_ms/1000:.1f}s "
            f"prefill={result.prefill_ms/1000:.1f}s decode={result.decode_ms/1000:.1f}s "
            f"skip={result.skip_ratio:.3f} tps={result.tps:.1f}"
            + (f" | {result.error}" if result.error else "")
        )

    # Compute summary stats
    succeeded = [r for r in rows if r.rc == 0]
    cold_rows = [r for r in succeeded if r.decision != "hit"]
    warm_rows = [r for r in succeeded if r.decision == "hit"]

    cold_ms = cold_rows[0].total_ms if cold_rows else 0.0
    warm_ms_list = [r.total_ms for r in warm_rows]
    warm_avg_ms = sum(warm_ms_list) / len(warm_ms_list) if warm_ms_list else 0.0
    warm_p50_ms = sorted(warm_ms_list)[len(warm_ms_list) // 2] if warm_ms_list else 0.0

    hit_rate = len(warm_rows) / len(succeeded) if succeeded else 0.0
    speedup = cold_ms / warm_avg_ms if warm_avg_ms > 0 else 0.0
    e2e_savings_pct = (1.0 - warm_avg_ms / cold_ms) * 100.0 if cold_ms > 0 else 0.0

    # H100 cost: use profile rate (single H100 = $3.50/hr, 8x H100 = $19.92/hr)
    h100_cost_per_ms = profile.h100_cost_per_hr / 3_600_000
    cost_saved_per_request = (cold_ms - warm_avg_ms) * h100_cost_per_ms if warm_avg_ms > 0 else 0.0

    summary = {
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "vram_mb": vram_mb,
        "model_path": model_path,
        "context_tokens": actual_ctx,
        "context_target": context_target,
        "fallback_used": fallback_used,
        "runs_requested": runs,
        "runs_succeeded": len(succeeded),
        "cold_runs": len(cold_rows),
        "warm_runs": len(warm_rows),
        "hit_rate": hit_rate,
        "cold_ttft_ms": cold_ms,
        "cold_ttft_s": cold_ms / 1000.0,
        "warm_ttft_avg_ms": warm_avg_ms,
        "warm_ttft_avg_s": warm_avg_ms / 1000.0,
        "warm_ttft_p50_ms": warm_p50_ms,
        "speedup_x": speedup,
        "e2e_savings_pct": e2e_savings_pct,
        "cost_saved_per_warm_request_usd": cost_saved_per_request,
        "draft_model": draft_model or "",
        "max_tokens": max_tokens,
        "seed": seed,
        "n_gpus": profile.n_gpus,
        "tensor_split": profile.tensor_split,
        "kv_fp8": profile.kv_fp8,
        "reasoning_enabled": orchestrator is not None,
    }

    # Write CSV (matches existing 817-run format)
    csv_path = run_dir / "raw_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_index", "decision", "skip_ratio",
            "total_ms", "decode_ms", "prefill_ms",
            "restore_ms", "duration_s", "context_tokens",
            "output_tokens", "tps", "rc", "error",
        ])
        for r in rows:
            writer.writerow([
                r.run_index, r.decision, f"{r.skip_ratio:.6f}",
                f"{r.total_ms:.1f}", f"{r.decode_ms:.1f}", f"{r.prefill_ms:.1f}",
                f"{r.restore_ms:.1f}", f"{r.total_ms/1000:.3f}",
                r.context_tokens, r.output_tokens,
                f"{r.tps:.2f}", r.rc, r.error,
            ])

    # Write summary JSON
    out = {"summary": summary, "rows": [asdict(r) for r in rows]}
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Print results table
    print()
    print("=" * 70)
    print(f"  AMF H100 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  GPU:              {gpu_name} x{profile.n_gpus} ({vram_mb} MB VRAM each)")
    print(f"  Tensor split:     {profile.tensor_split or 'none (single GPU)'}")
    print(f"  KV dtype:         {'FP8' if profile.kv_fp8 else 'BF16'}")
    print(f"  AI reasoning:     {'enabled' if orchestrator is not None else 'disabled'}")
    print(f"  Model:            {Path(model_path).name}")
    print(f"  Context:          {actual_ctx:,} tokens", end="")
    if fallback_used:
        print(f"  [FALLBACK from {context_target:,}]", end="")
    print()
    print(f"  Runs:             {len(succeeded)} succeeded / {runs} requested")
    print(f"  Hit rate:         {hit_rate*100:.2f}%  ({len(warm_rows)} warm / {len(cold_rows)} cold)")
    print()
    print(f"  Cold TTFT:        {cold_ms/1000:.1f}s  ({cold_ms:,.0f} ms)")
    print(f"  Warm TTFT (avg):  {warm_avg_ms/1000:.1f}s  ({warm_avg_ms:,.0f} ms)")
    print(f"  Warm TTFT (p50):  {warm_p50_ms/1000:.1f}s  ({warm_p50_ms:,.0f} ms)")
    print()
    print(f"  Speedup:          {speedup:.1f}x")
    print(f"  E2E savings:      {e2e_savings_pct:.1f}%")
    print(f"  Cost saved/req:   ${cost_saved_per_request:.4f} (at $3.50/hr H100 rate)")
    print("=" * 70)
    print(f"  Results dir:      {run_dir}")
    print(f"  CSV:              {csv_path}")
    print(f"  Summary JSON:     {summary_path}")
    print("=" * 70)

    return {"run_dir": str(run_dir), "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AMF H100 Benchmark — cold vs warm TTFT at 1M token context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile",
        default="",
        choices=list(NAMED_PROFILES.keys()),
        help=f"Named hardware profile: {list(NAMED_PROFILES.keys())}. "
             "Overrides --context and sets multi-GPU/FP8 config automatically.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("KORITH_H100_MODEL", "models/Llama-3.1-70B-Instruct-Q4_K_M.gguf"),
        help="Path to target GGUF model (default: %(default)s)",
    )
    parser.add_argument(
        "--draft-model",
        default=os.environ.get("KORITH_H100_DRAFT_MODEL", ""),
        help="Path to draft GGUF model for speculative decode (optional)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=int(os.environ.get("KORITH_H100_CONTEXT", "1000000")),
        help="Target context size in tokens — cascades down if too large (default: %(default)s)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("KORITH_H100_RUNS", "50")),
        help="Total runs including 1 cold (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Output tokens per run (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        default="platform_data/h100_bench",
        help="Output directory root (default: %(default)s)",
    )
    parser.add_argument(
        "--binary",
        default="",
        help="Path to korith_dynamic binary (auto-detected if omitted)",
    )
    parser.add_argument(
        "--force-context",
        action="store_true",
        help="Do not cascade to smaller context on failure",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable AI reasoning layer (Orchestrator + spec decode decisions). "
             "Requires platform/reasoning/ and Llama-3.1-8B on cuda:0.",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print H100 env vars for the target context and exit",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent.resolve()

    if args.print_env:
        ctx = args.context
        nearest = min(H100_PROFILES.keys(), key=lambda k: abs(k - ctx))
        p = H100_PROFILES[nearest]
        print(f"export KORITH_N_CTX={p.n_ctx}")
        print(f"export KORITH_N_BATCH={p.n_batch}")
        print(f"export KORITH_CUDA_DTYPE={p.cuda_dtype}")
        print(f"export KORITH_AMF_MAX_BYTES={p.amf_max_bytes}")
        print(f"export KORITH_AMF_STORAGE_BUDGET_GB={p.amf_storage_budget_gb}")
        print("export KORITH_ENABLE_AMF=1")
        print("export KORITH_ACCEL_ENABLED=1")
        print("export KORITH_KERNELS=1")
        print("export KORITH_KERNEL_BACKEND=cuda")
        return 0

    # Resolve binary
    binary = args.binary or find_binary(root)
    if not binary:
        print(
            "ERROR: korith_dynamic binary not found.\n"
            "Build with: cmake -S korith_c_api -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\n"
            "Or pass --binary /path/to/korith_dynamic",
            file=sys.stderr,
        )
        return 1

    model_path = args.model
    if not Path(model_path).exists():
        print(
            f"ERROR: model not found: {model_path}\n"
            "Download with huggingface-cli or set --model to an existing GGUF path.\n"
            "Suggested command:\n"
            "  huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF "
            "--include 'Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf' --local-dir models/",
            file=sys.stderr,
        )
        return 1

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # If a named profile was given, inject it into H100_PROFILES at the right ctx size
    if args.profile:
        named = NAMED_PROFILES[args.profile]
        H100_PROFILES[named.n_ctx] = named
        # Override context to match profile
        args.context = named.n_ctx
        print(f"[profile] Using named profile '{args.profile}': "
              f"ctx={named.n_ctx:,} gpus={named.n_gpus} "
              f"tensor_split={named.tensor_split or 'none'} "
              f"kv_fp8={named.kv_fp8}")

    # Optional: AI reasoning layer
    orchestrator = None
    if args.reasoning:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from platform.reasoning.metrics_collector import MetricsCollector
            from platform.reasoning.reasoning_model import ReasoningModel
            from platform.reasoning.decision_executor import DecisionExecutor
            from platform.reasoning.reward_calculator import RewardCalculator
            from platform.reasoning.learning_loop import LearningLoop
            from platform.reasoning.orchestrator import Orchestrator

            print("[reasoning] Initialising AI reasoning layer (Llama-3.1-8B on cuda:0)...")
            collector = MetricsCollector()
            model_r = ReasoningModel(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                device="cuda:0",
            )
            executor = DecisionExecutor(coordinator_client=None)
            reward_calc = RewardCalculator(
                db_path=str(output_root / "rewards.db")
            )
            loop = LearningLoop(
                model=model_r,
                reward_calculator=reward_calc,
                rl_level=1,
            )
            orchestrator = Orchestrator(
                collector, model_r, executor, reward_calc, loop,
                enabled=True, ab_testing=False,
            )
            orchestrator.start()
            print("[reasoning] Ready.")
        except Exception as exc:
            print(f"[reasoning] WARNING: failed to init reasoning layer: {exc}", file=sys.stderr)
            print("[reasoning] Continuing without reasoning layer.", file=sys.stderr)
            orchestrator = None

    result = run_benchmark(
        binary=binary,
        model_path=model_path,
        context_target=args.context,
        runs=max(2, args.runs),  # need at least 1 cold + 1 warm
        output_root=output_root,
        draft_model=args.draft_model,
        max_tokens=args.max_tokens,
        seed=args.seed,
        force_context=args.force_context,
        orchestrator=orchestrator,
    )

    if orchestrator is not None:
        try:
            orchestrator.stop()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
