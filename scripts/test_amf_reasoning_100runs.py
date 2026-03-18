#!/usr/bin/env python3
"""
AMF + AI Reasoning — 100 runs at 4K context.
Learning loop fires every 60s (not 1hr).
Shows ROI EMA, spec depth, admission decisions adapting over time.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
logging.getLogger("amf_100").setLevel(logging.INFO)
log = logging.getLogger("amf_100")

MODEL_PATH = "/home/korith/.axropus/models/llama-3.2-1b-q4.gguf"
AMF_PATH   = "/home/korith/.axropus/amf_100runs"
DATA_DIR   = "/home/korith/.axropus/data_100runs"

os.makedirs(AMF_PATH, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

os.environ["KORITH_PLATFORM_AMF_PATH"] = AMF_PATH
os.environ["KORITH_AMF_PATH"]          = AMF_PATH
os.environ["KORITH_DATA_DIR"]          = DATA_DIR
os.environ["KORITH_KV_FP8"]            = "0"
os.environ["KORITH_API_KEY_SALT"]      = "test-salt"
os.environ["KORITH_DEFAULT_MODEL_PATH"] = MODEL_PATH

import numpy as np
from platform.adapters.korith_local import Tier1LocalKorithAdapter
from platform.reasoning.metrics_collector import MetricsCollector
from platform.reasoning.decision_executor import DecisionExecutor
from platform.reasoning.reward_calculator import RewardCalculator
from platform.reasoning.learning_loop import LearningLoop
from platform.reasoning.orchestrator import Orchestrator
from platform.reasoning.reasoning_model import UnifiedDecisionVector

log.info("Stack imported")

# ── mock reasoning model — adapts spec depth based on live metrics ────────────
class AdaptiveReasoningModel:
    """
    Simulates the Llama 8B / NemoClaw decision engine.
    Reads the live metrics tensor and adapts decisions each cycle.
    """
    def decide(self, tensor):
        hit_rate     = float(tensor[2])
        storage_util = float(tensor[10])
        roi_ema      = float(tensor[29]) * 10.0  # denorm
        miss_streak  = float(tensor[24])
        accept_ema   = float(tensor[32])

        # Spec depth: adapt based on hit rate + acceptance EMA
        if hit_rate > 0.80:
            spec_depth = 8
        elif hit_rate > 0.60:
            spec_depth = 6
        elif hit_rate > 0.30:
            spec_depth = 4
        else:
            spec_depth = 2

        # Admission: open once ROI signal builds up
        admit = (roi_ema > 0.5) or (hit_rate > 0.5)

        # Throttle on miss streaks
        throttle = min(0.2 * miss_streak, 0.4)

        return UnifiedDecisionVector(
            cache_admission=admit,
            admission_priority=hit_rate,
            eviction_target=None, pre_warm_predictions=[],
            route_decision=None, batch_group=None,
            throttle_back_pressure=throttle,
            inference_ms=1.5, is_fallback=False, domain="unified",
            spec_decode_enable=True, spec_decode_depth=spec_depth,
            spec_mode="v2_clean", spec_v3_block_size=8,
            draft_temperature=0.8, kv_compression_enable=False,
            kv_compression_ratio=1.0, decode_batch_priority=2,
            early_stop_confidence=0.0, cuda_graph_hint=True,
            adaptive_depth_override=False,
        )
    def load(self): pass
    def unload(self): pass

# ── build stack ───────────────────────────────────────────────────────────────
db_path      = str(Path(tempfile.mkdtemp()) / "rewards.db")
collector    = MetricsCollector()
model_r      = AdaptiveReasoningModel()
executor     = DecisionExecutor(coordinator_client=None)
reward_calc  = RewardCalculator(db_path=db_path)
loop         = LearningLoop(
    reasoning_model=model_r,
    reward_calculator=reward_calc,
    interval_s=60,              # every 60s instead of 1hr
    min_positive_examples=3,    # fire with fewer examples
    rl_level=1,
)
orchestrator = Orchestrator(collector, model_r, executor, reward_calc, loop, enabled=True)
orchestrator.start()

adapter = Tier1LocalKorithAdapter(model_path=MODEL_PATH)

# ── run config ────────────────────────────────────────────────────────────────
DET_CFG = {"seed": 42, "n_ctx": 4096, "n_batch": 512}
POLICY  = {"allow_amf_reuse": True, "_tenant_id": "test"}

PROMPT = (
    "You are an expert in distributed systems. "
    "Explain the role of KV cache in transformer inference. "
    + "The KV cache stores key and value tensors computed during the attention mechanism. " * 60
    + "Question: What is the primary benefit of KV cache reuse?"
)
prefix_hash = hashlib.sha256(PROMPT[:200].encode()).hexdigest()[:16]

_run_counter = 0
def make_artifacts(tag):
    global _run_counter
    _run_counter += 1
    job_dir = Path(DATA_DIR) / "artifacts" / f"run_{_run_counter:03d}_{tag}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return {
        "job_dir":        str(job_dir),
        "output":         str(job_dir / "output.txt"),
        "log":            str(job_dir / "engine.log"),
        "engine_metrics": str(job_dir / "engine_metrics.json"),
        "engine_events":  str(job_dir / "engine_events.json"),
        "mf_snapshot":    str(job_dir / "mf_snapshot.bin"),
    }

# ── header ────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  AMF + AI REASONING  —  100 runs, 4K context, Llama 3.2 1B  (loop=60s)")
print("="*80)
print(f"  {'Run':<5} {'Mode':<10} {'ms':>7} {'Pfill':>7} {'SpecK':>6} {'Admitted':>9} {'HitRate':>8} {'ROI':>6} {'LoopCyc':>8}")
print("-"*80)

baseline_ms  = None
mf_snapshot  = None
warm_times   = []
loop_cycles_last = 0

for i in range(1, 101):
    outcome = orchestrator.on_request(
        prefix_hash=prefix_hash,
        tenant_id="test", node_id="gpu-0", worker_id="worker-0",
    )

    if i == 1:
        # Cold baseline
        artifacts = make_artifacts("base")
        result = adapter.run_baseline(
            prompt=PROMPT, max_tokens=32,
            deterministic_cfg=DET_CFG,
            policy={**POLICY, "allow_amf_reuse": False},
            artifacts=artifacts, mf_snapshot_in=None,
        )
        metrics  = result.get("engine_metrics", {})
        perf     = metrics.get("perf", {})
        total    = float(perf.get("total_ms", 0) or 0) or result.get("total_ms", 0)
        prefill  = float(perf.get("prefill_ms", 0) or 0)
        baseline_ms = total
        mode_label = "BASELINE"
        orchestrator.on_miss(prefix_hash)
        collector.record_miss(prefix_hash)
        collector.record_decode(decode_ms=total)

    else:
        artifacts = make_artifacts(f"accel")
        snap_in = mf_snapshot if (mf_snapshot and Path(mf_snapshot).exists()) else None
        result = adapter.run_accel(
            prompt=PROMPT, max_tokens=32,
            deterministic_cfg=DET_CFG,
            policy=POLICY,
            artifacts=artifacts, mf_snapshot_in=snap_in,
        )
        metrics  = result.get("engine_metrics", {})
        perf     = metrics.get("perf", {})
        total    = float(perf.get("total_ms", 0) or 0) or result.get("total_ms", 0)
        prefill  = float(perf.get("prefill_ms", 0) or 0)

        snap_out = artifacts["mf_snapshot"]
        if Path(snap_out).exists():
            mf_snapshot = snap_out

        is_hit = (prefill == 0.0 and i > 2)
        mode_label = "WARM" if is_hit else "AMF"
        if is_hit:
            orchestrator.on_hit(prefix_hash)
            collector.record_hit(prefix_hash=prefix_hash, roi=3.0, restore_ms=10.0)
            warm_times.append(total)
        else:
            orchestrator.on_miss(prefix_hash)
            collector.record_miss(prefix_hash)
        collector.record_decode(decode_ms=total)

    if result.get("exit_code", -1) != 0:
        print(f"  {i:<5} ERROR  exit={result.get('exit_code')}  {result.get('engine_errors', [])}")
        continue

    m        = collector.get_metrics_dict()
    hit_rate = m["hit_rate"]
    roi      = m["roi_ema_norm"] * 10
    spec_k   = outcome.spec_k or "—"
    admitted = "YES" if outcome.admitted else "NO "
    loop_cyc = getattr(loop, "_cycle_count", 0)

    savings = f"  ({100*(1-total/baseline_ms):+.0f}%)" if baseline_ms and i > 1 else ""

    # Print every run for first 20, then every 5
    if i <= 20 or i % 5 == 0:
        print(f"  {i:<5} {mode_label:<10} {total:>7.0f} {prefill:>7.0f} {str(spec_k):>6} {admitted:>9} {hit_rate:>7.1%} {roi:>6.2f} {loop_cyc:>8}{savings}")

    # Milestone summary every 25 runs
    if i in (25, 50, 75, 100):
        avg_warm = sum(warm_times[-20:]) / max(1, len(warm_times[-20:]))
        print(f"\n  ── MILESTONE run {i} ── hit_rate={hit_rate:.1%}  avg_warm={avg_warm:.0f}ms  roi={roi:.2f}  loop_cycles={loop_cyc}\n")

print("-"*80)

# ── final summary ─────────────────────────────────────────────────────────────
stats = orchestrator.get_stats()
m     = collector.get_metrics_dict()
avg_warm_all = sum(warm_times) / max(1, len(warm_times))

print(f"\n  Total runs        : 100")
print(f"  AI augmented      : {stats.ai_augmented_requests}/100")
print(f"  Final hit rate    : {m['hit_rate']:.1%}")
print(f"  Final ROI EMA     : {m['roi_ema_norm']*10:.2f}")
print(f"  Loop cycles fired : {getattr(loop, '_cycle_count', 0)}")
print(f"  Baseline (cold)   : {baseline_ms:.0f}ms")
print(f"  Avg warm time     : {avg_warm_all:.0f}ms")
if baseline_ms and avg_warm_all:
    savings = 100 * (1 - avg_warm_all / baseline_ms)
    print(f"  Avg savings       : {savings:.1f}%")

try:
    reward_stats = reward_calc.get_stats()
    print(f"\n  Reward stats      : {reward_stats}")
except Exception:
    pass

print(f"\n  AMF store files   : {len(list(Path(AMF_PATH).glob('*')))}")
print()
orchestrator.stop()
