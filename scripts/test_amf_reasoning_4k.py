#!/usr/bin/env python3
"""
AMF + AI Reasoning — 10 run test at 4K context.

Run pattern:
  Run 1   : baseline (no AMF) — measures cold prefill time
  Run 2-10: accel (AMF enabled) — run 2 is cold store, runs 3-10 are warm hits

Shows: cold time, warm time, savings %, and AI reasoning decisions per run.
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

logging.basicConfig(
    level=logging.WARNING,  # suppress noisy platform logs
    format="[%(asctime)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Show our own logs
logging.getLogger("amf_test").setLevel(logging.INFO)
log = logging.getLogger("amf_test")

MODEL_PATH = "/home/korith/.axropus/models/llama-3.2-1b-q4.gguf"
AMF_PATH   = "/home/korith/.axropus/amf_4k_test"
DATA_DIR   = "/home/korith/.axropus/data_4k_test"

os.makedirs(AMF_PATH, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Set env vars the adapter reads
os.environ["KORITH_PLATFORM_AMF_PATH"] = AMF_PATH
os.environ["KORITH_AMF_PATH"]          = AMF_PATH
os.environ["KORITH_DATA_DIR"]          = DATA_DIR
os.environ["KORITH_KV_FP8"]            = "0"
os.environ["KORITH_API_KEY_SALT"]      = "test-salt"
os.environ["KORITH_DEFAULT_MODEL_PATH"] = MODEL_PATH

# ── import platform stack ────────────────────────────────────────────────────
import numpy as np
from platform.adapters.korith_local import Tier1LocalKorithAdapter
from platform.reasoning.metrics_collector import MetricsCollector
from platform.reasoning.decision_executor import DecisionExecutor
from platform.reasoning.reward_calculator import RewardCalculator
from platform.reasoning.learning_loop import LearningLoop
from platform.reasoning.orchestrator import Orchestrator
from platform.reasoning.reasoning_model import UnifiedDecisionVector

log.info("Platform stack imported")

# ── mock reasoning model (no Llama 8B needed locally) ───────────────────────
class MockReasoningModel:
    def decide(self, tensor):
        hit_rate     = float(tensor[2])
        storage_util = float(tensor[10])
        spec_depth   = 6 if hit_rate > 0.5 else 4
        return UnifiedDecisionVector(
            cache_admission=storage_util < 0.85,
            admission_priority=hit_rate,
            eviction_target=None, pre_warm_predictions=[],
            route_decision=None, batch_group=None,
            throttle_back_pressure=0.0,
            inference_ms=2.0, is_fallback=False, domain="unified",
            spec_decode_enable=True, spec_decode_depth=spec_depth,
            spec_mode="v2_clean", spec_v3_block_size=8,
            draft_temperature=0.8, kv_compression_enable=False,
            kv_compression_ratio=1.0, decode_batch_priority=2,
            early_stop_confidence=0.0, cuda_graph_hint=True,
            adaptive_depth_override=False,
        )
    def load(self): pass
    def unload(self): pass

# ── build reasoning stack ────────────────────────────────────────────────────
db_path     = str(Path(tempfile.mkdtemp()) / "rewards.db")
collector   = MetricsCollector()
model_r     = MockReasoningModel()
executor    = DecisionExecutor(coordinator_client=None)
reward_calc = RewardCalculator(db_path=db_path)
loop        = LearningLoop(reasoning_model=model_r, reward_calculator=reward_calc, rl_level=0)
orchestrator = Orchestrator(collector, model_r, executor, reward_calc, loop, enabled=True)
orchestrator.start()

# ── build adapter ─────────────────────────────────────────────────────────────
adapter = Tier1LocalKorithAdapter(model_path=MODEL_PATH)
log.info("Adapter ready: %s", MODEL_PATH)

# ── helpers ───────────────────────────────────────────────────────────────────
_run_counter = 0

def make_artifacts(tag: str) -> dict:
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

DET_CFG = {"seed": 42, "n_ctx": 4096, "n_batch": 512}
POLICY  = {"allow_amf_reuse": True, "_tenant_id": "test"}

# 4K context prompt — ~900 tokens of real text
PROMPT = (
    "You are an expert in distributed systems. "
    "Explain the role of KV cache in transformer inference. "
    + "The KV cache stores key and value tensors computed during the attention mechanism. " * 60
    + "Question: What is the primary benefit of KV cache reuse?"
)

# ── run test ──────────────────────────────────────────────────────────────────
prefix_hash = hashlib.sha256(PROMPT[:200].encode()).hexdigest()[:16]

print("\n" + "="*72)
print("  AMF + AI REASONING  —  10 runs, 4K context, Llama 3.2 1B")
print("="*72)
print(f"  {'Run':<5} {'Mode':<12} {'Total ms':>10} {'Prefill ms':>11} {'Decode ms':>10}  {'AI Decision'}")
print("-"*72)

baseline_ms = None
mf_snapshot = None  # path to stored KV snapshot

for i in range(1, 11):
    # AI reasoning decision before the run
    outcome = orchestrator.on_request(
        prefix_hash=prefix_hash,
        tenant_id="test",
        node_id="gpu-0",
        worker_id="worker-0",
    )

    spec_k_label = f"specK={outcome.spec_k}" if outcome.spec_k else ""

    if i == 1:
        # Run 1: baseline (no AMF) — establishes cold time
        mode_label = "BASELINE"
        artifacts = make_artifacts("baseline")
        t0 = time.perf_counter()
        result = adapter.run_baseline(
            prompt=PROMPT,
            max_tokens=32,
            deterministic_cfg=DET_CFG,
            policy={**POLICY, "allow_amf_reuse": False},
            artifacts=artifacts,
            mf_snapshot_in=None,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        metrics = result.get("engine_metrics", {})
        perf    = metrics.get("perf", {})
        total   = float(perf.get("total_ms", elapsed) or elapsed)
        prefill = float(perf.get("prefill_ms", 0) or 0)
        decode  = float(perf.get("decode_ms", 0) or 0)
        baseline_ms = total

        orchestrator.on_miss(prefix_hash)
        collector.record_miss(prefix_hash)
        collector.record_decode(decode_ms=decode or total)

    else:
        # Runs 2-10: AMF accel
        mode_label = "AMF COLD" if i == 2 else "AMF WARM"
        artifacts = make_artifacts(f"accel_{i}")

        # Pass previous snapshot in if we have one
        snap_in = mf_snapshot if (mf_snapshot and Path(mf_snapshot).exists()) else None

        t0 = time.perf_counter()
        result = adapter.run_accel(
            prompt=PROMPT,
            max_tokens=32,
            deterministic_cfg=DET_CFG,
            policy=POLICY,
            artifacts=artifacts,
            mf_snapshot_in=snap_in,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        metrics = result.get("engine_metrics", {})
        perf    = metrics.get("perf", {})
        total   = float(perf.get("total_ms", elapsed) or elapsed)
        prefill = float(perf.get("prefill_ms", 0) or 0)
        decode  = float(perf.get("decode_ms", 0) or 0)

        # Store snapshot path for next run
        snap_out = artifacts["mf_snapshot"]
        if Path(snap_out).exists():
            mf_snapshot = snap_out

        if prefill == 0.0 and i > 2:
            orchestrator.on_hit(prefix_hash)
            collector.record_hit(prefix_hash=prefix_hash, roi=3.0, restore_ms=10.0)
        else:
            orchestrator.on_miss(prefix_hash)
            collector.record_miss(prefix_hash)
        collector.record_decode(decode_ms=decode or total)

    exit_code = result.get("exit_code", -1)
    if exit_code != 0:
        errors = result.get("engine_errors", [])
        print(f"  {i:<5} {mode_label:<12} {'ERROR':>10}   exit={exit_code} {errors}")
        continue

    savings = ""
    if baseline_ms and i > 1:
        pct = 100.0 * (1.0 - total / baseline_ms)
        savings = f"  ({pct:+.1f}%)"

    ai_tag = f"AI✓ {spec_k_label}" if outcome.ai_augmented else "pass"

    print(f"  {i:<5} {mode_label:<12} {total:>10.0f} {prefill:>11.0f} {decode:>10.0f}  {ai_tag}{savings}")

print("-"*72)

# ── summary ───────────────────────────────────────────────────────────────────
stats   = orchestrator.get_stats()
metrics = collector.get_metrics_dict()

print(f"\n  AI augmented     : {stats.ai_augmented_requests}/{stats.total_requests} requests")
print(f"  Hit rate         : {metrics['hit_rate']:.1%}")
print(f"  ROI EMA          : {metrics['roi_ema_norm']:.3f}")

if baseline_ms:
    print(f"\n  Baseline (cold)  : {baseline_ms:.0f} ms")

print(f"\n  AMF store        : {AMF_PATH}")
amf_files = list(Path(AMF_PATH).glob("*"))
print(f"  AMF entries      : {len(amf_files)} files")

print()
orchestrator.stop()
