#!/usr/bin/env python3
"""
Quick 10-run AI reasoning smoke test — no GPU, no Llama model needed.

Tests the full orchestrator pipeline:
  MetricsCollector → (mock) ReasoningModel → DecisionExecutor → Orchestrator
  + RewardCalculator scoring + LearningLoop stub

Simulates 10 requests at 4K context (mix of hits and misses) and prints
every decision the reasoning layer makes.
"""
from __future__ import annotations

import logging
import sys
import time
import tempfile
import os
import hashlib
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reasoning_test")

# ── import reasoning layer ───────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("numpy not installed — run: pip install numpy")

from platform.reasoning.metrics_collector import MetricsCollector
from platform.reasoning.reasoning_model import UnifiedDecisionVector
from platform.reasoning.decision_executor import DecisionExecutor, ExecutionContext
from platform.reasoning.reward_calculator import RewardCalculator
from platform.reasoning.learning_loop import LearningLoop
from platform.reasoning.orchestrator import Orchestrator

log.info("All reasoning components imported successfully")


# ── mock ReasoningModel ──────────────────────────────────────────────────────

class MockReasoningModel:
    """
    Simulates Llama 8B NF4 decisions without needing transformers.
    Produces realistic adaptive decisions based on the metrics tensor.
    """

    def decide(self, tensor: "np.ndarray") -> UnifiedDecisionVector:
        hit_rate    = float(tensor[2])   # Slot.HIT_RATE
        storage_util = float(tensor[10]) # Slot.STORAGE_UTIL
        miss_streak = float(tensor[24])  # Slot.MISS_STREAK_NORM

        # Adaptive spec decode depth: higher when hit rate is good
        if hit_rate > 0.7:
            spec_depth = 6
        elif hit_rate > 0.4:
            spec_depth = 4
        else:
            spec_depth = 2

        # Admission: don't admit if storage almost full
        admit = storage_util < 0.85

        # Throttle only when miss streak is bad
        throttle = min(0.3 * miss_streak, 0.5)

        t0 = time.monotonic()
        # Simulate 1-3ms "inference"
        time.sleep(0.002)
        inference_ms = (time.monotonic() - t0) * 1000

        return UnifiedDecisionVector(
            cache_admission=admit,
            admission_priority=hit_rate,
            eviction_target=None,
            pre_warm_predictions=[],
            route_decision=None,
            batch_group=None,
            throttle_back_pressure=throttle,
            inference_ms=inference_ms,
            is_fallback=False,
            domain="unified",
            # decode fields
            spec_decode_enable=True,
            spec_decode_depth=spec_depth,
            spec_mode="v2_clean",
            spec_v3_block_size=8,
            draft_temperature=0.8,
            kv_compression_enable=False,
            kv_compression_ratio=1.0,
            decode_batch_priority=2,
            early_stop_confidence=0.0,
            cuda_graph_hint=True,
            adaptive_depth_override=False,
        )

    def load(self): pass
    def unload(self): pass


# ── build the stack ───────────────────────────────────────────────────────────

db_path = str(Path(tempfile.mkdtemp()) / "rewards_test.db")
log.info("RewardCalculator DB: %s", db_path)

collector    = MetricsCollector()          # no coordinator, pure local
model        = MockReasoningModel()
executor     = DecisionExecutor(coordinator_client=None)
reward_calc  = RewardCalculator(db_path=db_path)
loop         = LearningLoop(reasoning_model=model, reward_calculator=reward_calc, rl_level=0)
orchestrator = Orchestrator(
    collector, model, executor, reward_calc, loop,
    enabled=True, ab_testing=False,
)

orchestrator.start()
log.info("Orchestrator started")

# ── simulate 10 requests ──────────────────────────────────────────────────────

# 4K context — these are the prefixes we'll reuse to simulate AMF hits
PREFIXES = [
    hashlib.sha256(f"system_prompt_4k_v{i}".encode()).hexdigest()[:16]
    for i in range(3)
]

# run pattern: miss on first see, hit on repeat
run_pattern = [
    # (prefix_idx, is_hit, label)
    (0, False, "COLD  run 1 — new prefix"),
    (0, True,  "WARM  run 2 — AMF hit"),
    (0, True,  "WARM  run 3 — AMF hit"),
    (1, False, "COLD  run 4 — new prefix"),
    (1, True,  "WARM  run 5 — AMF hit"),
    (2, False, "COLD  run 6 — new prefix"),
    (2, True,  "WARM  run 7 — AMF hit"),
    (0, True,  "WARM  run 8 — AMF hit (prefix 0 again)"),
    (1, True,  "WARM  run 9 — AMF hit (prefix 1 again)"),
    (2, True,  "WARM  run 10 — AMF hit (prefix 2 again)"),
]

print("\n" + "="*72)
print(f"  AI REASONING — 10-run smoke test  (4K context, 3 prefixes)")
print("="*72)
print(f"  {'Run':<5} {'Label':<40} {'Admitted':<10} {'SpecK':<7} {'Throttle':<10} {'AI?'}")
print("-"*72)

for i, (prefix_idx, is_hit, label) in enumerate(run_pattern, 1):
    prefix_hash = PREFIXES[prefix_idx]

    # Feed hit/miss signal into collector before the decision
    if is_hit:
        collector.record_hit(prefix_hash=prefix_hash, roi=2.5, restore_ms=8.0)
        # Simulate 4K decode: ~180ms
        collector.record_decode(decode_ms=180.0)
    else:
        collector.record_miss(prefix_hash=prefix_hash)
        # Cold: simulate 4K prefill + decode: ~400ms
        collector.record_decode(decode_ms=400.0)

    outcome = orchestrator.on_request(
        prefix_hash=prefix_hash,
        tenant_id="test-tenant",
        node_id="gpu-0",
        worker_id="worker-0",
    )

    # Report outcome back to orchestrator
    if is_hit:
        orchestrator.on_hit(prefix_hash)
    else:
        orchestrator.on_miss(prefix_hash)

    spec_k   = outcome.spec_k if outcome.spec_k is not None else "—"
    throttle = f"{outcome.throttle:.2f}" if outcome.throttle else "0.00"
    ai_tag   = "✓ AI" if outcome.ai_augmented else "pass"

    print(f"  {i:<5} {label:<40} {'YES' if outcome.admitted else 'NO':<10} {str(spec_k):<7} {throttle:<10} {ai_tag}")

    time.sleep(0.05)  # let metrics settle

print("-"*72)

# ── final stats ──────────────────────────────────────────────────────────────
stats = orchestrator.get_stats()
metrics = collector.get_metrics_dict()

print(f"\n  Total requests  : {stats.total_requests}")
print(f"  AI augmented    : {stats.ai_augmented_requests}")
print(f"  Hit rate        : {metrics['hit_rate']:.1%}")
print(f"  Reuse ratio     : {metrics['reuse_ratio']:.1%}")
print(f"  ROI EMA         : {metrics['roi_ema_norm']:.3f}")
print(f"  Avg spec depth  : {metrics['current_spec_depth']:.1f}")

try:
    reward_stats = reward_calc.get_stats()
    print(f"\n  Reward stats    : {reward_stats}")
except Exception as e:
    print(f"\n  Reward stats    : (unavailable: {e})")

print("\n  ✓ AI reasoning layer working — all 10 runs completed\n")

orchestrator.stop()
