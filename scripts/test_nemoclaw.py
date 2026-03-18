#!/usr/bin/env python3
"""
NemoClaw integration smoke test.
Tests the full stack: MetricsCollector → NemoClaw → DecisionExecutor → Orchestrator
With no API key: uses fallback decisions (same as production when Nebius NIM is live).
With NVIDIA_API_KEY set: calls real NemoClaw API.
"""
from __future__ import annotations
import logging, os, sys, tempfile, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(name)s — %(message)s")
logging.getLogger("nemoclaw_test").setLevel(logging.INFO)
log = logging.getLogger("nemoclaw_test")

import numpy as np
from platform.reasoning.metrics_collector import MetricsCollector
from platform.reasoning.nemoclaw_model import NemoClawReasoningModel
from platform.reasoning.decision_executor import DecisionExecutor
from platform.reasoning.reward_calculator import RewardCalculator
from platform.reasoning.learning_loop import LearningLoop
from platform.reasoning.orchestrator import Orchestrator

api_key  = os.environ.get("NVIDIA_API_KEY", "")
nim_url  = os.environ.get("NEMOCLAW_BASE_URL", "https://integrate.api.nvidia.com/v1")
has_key  = bool(api_key)

log.info("NemoClaw: api_key=%s  url=%s", "SET" if has_key else "NOT SET (fallback mode)", nim_url)

# ── build stack ────────────────────────────────────────────────────────────────
db_path     = str(Path(tempfile.mkdtemp()) / "rewards.db")
collector   = MetricsCollector()
model_r     = NemoClawReasoningModel(api_key=api_key, base_url=nim_url, max_inference_ms=100.0)
executor    = DecisionExecutor(coordinator_client=None)
reward_calc = RewardCalculator(db_path=db_path)
loop        = LearningLoop(reasoning_model=model_r, reward_calculator=reward_calc, rl_level=0)
orchestrator = Orchestrator(collector, model_r, executor, reward_calc, loop, enabled=True)
orchestrator.start()

log.info("Stack ready — NemoClaw replaces Llama 8B + LearningLoop")

# ── simulate 20 requests ───────────────────────────────────────────────────────
PREFIXES = [hashlib.sha256(f"prefix_{i}".encode()).hexdigest()[:16] for i in range(3)]

print("\n" + "="*70)
print(f"  NemoClaw AMF Controller  —  20 runs  (API: {'LIVE' if has_key else 'FALLBACK'})")
print("="*70)
print(f"  {'Run':<5} {'Prefix':<10} {'Hit?':<6} {'Admitted':<10} {'SpecK':<7} {'Throttle':<10} {'Fallback?'}")
print("-"*70)

pattern = [
    (0, False), (0, True), (0, True),
    (1, False), (1, True), (1, True),
    (2, False), (2, True), (2, True), (2, True),
    (0, True),  (1, True), (2, True), (0, True),
    (1, True),  (2, True), (0, True), (1, True),
    (2, True),  (0, True),
]

for i, (pidx, is_hit) in enumerate(pattern, 1):
    ph = PREFIXES[pidx]

    if is_hit:
        collector.record_hit(prefix_hash=ph, roi=3.0, restore_ms=8.0)
    else:
        collector.record_miss(prefix_hash=ph)
    collector.record_decode(decode_ms=190.0 if is_hit else 250.0)

    outcome = orchestrator.on_request(
        prefix_hash=ph, tenant_id="test", node_id="gpu-0", worker_id="w-0"
    )

    # Send reward back to NemoClaw's RL loop
    m = collector.get_metrics_dict()
    if outcome.decision_id:
        model_r.track_decision(
            decision_id=outcome.decision_id,
            prefix_hash=ph,
            spec_depth=outcome.spec_k or 4,
            admitted=outcome.admitted,
        )
        if is_hit:
            orchestrator.on_hit(ph)
            model_r.record_reward(outcome.decision_id, "hit_admitted", +1.0,
                                  hit_rate=m["hit_rate"])
        else:
            orchestrator.on_miss(ph)
            model_r.record_reward(outcome.decision_id, "miss_never_reused", -0.5,
                                  hit_rate=m["hit_rate"])

    spec_k   = str(outcome.spec_k) if outcome.spec_k else "—"
    throttle = f"{outcome.throttle:.2f}"
    fallback = "YES" if not outcome.ai_augmented else "no"
    hit_tag  = "HIT" if is_hit else "MISS"
    admitted = "YES" if outcome.admitted else "NO"

    print(f"  {i:<5} p{pidx:<9} {hit_tag:<6} {admitted:<10} {spec_k:<7} {throttle:<10} {fallback}")

print("-"*70)

# ── stats ──────────────────────────────────────────────────────────────────────
stats    = orchestrator.get_stats()
metrics  = collector.get_metrics_dict()
nc_stats = model_r.get_stats()

print(f"\n  AI augmented    : {stats.ai_augmented_requests}/20")
print(f"  Final hit rate  : {metrics['hit_rate']:.1%}")
print(f"  NemoClaw calls  : {nc_stats['total_calls']}")
print(f"  Fallback calls  : {nc_stats['fallback_calls']}  ({nc_stats['fallback_rate']:.0%})")
print(f"  Rewards sent    : {nc_stats['reward_sent']}")
print(f"  RL history len  : {nc_stats['rl_history_len']}")
print(f"  RL avg reward   : {nc_stats['rl_avg_reward']:+.2f}")
print(f"  RL updates      : {nc_stats['rl_updates']}")
if nc_stats['rl_history_len'] > 0:
    print(f"\n  Last 5 outcomes in RL context window:")
    for e in model_r._reward_history[-5:]:
        print(f"    {e['outcome']:<25} reward={e['reward']:+.1f}  spec={e['spec_depth']}  hit_rate={e['hit_rate']:.1%}")
print(f"\n  NemoClaw RL: {'LIVE API' if has_key else 'in-context only — set NVIDIA_API_KEY for full model'}")
print(f"  To enable: export NVIDIA_API_KEY=nvapi-... && python3 scripts/test_nemoclaw.py\n")

orchestrator.stop()
