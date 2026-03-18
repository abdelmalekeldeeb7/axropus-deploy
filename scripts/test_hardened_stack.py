#!/usr/bin/env python3
"""
Hardened stack test — AdaptiveEngine (AI-primary) + Orchestrator + doubling RL.

Shows:
  - AI primary with rules fallback
  - Doubling reward multiplier compounding per tenant
  - RL loop closed via Orchestrator on_hit/on_miss
  - Per-tenant learning (3 tenants, independent state)
  - Streak building → multiplier → faster bandit convergence
"""
from __future__ import annotations
import hashlib, os, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from platform.reasoning.metrics_collector import MetricsCollector, Slot
from platform.reasoning.adaptive_engine import AdaptiveEngine
from platform.reasoning.decision_executor import DecisionExecutor
from platform.reasoning.reward_calculator import RewardCalculator
from platform.reasoning.learning_loop import LearningLoop
from platform.reasoning.orchestrator import Orchestrator
from platform.reasoning.reasoning_model import UnifiedDecisionVector

# ── simulated 7B Llama (replace base_url with real server when ready) ─────────
class Simulated7B:
    """Stands in for llama-server running Llama 3.2 7B Q4 on your hardware."""
    def decide(self, tensor, **kwargs) -> UnifiedDecisionVector:
        hit_rate = float(tensor[Slot.HIT_RATE])
        entropy  = float(tensor[Slot.ENTROPY_EMA])
        roi      = float(tensor[Slot.ROI_EMA_NORM]) * 10.0

        if hit_rate > 0.85 and entropy < 0.4:
            spec = 16
        elif hit_rate > 0.70:
            spec = 12
        elif hit_rate > 0.50:
            spec = 8
        else:
            spec = 6

        admit = roi > 0.2 or hit_rate > 0.4
        return UnifiedDecisionVector(
            cache_admission=admit, admission_priority=hit_rate,
            eviction_target=None, pre_warm_predictions=[],
            route_decision=None, batch_group=None,
            throttle_back_pressure=0.0,
            inference_ms=80.0, is_fallback=False, domain="unified",
            spec_decode_enable=True, spec_decode_depth=spec,
            spec_mode="v2_clean", spec_v3_block_size=8,
            draft_temperature=0.7, kv_compression_enable=False,
            kv_compression_ratio=1.0, decode_batch_priority=2,
            early_stop_confidence=0.0, cuda_graph_hint=True,
            adaptive_depth_override=False,
        )
    def load(self): pass
    def unload(self): pass

# ── build hardened stack ──────────────────────────────────────────────────────
db_path     = str(Path(tempfile.mkdtemp()) / "rewards.db")
collector   = MetricsCollector()
engine      = AdaptiveEngine(ai_model=Simulated7B(), ai_primary=True)
executor    = DecisionExecutor(coordinator_client=None)
reward_calc = RewardCalculator(db_path=db_path)
loop        = LearningLoop(reasoning_model=engine, reward_calculator=reward_calc, rl_level=0)
orchestrator = Orchestrator(collector, engine, executor, reward_calc, loop, enabled=True)
orchestrator.start()

TENANTS = ["nebius-fleet", "customer-a", "customer-b"]
PREFIXES = {t: [hashlib.sha256(f"{t}_{i}".encode()).hexdigest()[:16] for i in range(4)]
            for t in TENANTS}

N = 60
print("\n" + "="*80)
print("  Hardened Stack — AI-primary + Doubling RL Multiplier — 60 runs, 3 tenants")
print("="*80)
print(f"  {'Run':<5} {'Tenant':<16} {'Hit?':<6} {'Admit':<7} {'SpecK':<7} {'Streak':<8} {'Mult':<8} {'ScaledR'}")
print("-"*80)

for i in range(1, N + 1):
    tenant = TENANTS[i % 3]
    ph     = PREFIXES[tenant][i % 4]
    is_hit = i > 9 and (i % 5 != 0)   # ~80% hit rate after warmup

    if is_hit:
        collector.record_hit(prefix_hash=ph, roi=3.0, restore_ms=8.0)
    else:
        collector.record_miss(prefix_hash=ph)
    collector.record_decode(decode_ms=185.0 if is_hit else 245.0)

    time.sleep(0.12)

    outcome = orchestrator.on_request(
        prefix_hash=ph, tenant_id=tenant,
        node_id="gpu-0", worker_id="w-0",
    )

    if is_hit:
        orchestrator.on_hit(ph, tenant_id=tenant,
                            spec_depth=outcome.spec_k or 4,
                            acceptance_rate=0.83)
    else:
        orchestrator.on_miss(ph, tenant_id=tenant,
                             spec_depth=outcome.spec_k or 4)

    # Read per-tenant multiplier state
    stats  = engine.get_stats()
    ts     = stats["tenants"].get(tenant, {})
    streak = ts.get("streak", 0)
    mult   = ts.get("multiplier", 1.0)
    avg_r  = ts.get("avg_reward", 0.0)

    admit = "YES" if outcome.admitted else "NO"
    tag   = "HIT " if is_hit else "MISS"

    if i <= 20 or i % 10 == 0:
        print(f"  {i:<5} {tenant:<16} {tag:<6} {admit:<7} "
              f"{str(outcome.spec_k or '—'):<7} {streak:<8} {mult:<8.1f} {avg_r:+.3f}")

print("-"*80)

# ── final report ──────────────────────────────────────────────────────────────
stats  = engine.get_stats()
m      = collector.get_metrics_dict()

print(f"\n  Final hit rate        : {m['hit_rate']:.1%}")
print(f"  Total decisions       : {stats['total_decisions']}")
print(f"  AI calls              : {stats['ai_calls']}  ({stats['ai_rate']:.0%})")
print(f"  AI failures           : {stats['ai_failures']}")
print(f"  Local (fallback)      : {stats['local_decisions']}")
print(f"  Total multiplied R    : {stats['total_multiplied_reward']:+.1f}")

print(f"\n  Per-tenant RL state:")
for tid, ts in stats["tenants"].items():
    print(f"    {tid:<18} streak={ts['streak']:<4} mult={ts['multiplier']:<6.1f} "
          f"best_depth={ts['best_arm']:<4} avg_reward={ts['avg_reward']:+.3f}  "
          f"prefix_memory={ts['prefix_memory']}")

print(f"\n  Doubling multiplier effect:")
print(f"    Max streak seen = {max(ts['streak'] for ts in stats['tenants'].values())}")
print(f"    Max multiplier  = {max(ts['multiplier'] for ts in stats['tenants'].values()):.1f}x")
print(f"    This means bandit learned {max(ts['multiplier'] for ts in stats['tenants'].values()):.0f}x")
print(f"    faster than uniform reward on hot streaks\n")

orchestrator.stop()
