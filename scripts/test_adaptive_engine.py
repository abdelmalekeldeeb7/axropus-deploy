#!/usr/bin/env python3
"""
Adaptive Engine smoke test — 50 runs, 3 tenants, no external calls.
Shows per-tenant learning, bandit convergence, hysteresis admission.
"""
from __future__ import annotations
import hashlib, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from platform.reasoning.metrics_collector import MetricsCollector, Slot
from platform.reasoning.adaptive_engine import AdaptiveEngine

import time
collector = MetricsCollector()
collector.start()
engine    = AdaptiveEngine()  # no AI model — pure local

TENANTS = ["nebius-agent-fleet", "customer-a", "customer-b"]
PREFIXES = {t: [hashlib.sha256(f"{t}_{i}".encode()).hexdigest()[:16] for i in range(4)]
            for t in TENANTS}

print("\n" + "="*75)
print("  Adaptive Engine — 50 runs, 3 tenants, zero external calls")
print("="*75)
print(f"  {'Run':<5} {'Tenant':<22} {'Prefix':<10} {'Hit?':<6} {'Admit':<7} {'SpecK':<7} {'Throttle':<10} {'PreWarm'}")
print("-"*75)

# Simulate a realistic workload pattern
pattern = []
for i in range(50):
    tenant = TENANTS[i % 3]
    pidx   = (i // 3) % 4
    ph     = PREFIXES[tenant][pidx]
    is_hit = i > 5 and (i % 7 != 0)  # mostly hits after warmup, occasional miss
    pattern.append((tenant, ph, is_hit))

for i, (tenant, ph, is_hit) in enumerate(pattern, 1):
    if is_hit:
        collector.record_hit(prefix_hash=ph, roi=3.0, restore_ms=8.0)
    else:
        collector.record_miss(prefix_hash=ph)
    collector.record_decode(decode_ms=180.0 if is_hit else 240.0)

    time.sleep(0.12)  # let collector's 100ms poll thread rebuild tensor
    tensor  = collector.get_metrics_tensor()
    dv      = engine.decide(tensor, tenant_id=tenant, prefix_hash=ph)

    # Feed outcome back
    engine.record_outcome(
        prefix_hash=ph, tenant_id=tenant,
        spec_depth=dv.spec_decode_depth,
        admitted=dv.cache_admission,
        was_hit=is_hit,
        acceptance_rate=0.85 if is_hit else 0.0,
    )

    prewarm = f"{len(dv.pre_warm_predictions)}px" if dv.pre_warm_predictions else "—"
    admit   = "YES" if dv.cache_admission else "NO"
    tag     = "HIT " if is_hit else "MISS"

    if i <= 15 or i % 10 == 0:
        print(f"  {i:<5} {tenant:<22} p{ph[:6]:<9} {tag:<6} {admit:<7} {dv.spec_decode_depth:<7} {dv.throttle_back_pressure:<10.3f} {prewarm}")

print("-"*75)

# Final stats
stats = engine.get_stats()
m     = collector.get_metrics_dict()

print(f"\n  Total decisions   : {stats['total_decisions']}")
print(f"  Local decisions   : {stats['local_decisions']}  (AI calls: {stats['ai_calls']})")
print(f"  Final hit rate    : {m['hit_rate']:.1%}")
print(f"  Tenants tracked   : {stats['tenant_count']}")

print(f"\n  Per-tenant state:")
for tid, ts in stats["tenants"].items():
    print(f"    {tid:<24} hit_ema={ts['hit_rate_ema']:.2f}  roi={ts['roi_ema']:.2f}  "
          f"admit={ts['admit_state']}  best_depth={ts['best_arm']}  "
          f"prefix_memory={ts['prefix_memory']}")

print()
collector.stop()
