#!/usr/bin/env python3
"""
AI-primary adaptive engine — decode improvement test.

Compares 3 modes over 60 requests:
  Mode A: Rules only (no AI)
  Mode B: AI in uncertainty band only (~30% of requests)
  Mode C: AI primary, rules as fallback (Option B — what we just built)

AI is simulated as a smart 7B Llama that reads entropy + hit_rate and
picks deeper spec depths more aggressively than the rule engine.
Decode savings are calculated from spec_decode_depth × acceptance_rate.

Spec decode savings model (from literature):
  depth 2  → ~10% decode reduction
  depth 4  → ~20% decode reduction
  depth 6  → ~30% decode reduction
  depth 8  → ~40% decode reduction
  depth 12 → ~52% decode reduction
  depth 16 → ~60% decode reduction
  (at 0.80 acceptance rate — drops proportionally at lower acceptance)
"""
from __future__ import annotations
import hashlib, sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from platform.reasoning.metrics_collector import MetricsCollector, Slot
from platform.reasoning.adaptive_engine import AdaptiveEngine
from platform.reasoning.reasoning_model import UnifiedDecisionVector

# ── simulated 7B Llama AI model ───────────────────────────────────────────────
class Simulated7BModel:
    """
    Simulates a fine-tuned 7B Llama making AMF decisions.
    More aggressive on spec depth than rules — reads entropy + acceptance EMA.
    In production: replace base_url with your llama-server endpoint.
    """
    def decide(self, tensor: np.ndarray) -> UnifiedDecisionVector:
        hit_rate    = float(tensor[Slot.HIT_RATE])
        roi_ema     = float(tensor[Slot.ROI_EMA_NORM]) * 10.0
        entropy     = float(tensor[Slot.ENTROPY_EMA])
        accept_ema  = float(tensor[Slot.ACCEPTANCE_EMA])
        miss_streak = float(tensor[Slot.MISS_STREAK_NORM])
        storage     = float(tensor[Slot.STORAGE_UTIL])

        # Smarter admission — uses ROI + hit trend together
        admit = (roi_ema > 0.2 or hit_rate > 0.4) and storage < 0.90

        # Deeper spec depth — AI is more confident at high hit rates
        if hit_rate > 0.85 and entropy < 0.4:
            spec = 16
        elif hit_rate > 0.70 and entropy < 0.55:
            spec = 12
        elif hit_rate > 0.55:
            spec = 8
        elif hit_rate > 0.35:
            spec = 6
        else:
            spec = 4

        # Lower draft temperature when acceptance EMA is high
        draft_temp = max(0.4, 0.9 - accept_ema * 0.5)

        throttle = min(miss_streak * 0.2, 0.3)

        return UnifiedDecisionVector(
            cache_admission=admit, admission_priority=hit_rate,
            eviction_target=None, pre_warm_predictions=[],
            route_decision=None, batch_group=None,
            throttle_back_pressure=throttle,
            inference_ms=80.0,   # ~80ms for 7B Q4 on consumer GPU
            is_fallback=False, domain="unified",
            spec_decode_enable=True, spec_decode_depth=spec,
            spec_mode="v2_clean", spec_v3_block_size=8,
            draft_temperature=draft_temp,
            kv_compression_enable=False, kv_compression_ratio=1.0,
            decode_batch_priority=2, early_stop_confidence=0.0,
            cuda_graph_hint=True, adaptive_depth_override=False,
        )

    def load(self): pass
    def unload(self): pass


# ── spec decode savings model ─────────────────────────────────────────────────
_DEPTH_SAVINGS = {2: 0.10, 4: 0.20, 6: 0.30, 8: 0.40, 12: 0.52, 16: 0.60}

def decode_savings(spec_depth: int, acceptance: float = 0.80) -> float:
    """Estimated decode time reduction at given spec depth and acceptance rate."""
    base = _DEPTH_SAVINGS.get(spec_depth, 0.20)
    return base * min(acceptance / 0.80, 1.0)


# ── run one mode ──────────────────────────────────────────────────────────────
def run_mode(name: str, engine: AdaptiveEngine, n: int = 60) -> dict:
    collector = MetricsCollector()
    collector.start()

    PREFIXES = [hashlib.sha256(f"p{i}".encode()).hexdigest()[:16] for i in range(5)]
    total_savings = 0.0
    depths = []
    admits = 0
    baseline_decode_ms = 190.0  # warm hit decode baseline

    for i in range(n):
        ph     = PREFIXES[i % 5]
        is_hit = i > 8 and (i % 6 != 0)   # ~83% hit rate after warmup

        if is_hit:
            collector.record_hit(prefix_hash=ph, roi=3.0, restore_ms=8.0)
        else:
            collector.record_miss(prefix_hash=ph)
        collector.record_decode(decode_ms=baseline_decode_ms if is_hit else 240.0)

        time.sleep(0.12)  # let poll thread rebuild tensor
        tensor = collector.get_metrics_tensor()
        dv     = engine.decide(tensor, tenant_id="test", prefix_hash=ph)

        engine.record_outcome(
            prefix_hash=ph, tenant_id="test",
            spec_depth=dv.spec_decode_depth,
            admitted=dv.cache_admission,
            was_hit=is_hit,
            acceptance_rate=0.82 if is_hit else 0.0,
        )

        if is_hit:
            savings = decode_savings(dv.spec_decode_depth, acceptance=0.82)
            total_savings += savings
            depths.append(dv.spec_decode_depth)
            if dv.cache_admission:
                admits += 1

    collector.stop()
    stats = engine.get_stats()
    m     = collector.get_metrics_dict()

    hits = sum(1 for i in range(n) if i > 8 and (i % 6 != 0))
    avg_savings  = total_savings / max(1, len(depths))
    avg_depth    = sum(depths) / max(1, len(depths))
    decode_saved = avg_savings * baseline_decode_ms

    return {
        "name":         name,
        "hit_rate":     m["hit_rate"],
        "avg_depth":    round(avg_depth, 1),
        "avg_savings":  round(avg_savings * 100, 1),
        "decode_saved": round(decode_saved, 1),
        "ai_calls":     stats["ai_calls"],
        "ai_rate":      stats["ai_rate"],
        "admits":       admits,
    }


# ── main ──────────────────────────────────────────────────────────────────────
ai = Simulated7BModel()

modes = [
    ("Rules only",          AdaptiveEngine(ai_model=None)),
    ("AI uncertainty band", AdaptiveEngine(ai_model=ai, ai_primary=False)),
    ("AI primary (Opt B)",  AdaptiveEngine(ai_model=ai, ai_primary=True)),
]

print("\n" + "="*72)
print("  Decode Improvement — Rules vs AI band vs AI primary")
print("="*72)
print(f"  {'Mode':<24} {'HitRate':>8} {'AvgDepth':>9} {'DecSaved%':>10} {'DecodeSaved':>12} {'AI%':>6}")
print("-"*72)

results = []
for name, engine in modes:
    print(f"  Running: {name}...", end="", flush=True)
    r = run_mode(name, engine, n=60)
    results.append(r)
    print(f"\r  {r['name']:<24} {r['hit_rate']:>7.1%} {r['avg_depth']:>9.1f} "
          f"{r['avg_savings']:>9.1f}% {r['decode_saved']:>10.1f}ms {r['ai_rate']:>5.0%}")

print("-"*72)

# Improvement over rules-only baseline
base = results[0]
print(f"\n  Baseline decode (warm hit): 190ms")
print(f"\n  vs Rules only:")
for r in results[1:]:
    extra = r["decode_saved"] - base["decode_saved"]
    depth_gain = r["avg_depth"] - base["avg_depth"]
    print(f"    {r['name']:<24} +{extra:.1f}ms decode saved  "
          f"(depth +{depth_gain:.1f})  AI called {r['ai_rate']:.0%} of requests")

print(f"\n  AI-primary effective decode: {190 - results[-1]['decode_saved']:.0f}ms "
      f"(was {190 - base['decode_saved']:.0f}ms rules-only)")
print(f"  Additional speedup from AI:  {(190 - base['decode_saved']) / max(1, 190 - results[-1]['decode_saved']):.2f}x\n")
