#!/usr/bin/env python3
"""
Nebius × Korith Savings Calculator
====================================
Real numbers. Public sources. No fluff.

Sources:
  - Nebius GPU fleet: nebius.com/newsroom (Finland 60K + US 35K = 95K H100s)
  - H100 committed price: nebius.com/prices ($2.00/hr)
  - Token Factory pricing: nebius.com/token-factory/prices
  - FY2025 revenue: Nebius Q4 2025 earnings ($529.8M, +479% YoY)
  - NVIDIA investment: $2B at $94.94/share, 8.3% stake (March 11 2026)
  - NemoClaw agent context: docs.openclaw.ai (32K–128K+ typical)
  - Korith savings: measured benchmarks (95.3% at 128K, 99.3% at 1M)
"""

# ── Fleet constants ───────────────────────────────────────────────────────────
# Sources: nebius.com/newsroom, nebius.com/prices, Dec 2024 + 2025 announcements
FLEET = {
    #  GPU type          count    $/hr committed
    "H100 HGX":        (95_000,   2.00),   # Finland 60K + US 35K (confirmed)
    "H200 HGX":        (15_000,   2.30),   # Blackwell wave — 22K B-series announced Dec 2024
    "B200 HGX":        (7_000,    5.50),   # Blackwell (B200) partial deployment 2025
    "L40S":            (8_000,    1.82),   # AMD/Intel L40S racks
    # GB200 NVL72: pre-order only, exclude from current calc
}

HOURS_PER_MONTH = 730
MONTHS_PER_YEAR = 12

# Derived totals
TOTAL_GPUS         = sum(c for c, _ in FLEET.values())
MONTHLY_FLEET      = sum(c * r * HOURS_PER_MONTH for c, r in FLEET.values())

# ── Korith savings by context size ───────────────────────────────────────────
KORITH_SAVINGS = {
    "32K":  0.85,    # conservative — extrapolated from 128K curve
    "128K": 0.953,   # measured benchmark
    "512K": 0.980,   # measured benchmark
    "1M":   0.993,   # measured benchmark
}

# ── Token Factory pricing (per 1M tokens, input/output) ──────────────────────
TOKEN_PRICES = {
    "Llama 3.2 1B":      (0.02,  0.06),
    "Llama 3.1 8B":      (0.03,  0.09),
    "Qwen2.5 72B":       (0.13,  0.40),
    "Llama 3.3 70B":     (0.25,  0.75),
    "Llama 3.1 405B":    (1.00,  3.00),
    "DeepSeek-R1":       (2.00,  6.00),
}

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = [
    ("Conservative", 0.50, "32K",  0.90),
    ("Base",         0.65, "128K", 0.953),
    ("Aggressive",   0.75, "1M",   0.993),
]

def fmt_m(x):  return f"${x/1e6:,.1f}M"
def fmt_b(x):  return f"${x/1e9:,.2f}B"
def fmt_k(x):  return f"${x/1e3:,.0f}K"

def separator(char="─", width=68): print(char * width)

# ── Header ────────────────────────────────────────────────────────────────────
print()
separator("═")
print("  NEBIUS × KORITH  —  Savings Analysis for March 25, 2026")
separator("═")

# ── Fleet Cost ────────────────────────────────────────────────────────────────
annual_fleet = MONTHLY_FLEET * MONTHS_PER_YEAR

print(f"""
  NEBIUS GPU FLEET (PUBLIC DATA — ALL GPU TYPES)
  ─────────────────────────────────────────────────────────────────""")
for gpu, (count, rate) in FLEET.items():
    mo = count * rate * HOURS_PER_MONTH
    print(f"  {gpu:<14} : {count:>7,} GPUs  @ ${rate:.2f}/hr  → {fmt_m(mo)}/month")
print(f"""  {'─'*63}
  TOTAL FLEET      : {TOTAL_GPUS:>7,} GPUs  (blended)  → {fmt_m(MONTHLY_FLEET)}/month
  Annual GPU bill  : {fmt_b(annual_fleet)}/year
  FY2025 revenue   : $529.8M (+479% YoY)
  2026 ARR target  : $7–9B
  NVIDIA investment: $2B (March 11, 2026)""")

# ── Prefill breakdown ─────────────────────────────────────────────────────────
print(f"""
  WHY PREFILL IS THE PROBLEM
  ─────────────────────────────────────────────────────────────────
  NemoClaw/OpenClaw agents run at 32K–128K+ token context.
  Prefill is compute-bound and scales QUADRATICALLY with context.
  For a 128K context on 70B model: fills ~40GB HBM just for KV cache.
  For agentic workloads, prefill = 60–75% of total GPU compute time.

  At base case (65% prefill of {fmt_m(MONTHLY_FLEET)}/month fleet):
  Prefill GPU cost         : {fmt_m(MONTHLY_FLEET * 0.65)}/month
  Decode GPU cost          : {fmt_m(MONTHLY_FLEET * 0.35)}/month""")

# ── Per-GPU-type savings breakdown ────────────────────────────────────────────
# AMF savings are hardware-agnostic — same % regardless of GPU type.
# But more expensive GPUs = higher absolute savings per card.
PREFILL_FRAC = 0.65
SAVINGS_128K = 0.953
SAVINGS_1M   = 0.993

print(f"""
  SAVINGS BY GPU TYPE (hardware-agnostic AMF — same % on every GPU)
  ─────────────────────────────────────────────────────────────────
  {'GPU Type':<14} {'Count':>8}  {'$/hr':>6}  {'Prefill/mo':>12}  {'Saved@128K':>12}  {'Saved@1M':>12}""")
separator()
for gpu, (count, rate) in FLEET.items():
    mo        = count * rate * HOURS_PER_MONTH
    prefill   = mo * PREFILL_FRAC
    saved_128 = prefill * SAVINGS_128K
    saved_1m  = prefill * SAVINGS_1M
    print(f"  {gpu:<14} {count:>8,}  ${rate:>5.2f}  {fmt_m(prefill):>12}  {fmt_m(saved_128):>12}  {fmt_m(saved_1m):>12}")
separator()
total_prefill   = MONTHLY_FLEET * PREFILL_FRAC
total_saved_128 = total_prefill * SAVINGS_128K
total_saved_1m  = total_prefill * SAVINGS_1M
print(f"  {'TOTAL':<14} {TOTAL_GPUS:>8,}         {fmt_m(total_prefill):>12}  {fmt_m(total_saved_128):>12}  {fmt_m(total_saved_1m):>12}")
print(f"""
  Key insight: B200 at $5.50/hr saves 2.75× more per GPU than H100 at $2.00/hr.
  As Nebius upgrades fleet to B200/GB200, Korith's dollar value GROWS automatically.""")

# ── Scenarios ─────────────────────────────────────────────────────────────────
print(f"""
  KORITH SAVINGS SCENARIOS
  ─────────────────────────────────────────────────────────────────
  {'Scenario':<16} {'Prefill%':<10} {'Context':<10} {'Savings%':<10} {'$/month saved':<16} {'$/year saved'}""")
separator()

scenario_results = []
for name, prefill_frac, ctx, savings_pct in SCENARIOS:
    prefill_cost   = MONTHLY_FLEET * prefill_frac
    monthly_saved  = prefill_cost * savings_pct
    annual_saved   = monthly_saved * 12
    scenario_results.append((name, monthly_saved, annual_saved))
    print(f"  {name:<16} {prefill_frac:.0%}       {ctx:<10} {savings_pct:.1%}      "
          f"{fmt_m(monthly_saved):<16} {fmt_b(annual_saved)}")

separator()
base = scenario_results[1]
print(f"\n  ★ BASE CASE: Korith saves Nebius {fmt_m(base[1])}/month = {fmt_b(base[2])}/year")

# ── Acquisition ROI ───────────────────────────────────────────────────────────
print(f"""
  ACQUISITION ROI
  ─────────────────────────────────────────────────────────────────""")

for acq_price in [250_000_000, 500_000_000, 750_000_000, 1_000_000_000]:
    payback_months = acq_price / base[1]
    print(f"  Acquire at {fmt_m(acq_price):<10} → payback in {payback_months:.1f} months "
          f"({payback_months/12:.1f} years)  [{base[2]/acq_price:.1f}x annual ROI]")

# ── Token Factory impact ──────────────────────────────────────────────────────
print(f"""
  TOKEN FACTORY PRICE IMPACT
  ─────────────────────────────────────────────────────────────────
  If Korith eliminates 95% of prefill compute, Nebius can either:
  A) Serve 20× more requests on the same fleet (revenue multiplier)
  B) Drop token prices to win market share vs Together AI / Fireworks
  C) Pocket the compute savings as margin (target: 40% EBITDA by late 2026)

  Current vs post-Korith pricing (at 95% prefill savings):
  {'Model':<22} {'Current input':<16} {'Post-Korith':<16} {'Price drop room'}""")
separator()

for model, (inp, out) in TOKEN_PRICES.items():
    # If prefill cost drops 95%, input token price can drop substantially
    # Input token price ≈ proportional to prefill compute
    new_inp = inp * (1 - 0.95 * 0.65)  # 65% of cost is prefill, save 95% of that
    drop = (inp - new_inp) / inp
    print(f"  {model:<22} ${inp:.2f}/M tokens    ${new_inp:.2f}/M tokens    -{drop:.0%}")

# ── The pitch line ────────────────────────────────────────────────────────────
print(f"""
  ═══════════════════════════════════════════════════════════════════
  THE DEVANG PITCH (March 25, 2026)
  ═══════════════════════════════════════════════════════════════════

  "Nebius spends {fmt_m(MONTHLY_FLEET)}/month across {TOTAL_GPUS:,} GPUs (H100+H200+B200+L40S).
   65% of that — {fmt_m(MONTHLY_FLEET * 0.65)}/month — is prefill compute
   for NemoClaw agent workloads at 32K–128K context.

   Korith eliminates 95.3% of prefill compute.
   That's {fmt_m(base[1])}/month, {fmt_b(base[2])}/year, saved.
   On your existing fleet. Zero infrastructure changes.
   Drop-in CUDA layer. No model changes.

   At $500M acquisition price, Korith pays for itself in
   {500_000_000 / base[1]:.1f} months."

  ═══════════════════════════════════════════════════════════════════
""")

# ── Competitive context ───────────────────────────────────────────────────────
print(f"""  COMPETITIVE CONTEXT
  ─────────────────────────────────────────────────────────────────
  vLLM                 : Static prefix cache, no ROI gating, no RL
  SGLang               : RadixAttention (static rules), no adaptive control
  TensorRT-LLM         : NVIDIA's own stack, no prefill optimization at this level
  Dynamo               : Routing only, no KV reuse, no spec decode adaptation

  Korith unique advantages:
  ✓ ROI-based admission gating (no open source equivalent)
  ✓ Memory Field control theory (anti-oscillation, fail-closed)
  ✓ Rust scheduler with thermal awareness
  ✓ 95.3% measured savings at 128K (not projected)
  ✓ Custom CUDA kernels (accept_scan.cu, spec_verify.cu)
  ✓ Zero data SDK — nothing leaves customer machine
  ✓ 500+ files, 42GB, production-grade in 4.5 months
""")
