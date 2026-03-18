#!/usr/bin/env python3
"""
Nebius Token Factory — Cold vs Warm TTFT Benchmark
=====================================================
Measures real TTFT and cost on Nemotron Ultra 253B via Nebius API.

What this shows:
  - Cold TTFT (no cache): full prefill cost
  - Warm TTFT (repeat request): whether Nebius has ANY prefix caching
  - Cost per request: real billing numbers
  - AMF opportunity: what KV cache reuse WOULD save on top

Usage:
  export NEBIUS_API_KEY="your-key-here"
  python3 scripts/nebius_api_benchmark.py

  # Or with a custom prompt file:
  python3 scripts/nebius_api_benchmark.py --prompt workloads/prompt_128k_tokens.txt

  # Run more warm calls:
  python3 scripts/nebius_api_benchmark.py --runs 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
# Nebius deployment endpoint — axropus.deploy/abdelmalekeldeeb7
# Try deployment URL first, fall back to Token Factory
NEBIUS_DEPLOYMENT_URL = "https://axropus.deploy/abdelmalekeldeeb7/v1"
NEBIUS_TOKEN_FACTORY_URL = "https://api.studio.nebius.com/v1"
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", NEBIUS_DEPLOYMENT_URL)

# Token Factory model IDs
MODELS = {
    "120b": "nvidia/Llama-3_1-Nemotron-70B-Instruct-HF",
    "253b": "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",      # $0.60/1M in, $1.80/1M out
}
DEFAULT_MODEL    = os.environ.get("NEBIUS_MODEL", MODELS["120b"])
# Pricing
INPUT_PRICE_PER_M  = float(os.environ.get("NEBIUS_INPUT_PRICE", "0.60"))
OUTPUT_PRICE_PER_M = float(os.environ.get("NEBIUS_OUTPUT_PRICE", "1.80"))
MAX_OUTPUT_TOKENS  = 128
DEFAULT_RUNS       = 5      # 1 cold + N-1 warm

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--prompt",   default="workloads/prompt_128k_tokens.txt")
parser.add_argument("--model",    default=DEFAULT_MODEL)
parser.add_argument("--runs",     type=int, default=DEFAULT_RUNS)
parser.add_argument("--tokens",   type=int, default=MAX_OUTPUT_TOKENS)
parser.add_argument("--short",    action="store_true", help="Use a short 4K prompt instead")
args = parser.parse_args()

# ── API key ───────────────────────────────────────────────────────────────────
api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
if not api_key:
    print("ERROR: Set NEBIUS_API_KEY environment variable")
    print("  export NEBIUS_API_KEY='your-key-here'")
    sys.exit(1)

# ── Load prompt ───────────────────────────────────────────────────────────────
prompt_path = ROOT / args.prompt
if not prompt_path.exists():
    print(f"ERROR: Prompt file not found: {prompt_path}")
    print("  Run: python3 scripts/h100_1m_benchmark.py --generate-prompt first")
    sys.exit(1)

full_prompt = prompt_path.read_text(encoding="utf-8").strip()
if args.short:
    # Truncate to ~4K tokens (~16K chars)
    full_prompt = full_prompt[:16_000]
    context_label = "4K"
else:
    context_label = "128K"

# Estimate token count (~4 chars per token)
est_input_tokens = len(full_prompt) // 4
print(f"\n  Prompt: {prompt_path.name}  (~{est_input_tokens:,} tokens, {len(full_prompt):,} chars)")

# ── OpenAI-compatible client (Nebius uses OpenAI API format) ──────────────────
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=NEBIUS_BASE_URL)

# ── Cost calculator ───────────────────────────────────────────────────────────
def calc_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000 * INPUT_PRICE_PER_M +
            output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M)

def fmt_ms(ms: float) -> str:
    if ms >= 60_000:
        return f"{ms/60_000:.1f}min"
    if ms >= 1_000:
        return f"{ms/1_000:.2f}s"
    return f"{ms:.0f}ms"

# ── Run benchmark ─────────────────────────────────────────────────────────────
print()
print("═" * 68)
print(f"  Nebius Token Factory Benchmark — {context_label} context")
print(f"  Model: {args.model.split('/')[-1]}")
print("═" * 68)
print(f"\n  {'Run':<6} {'Type':<8} {'TTFT':>10}  {'Total':>10}  {'In tok':>8}  {'Out tok':>8}  {'Cost':>10}  {'Speedup'}")
print("  " + "─" * 70)

results = []
cold_ttft_ms = None

for run_idx in range(1, args.runs + 1):
    run_type = "COLD" if run_idx == 1 else "WARM"

    t_start = time.perf_counter()
    first_token_t = None

    try:
        # Use streaming to measure TTFT accurately
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=args.tokens,
            temperature=0.0,
            stream=True,
        )

        output_chunks = []
        input_tokens_used = 0
        output_tokens_used = 0

        for chunk in stream:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                output_chunks.append(delta.content)
            # Capture usage from final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens_used  = chunk.usage.prompt_tokens or est_input_tokens
                output_tokens_used = chunk.usage.completion_tokens or len(output_chunks)

        t_end = time.perf_counter()

        if first_token_t is None:
            first_token_t = t_end

        ttft_ms  = (first_token_t - t_start) * 1000
        total_ms = (t_end - t_start) * 1000

        if input_tokens_used == 0:
            input_tokens_used = est_input_tokens
        if output_tokens_used == 0:
            output_tokens_used = len(" ".join(output_chunks).split())

        cost = calc_cost(input_tokens_used, output_tokens_used)

        if run_idx == 1:
            cold_ttft_ms = ttft_ms
            speedup_str = "baseline"
        else:
            speedup = cold_ttft_ms / ttft_ms if ttft_ms > 0 else 0
            speedup_str = f"{speedup:.1f}x"

        results.append({
            "run": run_idx,
            "type": run_type,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "input_tokens": input_tokens_used,
            "output_tokens": output_tokens_used,
            "cost": cost,
        })

        print(f"  {run_idx:<6} {run_type:<8} {fmt_ms(ttft_ms):>10}  {fmt_ms(total_ms):>10}  "
              f"{input_tokens_used:>8,}  {output_tokens_used:>8,}  ${cost:>9.4f}  {speedup_str:>8}")

    except Exception as exc:
        print(f"  {run_idx:<6} {run_type:<8} ERROR: {exc}")
        results.append({"run": run_idx, "type": run_type, "error": str(exc)})

# ── Summary ───────────────────────────────────────────────────────────────────
print("  " + "─" * 70)

cold = next((r for r in results if r["type"] == "COLD" and "error" not in r), None)
warm_runs = [r for r in results if r["type"] == "WARM" and "error" not in r]

if cold and warm_runs:
    avg_warm_ttft = sum(r["ttft_ms"] for r in warm_runs) / len(warm_runs)
    speedup = cold["ttft_ms"] / avg_warm_ttft if avg_warm_ttft > 0 else 0
    cold_cost = cold["cost"]
    avg_warm_cost = sum(r["cost"] for r in warm_runs) / len(warm_runs)
    cost_savings_pct = (cold_cost - avg_warm_cost) / cold_cost * 100 if cold_cost > 0 else 0

    print(f"""
  RESULTS SUMMARY
  ──────────────────────────────────────────────────────────────
  Cold TTFT         : {fmt_ms(cold['ttft_ms'])}
  Warm TTFT (avg)   : {fmt_ms(avg_warm_ttft)}
  TTFT Speedup      : {speedup:.1f}x

  Cold cost/request : ${cold_cost:.4f}
  Warm cost/request : ${avg_warm_cost:.4f}
  Cost reduction    : {cost_savings_pct:.1f}%
""")

    if speedup < 5:
        print("  ⚠  Nebius has NO significant prefix caching on their API.")
        print("     Cold == Warm == full prefill every time.")
        print("     This is the BASELINE that AMF eliminates.\n")
    else:
        print(f"  ✓  Nebius has some prefix caching ({speedup:.1f}x speedup).")
        print("     AMF would add ROI-gating, adaptive control, and 95.3% savings on top.\n")

    # AMF projection
    amf_savings_pct = 0.953
    amf_warm_cost = cold_cost * (1 - amf_savings_pct)
    print(f"""  AMF PROJECTION (95.3% prefill savings at 128K)
  ──────────────────────────────────────────────────────────────
  Current cold cost/req   : ${cold_cost:.4f}
  After AMF warm cost/req : ${amf_warm_cost:.6f}
  AMF savings/request     : ${cold_cost - amf_warm_cost:.4f}  ({amf_savings_pct*100:.1f}% of bill)

  At 1M requests/day: ${(cold_cost - amf_warm_cost) * 1_000_000 / 1_000:.0f}K/day = ${(cold_cost - amf_warm_cost) * 1_000_000 * 365 / 1_000_000:.1f}M/year saved
  At 10M requests/day: ${(cold_cost - amf_warm_cost) * 10_000_000 * 365 / 1_000_000:.0f}M/year saved
""")

print("  Note: For full KV-cache AMF benchmark on H100 (Tuesday):")
print("    1. Rent nebius GPU instance: nvidia-h100-80gb")
print("    2. Deploy model with vLLM")
print("    3. Run: python3 scripts/h100_1m_benchmark.py")
print()
