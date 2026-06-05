# Axropus AMF

Axropus AMF is a persistent KV materialization layer for vLLM. It captures
prefilled KV state, keeps it in a controlled AMF tier, then restores and
registers it back into vLLM so repeated long-context requests can skip prefill.

The project is focused on agentic and long-context inference workloads where the
same expensive context appears repeatedly:

- system prompts
- tool definitions
- repository or document context
- RAG blocks
- workflow state
- long agent traces

AMF is not a replacement for vLLM Automatic Prefix Caching or LMCache. It is a
hot recovery/materialization layer that can sit beside them.

## Core Idea

vLLM APC reuses prefix KV while the matching blocks are still alive inside the
engine. That is fast, but fragile under eviction, restarts, multi-tenant
rotation, and routing to a worker that never saw the prefix.

AMF separates reusable KV state from vLLM's live block lifetime:

1. A cold request runs normal vLLM prefill.
2. AMF snapshots the real vLLM KV blocks for the prefix.
3. Later, if APC has the prefix, AMF stays out of the way.
4. If APC misses but AMF has the materialized prefix, AMF restores KV into
   vLLM's live KV cache.
5. AMF registers the restored blocks as valid prefix cache state.
6. vLLM decodes without recomputing the prefix.

In short:

```text
cold prefill once -> materialize KV -> restore/register -> skip future prefill
```

## Stack Position

AMF is designed as a tier in a larger KV cache stack:

```text
G1: AMF hot GPU tier              raw or compressed VRAM materialization
G2: vLLM live paged KV / APC      normal in-engine prefix cache
G3: LMCache CPU tier              optional fallback
G4: LMCache NVMe / disk tier      optional fallback
G5: LMCache remote tier           optional fallback
```

The intended lookup flow is:

```text
1. vLLM APC live hit      -> use APC
2. AMF hot hit           -> restore/register into vLLM
3. LMCache hit           -> retrieve, promote into AMF, then future hits are hot
4. miss everywhere       -> cold prefill
```

## Implemented Components

Important files:

- `korith_vllm_ext/amf_worker_ext.py`  
  vLLM worker extension exposing `amf_save_kv` and `amf_restore_kv` through
  `llm.collective_rpc(...)`.

- `korith_vllm_ext/amf_kv_manager.py`  
  Direct save/restore of vLLM KV blocks, AMF keying, VRAM snapshot cache,
  disk persistence, and restore into physical KV blocks.

- `korith_vllm_ext/amf_scheduler_hook.py`  
  Scheduler-level prefill skip hook for AMF-aware vLLM paths.

- `korith_vllm_ext/lmcache_adapter.py`  
  Optional import-guarded LMCache adapter. AMF runs without LMCache installed.

- `korith_vllm_ext/tiered_router.py`  
  AMF + LMCache tier router. LMCache misses fall back cold; LMCache hits can be
  promoted into AMF.

- `scripts/production_realistic_apc_vs_amf.py`  
  Re-runnable synthetic benchmark comparing APC, LMCache, AMF, and AMF+LMCache
  under steady state and cache disruption events.

## Benchmark: 1K Reliability Run

These are saved local results under `benchmark_results/*_1k/`.

This was a synthetic 1,000-request stream, not a 1,000-concurrent throughput
run. The harness calls `llm.generate([prompt])` one request at a time. It is
intended to measure cache coverage, disruption recovery, latency, and GPU-second
proxy cost.

Configuration:

```text
Model: Qwen2.5-1.5B-Instruct
Requests: 1,000
Max model length: 24,576 tokens
Output tokens: 1
Suffix tokens: 48
Reusable prefixes: 16
Unique one-off prefixes: 20%
Reuse distribution: Zipfian, alpha=1.15
Hot prefixes: top 4
Synthetic working set: ~4,566 MB KV
```

Prefix length mix:

```text
Short:  20%, 2K-4K tokens
Medium: 60%, 8K-12K tokens
Long:   20%, 16K-20K tokens
```

Memory shapes:

```text
APC:     4 GB vLLM KV
LMCache: 1 GB vLLM KV + 4 GB local CPU LMCache
AMF:     1 GB vLLM KV + 3 GB FP8 compressed AMF GPU tier
```

Disruption model:

```text
Restart/reset:      every 100 requests, 10-request recovery window
Eviction-to-zero:   every 80 requests, 8-request recovery window
Cold routing:       simulated 8-worker fleet, 1-request recovery window
```

Results:

| Regime | Arm | Full hit rate | p50 latency | p95 latency | GPU-sec proxy | Restore p95 |
|---|---:|---:|---:|---:|---:|---:|
| steady_state | APC | 31.8% | 672.1 ms | 2406.8 ms | 497.71 | 0.0 ms |
| steady_state | LMCache | 33.2% | 63.1 ms | 2021.2 ms | 292.02 | 0.0 ms |
| steady_state | AMF | 77.9% | 47.5 ms | 1183.7 ms | 185.57 | 13.0 ms |
| post_restart | APC | 25.6% | 787.9 ms | 2430.2 ms | 69.61 | 0.0 ms |
| post_restart | LMCache | 36.6% | 64.0 ms | 2023.0 ms | 37.31 | 0.0 ms |
| post_restart | AMF | 79.3% | 47.4 ms | 1202.9 ms | 23.40 | 13.3 ms |
| post_eviction | APC | 32.8% | 276.0 ms | 2241.0 ms | 39.63 | 0.0 ms |
| post_eviction | LMCache | 34.3% | 59.0 ms | 1997.6 ms | 26.52 | 0.0 ms |
| post_eviction | AMF | 76.1% | 47.2 ms | 1145.5 ms | 17.72 | 12.8 ms |
| post_cold_routing | APC | 0.0% | 852.6 ms | 2437.9 ms | 119.98 | 0.0 ms |
| post_cold_routing | LMCache | 24.4% | 64.9 ms | 1684.4 ms | 50.40 | 0.0 ms |
| post_cold_routing | AMF | 77.9% | 47.7 ms | 1160.6 ms | 33.31 | 12.9 ms |
| blended | APC | 27.2% | 773.6 ms | 2420.4 ms | 726.93 | 0.0 ms |
| blended | LMCache | 32.4% | 64.0 ms | 2018.2 ms | 406.26 | 0.0 ms |
| blended | AMF | 77.9% | 47.5 ms | 1183.7 ms | 260.00 | 13.0 ms |

Blended GPU-second proxy:

```text
APC:     726.93
LMCache: 406.26
AMF:     260.00
```

On this run AMF used:

```text
2.80x fewer GPU-seconds than APC
1.56x fewer GPU-seconds than LMCache
```

Correctness gate:

```text
AMF sampled correctness checks: 25
Mismatches: 0
Correctness pass: true
```

All data above is synthetic and should be treated as benchmark evidence, not a
claim about every production workload.

## Why AMF Wins In Disruptions

APC is tied to live vLLM block state. If a reset, eviction flood, or cold worker
routing event removes those blocks, APC must prefill again.

AMF keeps a separate materialized KV state. In the disruption benchmark, the
vLLM/APC state is reset, while AMF's materialized tier is preserved. AMF then
restores the KV blocks and registers them back into vLLM. That is why AMF can
retain high full-hit coverage after reset-like events.

Important limitation: raw VRAM AMF state survives vLLM cache reset inside a live
process. If the whole process dies, raw VRAM state dies too. For process-level
survival, use AMF disk/NVMe persistence or an LMCache lower tier.

## AMF And LMCache

AMF and LMCache are complementary:

- LMCache is strong as a lower-tier KV storage and movement layer across CPU,
  NVMe, and remote storage.
- AMF is the hot GPU materialization and vLLM rehydration layer.
- CacheBlend-style work increases what text patterns are reusable.
- AMF focuses on making known KV state executable again inside vLLM quickly.

The combined stack is:

```text
CacheBlend / LMCache expands reusable KV coverage
LMCache stores and moves KV across workers or lower tiers
AMF restores/registers hot KV into vLLM for fast prefill elimination
```

## Reproducing Local 1K Results

APC:

```bash
python3 scripts/production_realistic_apc_vs_amf.py \
  --config configs/apc_reliability_4gb_1k.yaml \
  --only-arm apc
```

LMCache:

```bash
python3 scripts/production_realistic_apc_vs_amf.py \
  --config configs/lmcache_reliability_1plus4cpu_1k.yaml \
  --only-arm lmcache
```

AMF:

```bash
export PYTHONPATH="$PWD"

python3 scripts/production_realistic_apc_vs_amf.py \
  --config configs/amf_reliability_fp8_1k.yaml \
  --only-arm amf
```

Outputs are written to:

```text
benchmark_results/apc_reliability_4gb_1k/
benchmark_results/lmcache_reliability_1plus4cpu_1k/
benchmark_results/amf_reliability_fp8_1k/
```

## Running AMF As A vLLM Stack

For serving, use the packaged stack config and launcher:

```bash
cp configs/amf_stack.env.example configs/amf_stack.env
./scripts/verify_amf_stack.sh configs/amf_stack.env
./scripts/start_vllm_amf_stack.sh configs/amf_stack.env
```

The quickstart is in `docs/amf_stack_quickstart.md`.

The launcher starts an OpenAI-compatible vLLM server with:

- vLLM prefix caching enabled
- AMF worker extension enabled
- Korith scheduler enabled
- AMF hot VRAM tier configured from `configs/amf_stack.env`
- optional LMCache transfer config when `AXROPUS_ENABLE_LMCACHE=1`

Before claiming AMF+LMCache on a new machine, run:

```bash
RUN_AMF_LMCACHE_SMOKE=1 ./scripts/verify_amf_stack.sh configs/amf_stack.env
```

## H200 Larger-Model Path

The repo also includes H200-oriented configs and a runner:

```text
configs/h200_qwen36_27b_amf_smoke_64req.yaml
configs/h200_deepseek_v4_flash_smoke_32req.yaml
configs/h200_deepseek_v4_flash_amf_stack_256req.yaml
scripts/run_h200_deepseek_amf_benchmark.sh
```

Example:

```bash
cd ~/amf
source .venv/bin/activate
export PYTHONPATH="$PWD"

./scripts/run_h200_deepseek_amf_benchmark.sh qwen27 all
```

Run individual arms if model reload time is high:

```bash
./scripts/run_h200_deepseek_amf_benchmark.sh qwen27 apc
./scripts/run_h200_deepseek_amf_benchmark.sh qwen27 lmcache
./scripts/run_h200_deepseek_amf_benchmark.sh qwen27 amf
./scripts/run_h200_deepseek_amf_benchmark.sh qwen27 amf_lmcache
```

## Key Environment Variables

AMF:

```bash
export KORITH_ENABLE_AMF=1
export KORITH_AMF_PATH=/tmp/korith-amf
export KORITH_AMF_VRAM_FIRST=1
export KORITH_AMF_SYNC_SAVE=1
export KORITH_VRAM_CACHE_GB=0
export KORITH_VRAM_CACHE_DEVICE=cuda:0
export KORITH_VRAM_POOL_GB=3
export KORITH_VRAM_POOL_QUANT=fp8
```

LMCache:

```bash
export LMCACHE_LOCAL_CPU=true
export LMCACHE_MAX_LOCAL_CPU_SIZE=4
export LMCACHE_CHUNK_SIZE=256
```

AMF + LMCache fallback:

```bash
export AXROPUS_LMCACHE_FALLBACK=true
export AXROPUS_PROMOTE_LMCACHE_HITS=true
export AXROPUS_LMCACHE_WRITE_THROUGH=true
```

## Status

This repository is active research/engineering code. The AMF benchmark path is
working locally and on H200-oriented configs, but the code should still be
treated as experimental until tested under your exact model, vLLM version,
traffic pattern, and failure model.

The benchmark harness intentionally labels synthetic traffic as synthetic and
includes correctness checks for AMF restored outputs. Fast-but-wrong restore is
treated as a failed benchmark.
