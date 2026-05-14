# Axropus — Adaptive Memory Framework for LLM Inference

**Eliminating KV cache prefix recomputation at the architecture layer.**

On Llama 3.1 70B FP8 at 128K context on a single H200, Axropus AMF achieves **166× end-to-end speedup** over vLLM cold inference on a 500-request batched workload at 32-token output. With the FP8 codec fix applied at 256-token output: **136×**. vLLM's built-in prefix caching on the same 256-token workload: **1.1×**. Published state-of-the-art (VAST + LMCache) on comparable workloads: **7-8×**.

The gap between 1.1× and 136× isn't optimization. It's category. Existing systems cache or optimize prefill. Axropus eliminates it.

---

## Table of contents

1. [What AMF actually does](#what-amf-actually-does)
2. [Why this is architecturally different](#why-this-is-architecturally-different)
3. [Benchmark headlines](#benchmark-headlines)
4. [System architecture](#system-architecture)
5. [Codebase structure](#codebase-structure)
6. [The hot path, traced end to end](#the-hot-path-traced-end-to-end)
7. [Key engineering decisions](#key-engineering-decisions)
8. [Multi-node infrastructure](#multi-node-infrastructure)
9. [NVIDIA Dynamo integration](#nvidia-dynamo-integration)
10. [Correctness and invariants](#correctness-and-invariants)
11. [What's measured, what's projected, what's not done](#whats-measured-whats-projected-whats-not-done)
12. [Reproducibility](#reproducibility)
13. [Project scope](#project-scope)

---

## What AMF actually does

Most LLM inference workloads share long prefixes. A 128K-token system prompt that sits in front of every request. A retrieval-augmented context that's shared across 500 concurrent users. A codebase included in every call to an agent. Today, every request that hits a cold cache re-runs the full prefill over that shared prefix — on H200, for a 70B model at 128K, that's roughly 60 seconds of GPU compute. Per request.

Standard approaches tackle this by caching the KV blocks after prefill (vLLM prefix caching, SGLang RadixAttention) or offloading them to secondary storage (VAST + LMCache). Both still *recompute* or *reload* KV on every cold hit.

AMF materializes the KV state once, stores it in a compressed form, and restores it directly for subsequent requests. The prefill doesn't get faster — it doesn't happen. The hot path on a warm request is:

1. Restore compressed KV blocks from the in-memory VRAM pool (~374 ms, one-time per prefix per batch)
2. Register the restored blocks into vLLM's block table
3. Jump straight into decode

No prefill. No recomputation. No model forward pass over the shared prefix.

---

## Why this is architecturally different

Prefix caching and KV offload are incremental optimizations of the same underlying pattern: *"when a prefix repeats, reload its KV instead of recomputing."* They differ on storage location, compression, and routing.

AMF is a categorical change: *"a shared prefix's KV state is a materialized artifact, not a cache of compute."* The distinction shows up everywhere in the design:

- **Content-addressed indexing.** Prefixes are keyed by SHA-256 hash of token IDs + model hash + tenant + RoPE parameters + sampling config + RNG state. Ten dimensions of identity, not just "did these tokens appear before."
- **ROI-based admission.** A prefix only enters the pool if its expected compute-saved / restore-cost ratio exceeds a threshold. Low-value prefixes are dropped on arrival.
- **Deterministic replay enforcement.** Replay only runs under deterministic sampling; non-deterministic sampling falls through to cold prefill. Fails closed.
- **Value-per-byte eviction.** Not LRU. The pool ranks entries by `(hits × savings_ms) / bytes` and evicts the lowest-value entries first, so small high-value prefixes survive longer than large low-value ones.
- **Runtime-agnostic.** Operates at the KV cache layer, not vLLM internals. Same approach works for vLLM, Dynamo, SGLang, TensorRT-LLM. Verified on Llama 3.1 70B and Qwen 1.5B.

---

## Benchmark headlines

### Llama 3.1 70B FP8 at 128K, H200, 500-request batch

| Workload | Approach | Per-request wall-clock | Speedup |
|---|---|---:|---:|
| 32-token output | Cold baseline | ~60,000 ms | 1.0× |
| 32-token output | AMF (INT8 VRAM pool) | ~360 ms | **166×** |
| 256-token output | Cold baseline | ~60,000 ms | 1.0× |
| 256-token output | vLLM `--enable-prefix-caching` | ~54,000 ms | 1.1× |
| 256-token output | AMF (FP8 VRAM pool, scale fix applied) | ~441 ms | **136×** |

### Qwen 1.5B at 20K, 20-token output

| Approach | Speedup |
|---|---:|
| AMF | **229×** |

### Published state of the art, same workload class

| System | Speedup |
|---|---:|
| VAST + LMCache | 7-8× |
| vLLM prefix caching | 1.1× (measured on same 256-token workload) |

AMF is ~18-20× ahead of the published SOTA. The 1.1× vLLM result on the same benchmark is the strongest single data point: on *identical* hardware, *identical* prompt, *identical* harness, the gap between "cache prefill" and "eliminate prefill" is the difference between 1.1× and 136×.

---

## System architecture

AMF is structured in five layers, bottom-up:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Multi-node coordination                           │
│    Cluster router · Snapshot transfer · AMF coordinator    │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Runtime integrations                              │
│    vLLM V1 hook · Dynamo NIXL backend · AMF router plugin  │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Tiered cache & routing                            │
│    G1 AMF VRAM pool · G2 vLLM paged · G3 LMCache CPU/NVMe  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Codecs & kernels                                  │
│    FP8 (E4M3/E5M2) · INT4 · INT2 · NVFP4 · TurboQuant       │
│    FP8/INT4/NVFP4 decode attention kernels                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Core engine                                       │
│    Speculative decode · Replay gating · ROI · RNG state    │
└─────────────────────────────────────────────────────────────┘
```

Each layer has explicit contracts with the layers above and below. The boundary between Layer 3 and Layer 4 is the runtime-agnostic seam — Layer 3 knows nothing about vLLM internals, Layer 4 knows nothing about codec choices.

---

## Codebase structure

**~70,000 lines total**, across C++/CUDA (core engine + kernels), Python (integrations + platform), Rust (scheduler), and React (frontend).

```
axropus-deploy/
├── core/                         # C++ core engine (~16,750 LOC)
│   ├── engine.cpp                # Main inference loop with AMF replay gating (4,066 LOC)
│   ├── batch_executor.cpp        # Speculative decode + batch scheduling (3,290 LOC)
│   ├── amf_store.cpp             # AMF persistent store (lookup/save/evict)
│   ├── amf_direct_kv.cpp         # Direct-GPU KV restore path
│   ├── memory_field.cpp          # Control system for replay enable/disable
│   ├── collapse_controller.cpp   # Thermal/drift controller
│   ├── cuda_graph_decode.cu      # CUDA graph decode
│   ├── spec_v2.cpp               # Speculative decoding v2
│   └── kernels/
│       ├── accept_scan.cu        # SIMD-friendly first-mismatch scan (spec verify)
│       └── spec_verify.cu        # Fused speculative verification
│
├── korith_vllm_ext/              # vLLM integration (~5,700 LOC Python)
│   ├── amf_vllm_hook.py          # 4-seam integration (scheduler/worker/runner/blocks)
│   ├── amf_kv_manager.py         # Low-level vLLM KV save/restore with AMFK format
│   ├── amf_kv_connector.py       # vLLM connector plugin
│   ├── amf_scheduler_hook.py     # Scheduler intercept point
│   ├── compressed_vram_pool.py   # Slab-allocated VRAM pool (892 LOC)
│   ├── tiered_router.py          # G1/G3/cold routing with prefetch
│   ├── decode_scheduler.py       # Decode-phase scheduler (680 LOC)
│   ├── lmcache_adapter.py        # LMCache backend adapter
│   ├── codecs/
│   │   ├── amf_codec.py          # FP8 E4M3/E5M2 with scale sidecar (591 LOC)
│   │   ├── turboquant_codec.py   # PolarQuant + QJL reference (231 LOC)
│   │   └── nvfp4_codec.py        # NVFP4 codec
│   └── kernels/
│       ├── fp8_decode_attention.cu    # Online softmax, tile-rescaled
│       ├── int4_decode_attention.cu
│       ├── nvfp4_decode_attention.cu
│       └── bindings.cpp          # PyTorch extension bindings
│
├── controller/                   # Rust admission/QoS scheduler (~1,321 LOC)
│   ├── scheduler.rs              # Admission control
│   ├── scheduler_core.rs         # Core scheduling logic
│   ├── feedback.rs               # Feedback-loop tuning (542 LOC)
│   ├── thermodynamics.rs         # Thermal-aware scheduling
│   └── ffi.rs                    # C ABI for C++ engine interop
│
├── platform/                     # Platform services (~30K LOC Python)
│   ├── runtime/
│   │   ├── cluster_router.py     # Multi-node KV-aware router (1,904 LOC)
│   │   ├── cluster_worker.py     # Worker node implementation (3,366 LOC)
│   │   ├── amf_coordinator.py    # Multi-node coordinator service
│   │   ├── amf_coordinator_client.py
│   │   ├── dynamo_nixl_backend.py    # NVIDIA Dynamo storage backend (632 LOC)
│   │   ├── dynamo_amf_router.py      # Dynamo router plugin (334 LOC)
│   │   ├── dynamo_event_subscriber.py
│   │   ├── decode_cache_store.py
│   │   └── restore_store.py
│   ├── reasoning/                # RL-driven adaptive controller (~8K LOC)
│   │   ├── reasoning_model.py    # Llama 3.1 8B NF4 RL controller
│   │   ├── learning_loop.py      # Hourly LoRA fine-tuning
│   │   ├── federated_loop.py     # Federated learning (FedAvg, McMahan 2017)
│   │   ├── adaptive_engine.py
│   │   ├── reward_calculator.py
│   │   ├── rollout_buffer.py
│   │   └── nemoclaw_model.py
│   ├── gateway/                  # HTTP/gRPC gateway
│   ├── ledger/                   # Tamper-evident audit log
│   ├── queue/                    # NATS / Redis Streams integration
│   ├── economics/                # ROI tracking, billing
│   └── observability/            # Prometheus + Grafana
│
├── axropus/                      # CLI + FastAPI server
│   ├── server.py                 # Main server entry
│   ├── cli.py                    # Command-line interface
│   └── metrics.py                # Prometheus metrics
│
├── frontend/                     # React dashboard
│   ├── src/App.jsx
│   └── src/components/           # Sidebar, Terminal, StatusBadge, etc.
│
├── benchmarks/                   # Benchmark harnesses
│   ├── multi_request_benchmark.py       # Simulation harness for unit tests
│   └── axropus_vs_lmcache_h200.md       # H200 benchmark methodology doc
│
├── deploy/
│   ├── helm/                     # Kubernetes Helm charts
│   ├── docker/                   # Dockerfiles (including Dockerfile.dynamo-amf)
│   ├── observability/            # Grafana dashboards
│   └── profiles/                 # Deployment profiles
│
├── holographic/                  # Physics-inspired control experiment
│
└── docs/
    ├── platform/                 # Architecture design docs
    └── pilot/                    # Customer pilot playbooks
```

**Languages:**
- C++: 13,288 LOC (engine + kernels wrapper)
- CUDA: 1,990 LOC (decode attention, spec verify, accept scan)
- Python: 54,082 LOC (integrations, platform, reasoning, frontend data)
- Rust: 1,321 LOC (scheduler, FFI, thermodynamics)
- Headers: 1,472 LOC
- React/JSX: frontend components

---

## The hot path, traced end to end

A user sends a 128K-token prompt with a unique suffix. Here's what happens.

### Request arrival

```
user request
    → axropus.server (FastAPI)
        → korith_vllm_ext.amf_vllm_hook.AMFvLLMHook.on_request_arrival(seq_id, token_ids)
```

The hook computes a SHA-256 prefix hash over the tokens, truncates to 64 bits (16 hex chars), and queries the tiered router.

### Tiered lookup

```
korith_vllm_ext.tiered_router.TieredCacheRouter.lookup(prefix_hash, token_ids)
    ├─ G1: compressed_vram_pool.CompressedVRAMPool.get(prefix_hash)
    │    └─ Hit → return PoolEntry with slab block references
    ├─ G3: lmcache_adapter.LMCacheAdapter.lookup(prefix_hash)
    │    └─ Hit → schedule async G3→G1 promotion, return KV tensor
    └─ Miss → return COLD with miss reason
```

Miss reasons are tracked per-category: `FIRST_REQUEST`, `EVICTED_FROM_AMF`, `LMCACHE_MISS`, `LMCACHE_ERROR`, `FORMAT_MISMATCH`, `PREFIX_TOO_SHORT`. Operators diagnose hit-rate regressions from these counters.

### G1 hit → skip prefill

```
hook returns HookAction.SKIP_PREFILL_TO_DECODE
    → hook.inject_blocks(seq_id, vllm_block_table)
        → pool.map_into_vllm(prefix_hash, seq_id, block_table)
            ↳ Maps slab block pointers into vLLM's block table as is_external=True,
              so vLLM's allocator doesn't try to free them.
    → hook.reapply_fp8_scales(model, seq_id)
        ↳ Serialized FP8 scales re-applied to attention modules;
          calculate_kv_scales forced False to prevent scale drift.
    → hook.should_skip_prefill_tensors(seq_id) → True
        ↳ Model runner skips prefill tensor prep.
    → Decode starts directly on the restored cache.
```

### Cold path → populate cache

On a cold request, vLLM runs the normal prefill. The hook's `on_prefill_complete(seq_id, kv, saved_ms)` callback fires at the end:

```
hook.on_prefill_complete
    → router.store_after_prefill(prefix_hash, token_ids, kv, savings_ms)
        ├─ pool.put_from_raw(prefix_hash, kv, format=default_format)
        │    ├─ Codec compresses layer-by-layer (FP8 / INT4 / TurboQuant)
        │    ├─ Allocate slab blocks across layers (evict if needed)
        │    ├─ Write compressed blobs to VRAM slabs
        │    └─ Register PoolEntry with hit count, savings EMA
        └─ Mirror to LMCache (if write policy allows)
```

### Core engine replay gating (C++)

When AMF is used via the core C++ engine (not the vLLM path), the replay decision goes through explicit invariants:

```cpp
// core/engine.cpp — invariants enforced on every request
// - AMF replay only allowed under deterministic sampling.
// - ROI must only be computed when baseline is stable.
// - Sustained negative ROI disables replay.
// - Replay remains disabled until MF cooldown expires.
// - AMF fails closed on corruption or uncertainty.
// - MF outputs are bounded/monotonic and must not oscillate replay enable/disable.
// - MF never overrides AMF hard disables.
// - Baseline EMAs must converge before ROI use; ROI suppressed if baseline unstable.
```

The engine logs an `AMF_KEY` event containing ten identity signals (model_hash, tenant_hash, prompt_hash, n_ctx, n_batch, rope_base, rope_scale, sampling_hash, rng_hash). A cache hit only counts if all ten match.

---

## Key engineering decisions

### The FP8 scale-drift fix

vLLM's default behavior on FP8 KV cache is to compute per-tensor scales from the running batch statistics. When `calculate_kv_scales=True` (the default), the scale is derived from the absmax of the first forward pass.

This silently breaks restore. You save an FP8 KV blob quantized against scale `S1`. You restore it in a later session. vLLM recomputes the scale against the new batch and gets `S2 ≠ S1`. The stored bytes dequantize wrong. Subsequent decode produces garbage.

The fix is two-part:

1. **Serialize scales.** `FP8ScaleSidecar` dataclass carries `k_scale`, `v_scale`, `q_scale`, `prob_scale` alongside every compressed blob.
2. **Re-apply on restore.** `apply_fp8_scales()` walks the model's modules, detects attention by attribute presence (not class name — works across vLLM 0.6.x, 0.7.x, FlashAttention, XFormers, FlashInfer forks), writes the stored scales back via `old_val.data.fill_()` (handles `nn.Parameter` with `requires_grad=True`), and forces `calculate_kv_scales=False` for warm sequences.

Attribute-based detection means this works without hardcoding vLLM internals. The code tries both `_k_scale` and `k_scale` and `_kv_scale`, handles both Parameter and float attributes, and fails gracefully on exceptions.

### The compressed VRAM pool

Single contiguous VRAM allocation per layer, pre-sized at startup. Sub-allocated in fixed-size slab blocks (default 128 KB). Entries keyed by prefix hash, each owning one slab block index per layer.

Eviction is reuse-score-weighted, not LRU:

```python
score = α × age_seconds + β / hit_count + γ × size_mb − δ × avg_savings_ms
```

High-value prefixes (many hits, high savings) survive even when they're not the most recently used. Large low-value prefixes get dropped first. Default weights: α=0.3, β=0.4, γ=0.1, δ=0.2.

Pool auto-detects already-compressed dtypes (FP8, INT8, uint8) and stores raw instead of double-quantizing to INT4 — which would destroy quality.

### Codec plumbing

Six format IDs implemented:

- **FP8 E4M3** — per-tensor scale, 1 byte/element (Hopper, Blackwell)
- **FP8 E5M2** — wider exponent, per-tensor scale (long-tail distributions)
- **INT4 sym per-channel** — one scale per head (Ampere+)
- **INT4 sym per-block** — one scale per 128 tokens (higher accuracy)
- **INT2 sym per-channel** — aggressive compression for cold prefixes
- **TurboQuant** — PolarQuant + QJL, ~3 bits/element (Zandieh & Mirrokni, ICLR 2026)

TurboQuant is implemented from the paper: polar decomposition separates radius (fp16) from unit direction, direction is projected via a QJL Rademacher sign matrix into `k = 3 × head_dim` dimensions, and each projected component is 1-bit sign-encoded. Deterministic seeds per (head_dim, k) mean compressor and decompressor agree without shipping the projection matrix. Pseudo-inverse cached lazily per seed.

### FP8 decode attention kernel

253 lines of CUDA. One CTA per (batch, head) pair. Online softmax with tile-level accumulator rescaling:

```cuda
float rescale = expf_fast(old_max_smem - smem_state.m);
#pragma unroll
for (int d = 0; d < HeadDim; ++d) {
    out_acc[d] *= rescale;
}
```

The rescale is critical. Without it, the accumulator from earlier tiles carries weights computed under the old softmax max, producing wrong attention output for any sequence longer than one tile (128 tokens). Online softmax is easy to get subtly wrong — this implementation has been validated against FP16 reference.

The kernel is intentionally scalar and bandwidth-bound. The comment block says so explicitly:

```cpp
// NOT tensor-core-accelerated — this is a bandwidth-reduction kernel that
// wins via 2x fewer HBM bytes read, not via MMA throughput.
// TODO: WGMMA + TMA rewrite for full Hopper utilisation (requires H200).
```

Current throughput is dominated by the architectural win (prefill elimination), not the kernel. WGMMA rewrite is the obvious next step and is expected to add 2-3× on top of current numbers.

---

## Multi-node infrastructure

AMF wasn't built for a single node. The multi-node layer exists because commercial inference workloads don't run on one GPU — they run across fleets.

### AMF Coordinator service

HTTP + SQLite-backed registry that tracks which nodes have which prefixes cached. Workers register on startup, report their AMF pool state, and heartbeat. Routers query the coordinator to find which nodes can serve a given prefix hash.

The coordinator's contract:

```
POST /register           — worker node registration
POST /snapshot           — report a new prefix stored locally
GET  /lookup/{hash}      — return all nodes caching this prefix
POST /heartbeat          — liveness + load signal
DELETE /node/{id}        — deregister on shutdown
```

### KV-cache-aware cluster router (1,904 LOC)

`platform/runtime/cluster_router.py`. Routes requests across nodes using a cost model:

- **Local hit.** Prefix already on this node → serve immediately.
- **Remote hit.** Prefix on another node → compute `transfer_cost_ms` vs `recompute_cost_ms`.
    - If transfer < recompute: fetch KV from remote node via snapshot transfer, serve locally.
    - Else: prefill locally.
- **No hit anywhere.** Route to least-loaded node with capacity.

Transfer cost estimated from measured RTT + bandwidth (calibrated from 817-run averages in benchmark data). Recompute cost estimated from quadratic prefill model (`3.9e-7 × t² + 0.17 × t` ms for H200, fit to calibration points at 120K, 252K, 1M tokens).

### Snapshot transfer

When a remote node has a prefix the local node needs, the snapshot transfer path moves the compressed KV blob over NIXL or TCP. The transfer is at the compressed layer — 2.5 GB for a 128K 70B prefix at INT4, not 10 GB at FP16.

### Federated learning loop

`platform/reasoning/federated_loop.py` implements Federated Averaging (McMahan et al., 2017). Each node trains a local LoRA adapter on its own request patterns (PPO/DPO rewards based on accept rates and quality metrics). The coordinator aggregates deltas across nodes periodically and broadcasts the averaged adapter. Nodes converge on a shared routing/admission policy while keeping training data local.

---

## NVIDIA Dynamo integration

Dynamo is NVIDIA's disaggregated inference stack — prefill workers and decode workers on separate GPUs, connected by NIXL (NVIDIA Inference Xfer Library) for KV tensor movement.

AMF integrates as a **persistent storage backend** for Dynamo's KV Block Manager, plus an **AMF-aware router plugin** that extends Dynamo's native KV overlap routing with persistent snapshot availability.

### NIXL storage backend (`platform/runtime/dynamo_nixl_backend.py`, 632 LOC)

Implements the NIXL block-level storage interface: `register_volume`, `put`, `get`, `delete`, `stats`.

Dynamo sends individual KV blocks (16-256 tokens each). AMF's natural unit is a whole-prefix snapshot. `BlockToSnapshotAggregator` buffers incoming blocks for `agg_window_ms` (default 500 ms) and flushes them as a single prefix snapshot. Flushed snapshots go through:

1. **AdmissionGate** — ROI check. `prefill_ms / restore_ms` must exceed `min_admit_roi` (default 1.5). Rejected prefixes are dropped.
2. **CorrectnessValidator** — Metadata check on restore. `model_hash`, `kv_version` must match between stored and requested. Rope params, sampling hash checked as well. Mismatch → return `None` → Dynamo falls back to recompute.
3. **Value-per-byte eviction** — When storage exceeds `eviction_watermark × max_storage_bytes` (default 85% of 100 GiB), entries are ranked by `(prefill_ms × (1 + hit_count)) / size_bytes` and the lowest-value entries evicted first.

### AMF-aware router plugin (`platform/runtime/dynamo_amf_router.py`, 334 LOC)

Extends Dynamo's KV overlap routing with a third signal:

```
score = (kv_overlap × kv_weight)
      + (amf_restore_savings_pct × restore_weight)
      - (load × load_weight)
```

Default weights: `kv_weight = 0.5`, `restore_weight = 0.8`, `load_weight = 0.3`.

For short context (< 4096 tokens) the plugin defers entirely to Dynamo's native scoring — AMF adds no meaningful value for short prompts. For long context, AMF snapshot availability can dominate because Dynamo's in-memory blocks are likely already evicted.

`canonicalize_prompt_text` normalizes prompts before hashing to maximize hit rate across minor formatting variations (trailing whitespace, different newline handling).

---

## Correctness and invariants

### Engine-level invariants (C++)

Stated formally at the top of `core/engine.cpp`:

```
- AMF replay only allowed under deterministic sampling.
- ROI must only be computed when baseline is stable.
- Sustained negative ROI disables replay.
- Replay remains disabled until MF cooldown expires.
- AMF fails closed on corruption or uncertainty.
- MF outputs are bounded/monotonic and must not oscillate replay enable/disable.
- MF never overrides AMF hard disables.
- Baseline EMAs must converge before ROI use; ROI suppressed if baseline unstable.
```

These aren't comments — they're enforced by the code. The lookup path emits `AMF_REPLAY_BLOCKED` events with reasons: `memory_field_cooldown`, `negative_roi`, `non_deterministic`, `env_restore_disable`, `low_roi`. Every block is tagged and telemetry-logged.

### Ten-dimensional cache key

Cache hits require agreement on:

1. `model_hash` — SHA-256 of model weights
2. `tenant_hash` — tenant isolation
3. `prompt_hash` — FNV-1a over token IDs
4. `n_ctx` — context window size
5. `n_batch` — batch size
6. `rope_base_bits` — RoPE base (bitcast from float)
7. `rope_scale_bits` — RoPE scale
8. `sampling_hash` — sampling config hash
9. `rng_hash` — RNG state hash
10. `kv_version` — storage format version

All ten logged on every lookup as the `AMF_KEY` event. Any mismatch → miss with explicit reason.

### RNG state restoration

On warm restore, `amf_restore_rng_state()` runs after KV restore and before decode. Deterministic sampling requires the RNG state to match what was present when the prefill originally ran. Most systems miss this; it's the detail that makes replay actually reproducible rather than just "similar."

### Tamper-evident audit log

`platform/ledger/` implements a hash-chained audit log. Every AMF decision (hit, miss, block, evict, insert) is appended with a hash that chains to the previous entry. Tampering with historical entries would require rewriting the whole chain. Useful for compliance audits and multi-tenant accountability.

### Validation testing

1,000-run AMF soak test at 128K context achieved 99.9% hit rate. ROI-based quality gating catches regressions in production.

---

## What's measured, what's projected, what's not done

Honest picture. This distinction matters.

### Measured on real H200

- 166× speedup on Llama 3.1 70B FP8 at 128K, 32-token output, 500-request batch (INT4 VRAM pool, BF16 KV path pre-FP8-fix)
- 136× speedup on same workload at 256-token output with FP8 codec fix applied
- 1.1× for vLLM `--enable-prefix-caching` on identical 256-token workload
- 229× on Qwen 1.5B at 20K context, 20-token output
- 99.9% hit rate over 1,000-run soak test at 128K
- P50 350 ms, P99 372 ms at 32-token output; P50/P99 spread under 10% (indicates warm path is decode-bound, not I/O-bound)

### Projected but not re-benchmarked

- **32-token run with FP8 fix applied.** Current 166× headline used BF16 KV on the 32-token run because the FP8 codec bug was still present. The FP8 fix has been applied to the 256-token run (126× → 136×). Re-running the 32-token benchmark with FP8 is expected to push the number above 166×, likely toward ~200-300×.
- **WGMMA + TMA kernel rewrite.** FP8 decode kernel is currently scalar/bandwidth-bound. Tensor-core utilization via WGMMA is the obvious next step and is expected to add 2-3× on top of current numbers.
- **Asymptotic convergence.** Math confirms 166× is the asymptote as N → ∞ for the 32-token workload; N=500 already at 125×, N=5000 at 162×, N=10000 at 164×. Predicted to hold at N=2000 and N=5000 on re-benchmark.

### Not done yet

- **Llama 3.1 405B.** Same methodology expected to produce similar ratio; absolute savings ~6× larger. Pending H200 time.
- **Multi-tenant mix at realistic traffic.** 100 tenants × 5 requests each to measure prefix cache hit rate under real traffic shape.
- **TurboQuant fused CUDA kernel.** Python reference is working; fused CUDA version is the next kernel to write.
- **Large-scale Dynamo deployment test.** The NIXL backend and router plugin are implemented and unit-tested. Full Dynamo multi-node deployment benchmark is pending.

---

## Reproducibility

```bash
# 1. Load a 128K-token prompt into /tmp/long_prompt.txt

# 2. Run with AMF enabled
KORITH_ENABLE_AMF=1 \
KORITH_AMF_PATH=/tmp/amf \
KORITH_VRAM_POOL_GB=50 \
KORITH_VRAM_POOL_QUANT=int4 \
python3 -m <benchmark_module> \
    --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --num-requests 500 \
    --prefix-file /tmp/long_prompt.txt \
    --max-tokens 32      # or 256 for FP8-path run
```

Three critical output lines are emitted to stdout:

```
[RESULTS] Cold TTFT:       60XXX ms
[RESULTS] Warm P50:        350.X ms
[RESULTS] TTFT speedup:    166.X x
```

Benchmark methodology lives in `benchmarks/axropus_vs_lmcache_h200.md`. Full methodology doc (`500req_128k_h200.md`) with honest caveats, economic breakdown, convergence math, and caveats on measured-vs-projected distinctions available on request.

---

## Project scope

- **Core engine:** C++/CUDA inference engine built on llama.cpp with custom kernels, AMF replay gating, speculative decode (Spec V2, ~49% decode speedup), CUDA graph decode, engine pool, thermal controller.
- **vLLM integration:** Full V1 engine extension via 4-seam integration (scheduler, worker, model runner, block manager) with AMFK binary format, chunked D→H copy, direct NVMe write via ctypes, FP8 scale sidecar, stacked/split KV layout handling.
- **Runtime/scheduling:** Rust admission control and per-request QoS scheduler with feedback loop, thermodynamic awareness, FFI bindings into C++ engine.
- **Multi-node:** Coordinator service, KV-cache-aware cluster router, cross-node snapshot transfer, bandwidth/RTT-based routing with economic transfer-vs-recompute logic.
- **NVIDIA Dynamo:** NIXL storage backend with admission gate and correctness validator, AMF-aware router plugin with weighted scoring across kv_overlap + amf_savings + load.
- **Reasoning/learning:** Llama 3.1 8B NF4 RL controller for adaptive policy decisions, hourly LoRA fine-tuning, federated learning loop with FedAvg (McMahan 2017).
- **Platform:** FastAPI control plane, React dashboard (model hub, playground, claws, billing, auth), Postgres + SQLAlchemy, JWT/bcrypt, tenant metering.
- **Deployment:** Helm charts for Kubernetes, Docker Compose stacks (including Dynamo profile), Grafana dashboards, Prometheus metrics, chaos testing.

Built solo over four months. ~70,000 lines. Pre-seed stage. Wilson Sonsini handling Delaware C-Corp incorporation.

---

## Contact

**Malek Eldeeb**
Abdelmalekeldeeb7@gmail.                                                                     
github.com/Abdelmalekeldeeb7
Riverside, CA · Open to relocation (NYC / SF / Stockholm
