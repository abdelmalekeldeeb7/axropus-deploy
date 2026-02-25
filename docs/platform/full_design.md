Korith Full Platform Design (Inference Scheduling Middleware)

1) Product Definition
Korith is a governed inference scheduling middleware that reduces GPU cost by:
- skipping redundant prefill compute via AMF KV-prefix replay
- exposing first-class scheduling signals (hit/miss/skip/ROI/policy)
- enforcing deterministic governance via MF/CP
- producing an immutable audit ledger and reproducible policy state (MF snapshot/restore)

Primary buyer: AI inference/platform teams operating GPU fleets.

2) Core Architecture (Single Node -> Cluster)

A) Router (front door)
Responsibilities:
- accepts requests (HTTP/gRPC; CLI optional)
- normalizes prompt and config
- computes request fingerprint (model_hash + tokenizer_hash + prompt_hash + sampling_hash + n_ctx + n_batch)
- assigns to a worker with session affinity
- backpressure + queue discipline

Key outputs per request:
- routing decision (gpu_id/worker_id/lane)
- queue latency metrics

B) Workers (one per GPU)
Responsibilities:
- runs the Korith runtime (engine)
- executes:
  1) AMF lookup / restore decision
  2) MF update + policy apply (clamped outputs)
  3) CP gating (baseline always legal; spec gated later)
  4) decode stream (stream tokens)
- writes artifacts: logs, output, metrics.json, events.jsonl
- emits health line

Worker lanes:
- HIT lane (low latency): prioritize AMF hits, smaller batches, lower tail
- MISS lane (throughput): batch prefill aggressively, maximize utilization

C) Shared Stores
1) Ledger store (immutable history)
- SQLite for MVP; Postgres for cluster
- stores request spec, hashes, status, pointers to artifacts
- stores scheduling signals and cost/savings summaries

2) Artifact store
- local filesystem for MVP
- S3/MinIO for cluster
- stores: logs, outputs, metrics.json, events.jsonl, MF snapshots

3) MF snapshot store (policy restore)
- table or artifact blob keyed by request/job_id and fingerprint
- restore is explicit and fail-closed

4) Queue (optional but recommended)
- in-proc for MVP
- NATS / Redis Streams / SQS for scale

3) Interfaces

A) Inference API (primary)
- POST /v1/generate (streaming supported)
  - inputs: model_id/model_path, prompt, deterministic config, policy flags
  - outputs: streamed tokens + final metrics payload

- GET /v1/health
  - router + worker health summaries

- GET /v1/metrics (operator)
  - prometheus scrape or JSON summary

B) Operator / Audit API
- GET /v1/requests/{id}
- GET /v1/requests/{id}/logs
- GET /v1/requests/{id}/events
- GET /v1/requests/{id}/snapshot
- POST /v1/requests/{id}/restore (privileged)

C) CLI (secondary)
- korith submit
- korith logs
- korith show
- korith restore
- korith benchmark

4) Canonical Contracts (Do not break casually)

A) Request Fingerprint
Fields that define determinism and reuse:
- model_hash (weights + arch + tokenizer)
- tokenizer_hash
- prompt_hash (normalized prompt template + content)
- sampling_hash (temperature/top_p/top_k/greedy/benchmark gates/seed)
- n_ctx, n_batch
- runtime build/version hash (optional but recommended)

B) RunMetrics.v1 (single source of truth)
Emitted as metrics.json per request/job and returned by API.

Required:
- ids: request_id, parent_id (optional), timestamp
- model: model_id, model_hash, tokenizer_hash
- input: prompt_hash, input_tokens, prompt_tokens
- scheduling: lane, worker_id, gpu_id, queue_latency_ms
- AMF:
  - decision: hit|miss|blocked
  - key fields: prefix_len, skipped_tokens, skip_ratio
  - restore_ms, baseline_ms (if stable), saved_ms, roi
- MF:
  - min_admit_roi, eviction_pressure, replay_disable_mask, cooldown_ms
  - mf_snapshot_id
- performance:
  - tokens_out, avg_tps, p50/p95 token latency (optional)
- health:
  - KORITH_HEALTH line fields
- errors:
  - error_code, error_summary

C) Events stream (events.jsonl)
Append-only, ordered. Examples:
- AMF_HIT / AMF_MISS / AMF_BLOCK
- MF_APPLY / MF_RESTORE
- CP_DECISION
- ROUTER_ASSIGN
- WORKER_START / WORKER_END
- KORITH_HEALTH

5) AMF Integration (platform-native)
AMF responsibilities:
- restore KV prefix when deterministic and key matches
- compute saved_ms and roi based on stable baseline measurement
- admission and eviction governed by EMA ROI
- fail-closed on corruption/uncertainty

Platform responsibilities:
- expose AMF decision + roi + skip_ratio as scheduler signals
- ensure prompt normalization and sampling config are stable
- metrics.json is truth (logs are audit)

6) MF Integration (platform-native)
MF responsibilities:
- take runtime signals (hits/misses/pressure/roi trend)
- output bounded knobs (min_admit_roi, eviction_pressure, replay mask, cooldown)
- avoid oscillation and enforce cooldown invariants

Platform responsibilities:
- store MF outputs in metrics.json and ledger
- snapshot MF after policy evaluation
- restore MF snapshot explicitly and fail-closed if fingerprint mismatch
- MF never overrides AMF hard disables

7) Scheduling Strategy (why you win)
Scheduler consumes the signals you generate:
- if AMF decision likely hit (or proven hit history): route to HIT lane
- if miss: batch in MISS lane
- if MF disables replay: route to MISS lane until cooldown expires
- if roi_slope collapses: reduce admissions or shift traffic

8) Scaling to Larger Models
Changes:
- KV state is larger; restore IO becomes more expensive
- decode becomes more memory-bandwidth bound
- batching becomes more important on misses
- session affinity becomes more valuable

Requirements:
- one worker per GPU for predictability
- fast restore path (NVMe, mmap, pinned buffers)
- strong fingerprinting to prevent false hits
- artifacts offload to S3/MinIO for multi-node

8.1) "Support All Models" (Truthful Definition)
Korith can be "available for all models" in two different senses:

A) Platform availability (universal)
- Any model/runtime can be routed, audited, and scheduled by Korith via a backend adapter.
- The ledger, artifacts, and determinism policy are still valuable even without KV replay.

B) Native compute skipping (backend-dependent)
- AMF KV-prefix replay requires the backend to expose safe KV serialize/restore hooks.
- Until a backend exposes these hooks, Korith must treat it as "baseline-only" and report AMF as unavailable.

This is why model support must be capability-based and tiered:
- Tier 1: native KV replay (AMF + MF full loop)
- Tier 2: baseline-only (gateway + ledger + scheduling telemetry, but no KV replay)
- Tier 3: planned deep KV integration (vLLM / TRT-LLM / TGI once hooks are implemented)

8.2) Backend Capability Negotiation
Each worker must report backend capabilities so the router/scheduler can make correct, safe decisions:
- supports_kv_serialize: bool
- supports_kv_restore: bool
- supports_deterministic_seed: bool
- supports_streaming: bool
- supports_logits_access: bool
- supports_spec_decode: bool (future)

AMF/MF are activated only when capabilities are sufficient; otherwise fail closed to baseline.

8.3) Portable Backend Interface (minimal)
Formalize an internal adapter contract so Korith can target multiple runtimes without rewriting governance:

InferenceBackend:
- load_model()
- tokenize(prompt)
- run(prompt, max_tokens, sampling_cfg) -> output_text + optional telemetry
- get_fingerprint() -> model_hash + tokenizer_hash + backend_version
- get_capabilities() -> capability set above

If KV hooks exist:
- serialize_kv(prefix_len) -> blob_id
- restore_kv(blob_id) -> restore_ms

9) Speculative Decode (Phase after baseline stability)
Goal: reduce decode cost after prefill.

Design:
- draft model proposes up to k tokens
- verifier model checks proposals; commits only verified tokens
- bounded rollback and strict determinism controls
- speculation enabled only when baseline stable and MF/AMF signals indicate benefit

Metrics:
- spec_enabled, proposed_tokens, accepted_tokens, acceptance_rate
- spec_cost_ms, spec_saved_ms, spec_roi
- auto-disable events on sustained underperformance

10) Cloud / Deployment
Profiles:
- local: SQLite + FS + in-proc queue
- staging: Postgres + FS/S3 + Redis/NATS
- cloud: Postgres + S3 + NATS/SQS + per-GPU workers

Packaging:
- Docker images: router, worker
- Helm chart: router + worker DaemonSet (one per GPU node) + services

Security:
- API keys
- privileged endpoints allowlist (restore, raw logs)

11) Benchmark & Proof Layer (sales-critical)
Benchmark suite:
- cold/warm/warm
- prompt length scaling
- hit-rate under templated workloads
- decode-heavy vs prefill-heavy workloads
- soak test 10k+ tokens mixed prompts

Outputs:
- metrics.json per run
- one-line summary per run:
  [KORITH_HEALTH] hit_rate=... avg_skip_ratio=... avg_roi_ema=... roi_slope=...

12) Build Order (execution)
1) Freeze RunMetrics.v1 + events.jsonl
2) Router + worker-per-GPU (single-node)
3) Immutable ledger + artifact store (SQLite + FS)
4) MF snapshot restore apply path (real, fail-closed)
5) Benchmark suite + golden ranges + soak harness

13) Savings Target Analyzer (execution gate)
Use real ledger artifacts to compute:
- current blended savings %
- prefill/decode shares
- decode cut required to reach target savings bands

CLI:
- `python3 platform/korith_platform.py savings-target --ledger platform_data/ledger.sqlite --targets 0.5,0.6,0.7 --out platform_data/savings_target_report.json`

Output:
- `summary.current_savings_pct`
- `summary.prefill_share_pct`, `summary.decode_share_pct`
- per-target:
  - `required_decode_cut_pct`
  - `additional_decode_cut_pct`
  - `feasible`

14) Phase 6 Decode + Kernel Design (explicit)
Objective:
- close the remaining savings gap by reducing decode cost, while preserving AMF/MF correctness.

14.1) Decode architecture
- keep AMF/MF prefix path unchanged.
- add decode execution modes:
  - baseline decode
  - accelerated decode (kernel path)
  - speculative decode (draft + verify)
- select mode at runtime from capabilities and policy:
  - `SPEC_HIT` -> replay restore + spec decode
  - `HIT` -> replay restore + non-spec decode
  - `SPEC_MISS` -> normal prefill + spec decode
  - `MISS` -> normal prefill + non-spec decode

14.2) Kernel abstraction
- define backend-agnostic kernel contract:
  - prefill()
  - decode_step()
  - apply_kv_replay()
  - verify_tokens()
- runtime toggles:
  - `KORITH_KERNELS=0|1`
  - `KORITH_KERNEL_BACKEND=none|cuda|triton`
  - `KORITH_KERNEL_VERIFY=0|1`
- fallback rule:
  - if kernel verify fails, emit `KERNEL_FALLBACK` and continue baseline.

14.3) CUDA priority order
1) KV restore/apply path
2) decode-step attention + KV append
3) prefill attention path
4) logits/sampling path

14.4) Speculative decode design
- draft model proposes `k` tokens.
- verify model validates proposals and accepts prefix.
- rejection path commits verify token and continues.
- hard gates:
  - enable only when adapter supports `verify_tokens` or logits verify.
  - disable on sustained low acceptance.
- controls:
  - `KORITH_SPEC_ENABLED=0|1`
  - `KORITH_SPEC_K`
  - `KORITH_SPEC_MIN_ACCEPT`
  - `KORITH_SPEC_DISABLE_AFTER_N`

14.5) Spec governance persistence
- persist state in `spec_governance`:
  - `spec_disabled`
  - `reason`
  - `cooldown_until`
  - `bad_accept_streak`
- events:
  - `SPEC_ENABLE`
  - `SPEC_DISABLE`
  - `SPEC_STEP`
  - `SPEC_SUMMARY`

14.6) Metrics additions (authoritative)
- extend `metrics.json` only additively:
  - `kernels`: `enabled`, `backend`, `verify_ok`, `fallback`, `ms_saved`
  - `spec`: `supported`, `enabled`, `k`, `proposed_tokens`, `accepted_tokens`, `acceptance_rate`, `draft_ms`, `verify_ms`, `saved_ms`, `roi`, `speedup_est`
- keep existing AMF/MF/perf fields unchanged.

15) Full Stack Execution Plan (what we will do)
Phase A: decode observability lock
1) enforce additive `spec`/`kernels` metrics in every run.
2) add replay+decode blended savings report gates.
3) block rollout if metrics are missing or invalid.

Phase B: decode performance rollout
1) enable spec in shadow mode (`enabled=false`, measurements on).
2) enable spec on low-risk workloads.
3) tune acceptance thresholds per workload class.
4) enable kernel path with verify fallback.

Phase C: cluster compounding
1) enable replay-aware multi-node locality routing.
2) enable transfer-vs-recompute policy.
3) keep lane priorities (`SPEC_HIT > HIT > SPEC_MISS > MISS`).

Phase D: enterprise hardening
1) private/BYOA deployment defaults.
2) no-retention defaults and org isolation.
3) compliance gating per backend/model.
4) pilot proof pack and ROI export.

16) KPI gates for target savings
To target `50-70%` blended savings, enforce:
- AMF hit rate on repeatable traffic: `>= 0.70`
- AMF skip ratio on hits: `>= 0.60`
- spec enabled runs (supported workloads): `>= 0.70`
- spec acceptance rate: `>= 0.65`
- decode reduction (weighted): `>= 0.35` for 50% target, `>= 0.55` for 70% target
- queue+transfer overhead share: `<= 0.10`

17) Current status vs target
- prefix replay path: implemented and validated on local runs.
- decode path: partially implemented, not yet tuned for sustained high acceptance/speedup.
- cluster locality and transfer economics: implemented baseline, requires soak and tuning.
- security/compliance enterprise packaging: in progress.

18) Command set for execution tracking
- savings gap report:
  - `python3 platform/korith_platform.py savings-target --ledger platform_data/ledger.sqlite --targets 0.5,0.6,0.7 --out platform_data/savings_target_report.json`
- spec benchmark:
  - `python3 platform/korith_platform.py bench spec --jobspec demo/jobs/ticket_triage.json --runs 10 --url http://127.0.0.1:8000 --api-key \"$KORITH_API_KEY\"`
- kernel status:
  - `python3 platform/korith_platform.py kernels status --url http://127.0.0.1:8000 --api-key \"$KORITH_API_KEY\"`
- spec governance status:
  - `python3 platform/korith_platform.py spec status <fingerprint_hash> --url http://127.0.0.1:8000 --api-key \"$KORITH_API_KEY\"`
