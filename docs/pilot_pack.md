Korith Pilot Pack (Inference Scheduling Middleware)

Purpose
- Run a paid pilot with an inference/platform team to prove measurable savings.
- Korith acts as governed middleware: scheduler signals + auditable compute reuse.

Target Buyer
- AI inference / platform teams operating GPU fleets.

What Korith Delivers (Pilot Scope)
1) Governed execution
- Deterministic by default
- Fail-closed on uncertainty/corruption

2) Scheduling signals (per request)
- amf_decision: hit|miss|blocked
- prefix_len / skipped_tokens / skip_ratio
- restore_ms / saved_ms / roi
- replay_disable_mask / cooldown_ms
- worker_id / gpu_id / queue_latency_ms

3) Audit artifacts
- metrics.json (versioned contract)
- events.jsonl (append-only event stream)
- request logs and output artifacts

4) Benchmark + proof
- cold/warm/warm benchmark harness
- ranges-based pass/fail checks
- summary report that ties exactly to the ledger

Deployment Model
- On-prem or customer cloud VPC
- Router + worker-per-GPU (single-node first)
- Storage: local FS + SQLite (pilot), optional S3/Postgres

Integration Options
A) Drop-in Inference Endpoint
- POST /v1/generate (streaming optional)

B) Middleware in Front of Existing Stack
- Routes to worker pools, emits scheduling telemetry, maintains ledger

Model/Runtime Support (Truthful Matrix)
Tier 1 (native compute skipping via AMF KV replay)
- llama.cpp / GGUF path (current Korith integration)

Tier 2 (platform support without KV replay)
- Any runtime reachable via API/adapter
- Provides: scheduling signals, ledger, audit, cost reporting
- Does NOT provide: KV-prefix replay unless backend exposes safe serialize/restore hooks

Tier 3 (planned deep KV integration)
- vLLM, TensorRT-LLM, TGI (requires backend hooks for KV serialize/restore)

Pilot Requirements
- Workloads with meaningful repetition (templated prompts, RAG templates, agent loops)
- Pinned deterministic settings for replay-enabled paths
- Access to run representative traffic (sample set is fine)

Success Criteria (Define Before Pilot Starts)
- Hit-rate: stable and measurable on target workload subset
- Skip ratio: high for repeated prompts
- ROI: >= 1 on replay-enabled paths after baseline stabilizes
- Fleet impact: measurable latency/cost reduction on target traffic slice
- Trust: every decision auditable via metrics.json + events.jsonl

Failure Semantics (What Happens When Things Go Wrong)
- Corruption or mismatch -> replay blocked, baseline path continues
- Negative ROI streak -> replay disabled per policy until cooldown
- MF oscillation -> replay forced disabled with extended cooldown
- All events logged and recorded in ledger

Deliverables to Customer
- Deployment bundle (Docker/Helm optional)
- Benchmark results (cold/warm/warm)
- Savings report (ledger-backed)
- Runbook: rollback + health checks

Commercial Terms (Typical)
- Paid pilot: $25k-$150k depending on integration scope and fleet size
- Conversion options:
  - Annual license + support
  - Per-GPU pricing
  - Share-of-savings (only with ledger-backed reporting)
