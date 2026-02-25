Execution Design (Step-by-Step for Codex Agent)

Phase 0: Preconditions
1) Confirm deterministic runtime config exists (model path, seed, n_ctx, n_batch).
2) Confirm AMF/MF/CP logging is enabled.
3) Confirm a stable prompt/workload input file exists.
4) Confirm backend capabilities:
   - Tier 1 (korith_local): KV replay available (AMF/MF full loop)
   - Tier 2 (openai_compatible): baseline-only (no KV replay), ledger still works

Phase 1: Gateway + Schema
1) Implement canonical job schema in code and validate input.
2) Build CLI commands: submit, status, logs, history, restore.
3) Add HTTP endpoints mirroring CLI.
4) Ensure each request writes a job record stub to the audit ledger.

Phase 2: Runtime Integration
1) Convert job input into Korith prompt.
2) Execute Korith with deterministic policy fields.
3) Capture:
   - AMF decisions
   - MF policy outputs
   - Run summary + health line
4) Populate job output + metrics.

Phase 3: Audit Ledger
1) Append immutable job record (inputs, outputs, metrics, runtime decisions).
2) Store log path or inline summary line.
3) Ensure records are immutable after write.

Phase 4: MF History Store
1) On each run, snapshot MF state with model_hash + prompt_hash.
2) Store MF snapshot in history store keyed by job_id.
3) Implement restore endpoint:
   - Validate model_hash + prompt_hash
   - Apply MF snapshot
   - Log [MF_RESTORE]

Phase 5: Validation
1) Cold run → AMF_MISS, snapshot stored, ledger appended.
2) Warm run → AMF_HIT, skip_ratio and ROI logged.
3) Restore test → MF_RESTORE logged, replay behavior stable.
4) Ensure audit ledger includes replay decisions + metrics.

Phase 6: Demo Readiness
1) Provide a deterministic workload example (ticket triage).
2) Run 3-shot demo (warm/warm/main) and verify summary line.
3) Export logs for investor/demo evidence.
