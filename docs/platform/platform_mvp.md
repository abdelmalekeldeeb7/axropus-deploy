Korith Platform MVP (Workload Ownership + Audit Trail)

Goal
- Provide a governed AI workload engine that exposes deterministic replay savings (AMF) and policy history (MF).
- No UI required; CLI + HTTP gateway only.

Primary Workload
- Ticket triage: classify priority, summarize, tag, and suggest assignee.

Core Promise
- Every job produces: output + replay decision + ROI/skip metrics + policy snapshot.
- Workloads are owned, auditable, and restorable.

Architecture (Functional)
1) Workload Gateway
   - CLI and HTTP submit jobs.
   - Normalizes to a single job schema.

2) Deterministic Runtime
   - Korith engine executes with AMF/MF/CP.
   - Determinism enforced by policy.

3) History Ledger
   - Append-only job history with metrics and artifacts.
   - MF snapshot stored per job for restore.

4) Governance Layer
   - Replay is allowed only under deterministic settings.
   - ROI computed only when baseline stable.

Deliverables
- Job schema (docs/platform/job_schema.json)
- CLI contract (docs/platform/cli.md)
- HTTP API contract (docs/platform/api.md)
- MF history/restore contract (docs/platform/mf_history.md)

Success Criteria
- Job produces deterministic output and audit trail.
- On Tier 1 backends (KV replay available): AMF hit shows skip_ratio and ROI.
- On Tier 2 backends (no KV hooks): platform still produces an audit trail; AMF is reported unavailable and replay is baseline-only.
- MF snapshot restore is deterministic and logged.
