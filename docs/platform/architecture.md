Korith Platform Architecture (Components + Data Flow)

Components
1) Workload Gateway
   - Interfaces: CLI and HTTP
   - Responsibility: accept jobs, validate schema, enqueue for execution

2) Job Router
   - Responsibility: normalize inputs into canonical job schema

3) Deterministic Runtime
   - Korith engine with AMF/MF/CP
   - Responsibility: deterministic execution, replay decisions, ROI metrics

4) Audit Ledger
   - Append-only job history store (JSONL or SQLite)
   - Responsibility: immutable record of input, decisions, metrics, outputs

5) MF History Store
   - Snapshot store for MF policy state per job
   - Responsibility: restore policy snapshots deterministically

6) Metrics Exporter
   - Emits run summary and health line

Data Flow
1) Client submits job → Workload Gateway
2) Gateway validates → Job Router
3) Router emits canonical job → Runtime
4) Runtime executes → outputs + AMF/MF decisions + metrics
5) Audit Ledger appends → job history
6) MF History Store records snapshot
7) Gateway exposes status/logs/history
8) Metrics Exporter emits health line
