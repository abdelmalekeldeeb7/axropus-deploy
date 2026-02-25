Audit Ledger (Job History)

Storage
- Append-only JSONL or SQLite table
- One record per job (or per event if needed)
- Immutable after write

Minimum Fields
- job_id
- created_at
- owner_id
- workload_type
- input (hashes allowed if redaction needed)
- policy (deterministic config)
- runtime (AMF/MF decisions)
- metrics (skip_ratio, roi, avg_tps)
- output (summary fields)
- audit (summary line + log path)

Invariants
- Records are chronological and immutable.
- Replay decisions must be included.
- ROI must only be present when baseline is stable.
