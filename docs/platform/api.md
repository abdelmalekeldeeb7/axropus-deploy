Korith HTTP API (Workload Gateway)

POST /jobs
- Body: job_schema.json
- Response: {"job_id": "..."}

GET /jobs/{job_id}
- Response: full job record with output + metrics + history

GET /jobs/{job_id}/logs
- Response: audit log lines for the job

GET /jobs/{job_id}/history
- Response: MF snapshot metadata for the job

POST /jobs/{job_id}/restore
- Applies MF snapshot for the job
- Response: {"restored": true}

Notes
- Deterministic policy must be required for AMF replay.
- Replay decisions and ROI must be logged and returned.
- Model support is capability-based:
  - Tier 1 backends expose KV serialize/restore -> AMF/MF replay enabled.
  - Tier 2 backends do not expose KV hooks -> baseline-only execution, audit still recorded.
