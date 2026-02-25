CLI + HTTP Gateway Spec

CLI
- korith submit <job.json>
  - Returns job_id

- korith status <job_id>
  - Returns output + metrics + runtime decisions

- korith logs <job_id>
  - Returns full audit log

- korith history <job_id>
  - Returns MF snapshot metadata

- korith restore <job_id>
  - Applies MF snapshot and logs MF_RESTORE

HTTP
- POST /jobs
  - Body: job_schema.json
  - Response: {"job_id": "..."}
  - runtime.backend determines execution path:
    - korith_local: runs local Korith binary (Tier 1)
    - openai_compatible: calls an OpenAI-compatible endpoint (Tier 2)

- GET /jobs/{job_id}
  - Response: full job record

- GET /jobs/{job_id}/logs
  - Response: audit log lines

- GET /jobs/{job_id}/history
  - Response: MF snapshot metadata

- POST /jobs/{job_id}/restore
  - Response: {"restored": true}
