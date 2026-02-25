Korith CLI (Workload Gateway)

Commands
- korith submit <job.json>
  - Submits a workload job and returns job_id.

- korith status <job_id>
  - Returns summary fields: status, output, metrics, history.

- korith logs <job_id>
  - Returns full audit log for the job.

- korith history <job_id>
  - Returns MF snapshot metadata for the job.

- korith restore <job_id>
  - Applies MF snapshot for the job and logs MF_RESTORE.

Notes
- CLI should emit deterministic config and AMF/MF metrics.
- CLI must not alter runtime policies beyond the job policy.
- Model support is backend + capability based; the same CLI submits jobs to either local Korith (Tier 1) or OpenAI-compatible runtimes (Tier 2).
