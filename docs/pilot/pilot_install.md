Pilot Install Guide

Prerequisites
- Ubuntu 22.04+ or compatible Linux host
- NVIDIA GPU with recent driver
- 16 GB RAM minimum
- 30 GB free disk for models, artifacts, and logs

Quick Deploy (under 1 hour)
1. Configure deployment values in `deploy/config/config.yaml`.
2. Place model files in `models/`.
3. Run:
   - `bash deploy/install.sh`
4. Verify services:
   - `http://127.0.0.1:8000/v1/health`
   - `http://127.0.0.1:8080/v1/health`
5. Submit first job:
   - `python platform/korith_platform.py submit demo/jobs/ticket_triage.json --url http://127.0.0.1:8080`
6. Inspect results:
   - `python platform/korith_platform.py metrics <job_id> --url http://127.0.0.1:8080`
   - `python platform/korith_platform.py events <job_id> --url http://127.0.0.1:8080`

Rollback
- `bash deploy/rollback.sh`

Notes
- For baseline-only backends, set `runtime.backend_id` and `runtime.model_endpoint` in `deploy/config/config.yaml`.
- For replay-enabled local backend, keep `runtime.backend_id: korith_local`.
