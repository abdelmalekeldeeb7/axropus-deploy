Platform Quickstart (CLI)

Fast deploy from repo root:
- `./scripts/korith_deploy_easy.sh up`
- `export KORITH_API_KEY="$(./scripts/korith_deploy_easy.sh key)"`
- `./scripts/korith_deploy_easy.sh smoke`
- `./scripts/korith_deploy_easy.sh submit-demo`

1) Submit a job
   `./platform/korith_platform.py submit demo/jobs/ticket_triage.json --url http://127.0.0.1:8080 --api-key "$KORITH_API_KEY"`

2) Check status
   `./platform/korith_platform.py status <job_id> --url http://127.0.0.1:8080 --api-key "$KORITH_API_KEY"`

3) View logs
   `./platform/korith_platform.py logs <job_id> --url http://127.0.0.1:8080 --api-key "$KORITH_API_KEY"`

4) View MF snapshot
   `./platform/korith_platform.py history --limit 20 --url http://127.0.0.1:8080 --api-key "$KORITH_API_KEY"`

5) Restore snapshot (writes restore request)
   `./platform/korith_platform.py restore <job_id> --url http://127.0.0.1:8080 --api-key "$KORITH_API_KEY"`
   - This writes a restore request file under platform_data/restore_requests.

Notes
- Tier 1 (AMF/MF native): use runtime.backend=korith_local and a local GGUF model_path.
- Tier 2 (baseline-only): set runtime.backend=openai_compatible and runtime.endpoint to your runtime's OpenAI-compatible URL.
