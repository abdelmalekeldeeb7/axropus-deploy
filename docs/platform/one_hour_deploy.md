One-Hour Deployment

Important
- Run commands from repo root: `/home/korith/korith`, or use absolute paths like `bash /home/korith/korith/deploy/install.sh`.

1. Prepare config
- Edit `deploy/config/config.yaml`.
- Verify model path and backend adapter fields.
- Ensure `security.api_key_salt` is set (installer will auto-generate if placeholder remains).

2. Deploy
- `bash deploy/install.sh`
- Or from any directory: `bash /home/korith/korith/deploy/install.sh`
- Easy helper: `./scripts/korith_deploy_easy.sh up`

3. Validate services
- Router: `curl http://127.0.0.1:8000/health`
- Gateway: `curl http://127.0.0.1:8080/health`

4. Export API key
- `export KORITH_API_KEY="$(python3 -c 'import json;print(json.load(open("deploy/state/bootstrap_key.json"))["api_key"])')"`
- Verify auth:
  - `curl -H "Authorization: Bearer $KORITH_API_KEY" http://127.0.0.1:8080/v1/health`

5. Submit workload
- `python platform/korith_platform.py submit demo/jobs/ticket_triage.json --url http://127.0.0.1:8080 --api-key "$KORITH_API_KEY"`

6. Fetch telemetry
- `python platform/korith_platform.py metrics <job_id> --url http://127.0.0.1:8080 --api-key "$KORITH_API_KEY"`
- `python platform/korith_platform.py events <job_id> --url http://127.0.0.1:8080 --api-key "$KORITH_API_KEY"`

7. Generate savings report
- `python platform/korith_platform.py report --ledger ./platform_data/ledger.sqlite --out ./platform_data/savings_report.json --gpu_hourly_cost 2.5`

8. Roll back if needed
- `bash deploy/rollback.sh`
