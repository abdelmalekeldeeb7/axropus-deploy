Korith Deployment

Single-Node Startup
1. Edit `deploy/config/config.yaml`.
2. Run `bash deploy/install.sh`.
3. Submit a job:
   - `python platform/korith_platform.py submit demo/jobs/ticket_triage.json --url http://127.0.0.1:8080`

Core Endpoints
- Router health: `http://127.0.0.1:8000/v1/health`
- Gateway health: `http://127.0.0.1:8080/v1/health`
- Router metrics: `http://127.0.0.1:8000/metrics`
- Worker metrics: `http://127.0.0.1:9000/metrics`

Compose Services
- `router`
- `worker-gpu0`
- `gateway`

Observability
- Prometheus config: `deploy/observability/prometheus.yml`
- Grafana dashboard: `deploy/observability/grafana/korith-dashboard.json`

Rollback
- `bash deploy/rollback.sh`
