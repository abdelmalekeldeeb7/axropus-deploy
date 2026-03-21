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

---

Deploying AMF with NVIDIA Dynamo
=================================

Prerequisites
-------------
- Dynamo 1.0+ workers configured with NATS KVBM event emission enabled
- NATS server (included in `docker-compose.dynamo.yml`)
- vLLM backend workers (Dynamo manages these)
- Docker Compose v2

Quick Start
-----------
```bash
# Start base Korith stack + Dynamo integration services:
docker compose \
  -f deploy/docker-compose.yml \
  -f deploy/docker-compose.dynamo.yml \
  up -d
```

This starts three additional services:
- `nats` — NATS 2.x server on port 4222 (event plane)
- `amf-coordinator` — AMF prefix index on port 8500
- `amf-event-subscriber` — subscribes to Dynamo KVBM events, triggers AMF persist

Environment Variables Reference
--------------------------------

| Variable | Default | Description |
|---|---|---|
| `DYNAMO_NATS_URL` | `nats://nats:4222` | NATS server URL |
| `DYNAMO_NATS_SUBJECT_PREFIX` | `dynamo.kv.block` | KVBM event subject prefix |
| `DYNAMO_WORKER` | `false` | Set `true` inside Dynamo worker containers |
| `AMF_COORDINATOR_URL` | `http://amf-coordinator:8500` | AMF coordinator HTTP endpoint |
| `AMF_SNAPSHOT_DIR` | `/data/amf/snapshots` | Local NVMe path for KV snapshots |
| `AMF_NODE_ID` | `amf-node-0` | Node identifier |
| `AMF_MIN_PREFIX_TOKENS` | `256` | Ignore prefixes shorter than this |
| `AMF_MIN_SAVED_MS_TO_PERSIST` | `500` | ROI gate: skip if estimated prefill < threshold |
| `AMF_NIXL_MIN_ADMIT_ROI` | `1.5` | NIXL admission ROI threshold |
| `AMF_NIXL_AGG_WINDOW_MS` | `500` | Block aggregation window (ms) |
| `AMF_NIXL_MAX_STORAGE_BYTES` | `107374182400` | Max snapshot storage (100 GiB) |
| `AMF_ROUTER_ENABLED` | `true` | Enable AMF-aware routing |
| `AMF_RESTORE_WEIGHT` | `0.8` | Weight of AMF restore savings in routing score |
| `AMF_KV_OVERLAP_WEIGHT` | `0.5` | Weight of Dynamo KV overlap in routing score |
| `AMF_LOAD_WEIGHT` | `0.3` | Weight of worker load in routing score |
| `AMF_MIN_PREFIX_TOKENS_FOR_AMF` | `4096` | Below this, defer entirely to Dynamo routing |

Verifying the Integration
--------------------------
```bash
# 1. Check AMF coordinator health and prefix index stats:
curl http://localhost:8500/health

# 2. Check NATS subscription count (should show 4 subscriptions):
curl http://localhost:8222/subsz

# 3. Tail the subscriber logs to confirm events are flowing:
docker logs -f korith-amf-event-subscriber

# 4. After workload, check coordinator metrics:
curl http://localhost:8500/metrics
```

Benchmark: AMF On vs Off
-------------------------
```bash
# With AMF disabled (pure Dynamo):
AMF_ROUTER_ENABLED=false AMF_MIN_SAVED_MS_TO_PERSIST=999999999 \
  docker compose -f deploy/docker-compose.dynamo.yml up -d

# With AMF enabled:
AMF_ROUTER_ENABLED=true \
  docker compose -f deploy/docker-compose.dynamo.yml up -d

# Compare time-to-first-token for 120K+ context requests.
# Expected speedup: ~21x at 120K tokens (11,446 ms restore vs 243,610 ms recompute).
```
