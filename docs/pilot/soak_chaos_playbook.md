# Soak + Chaos Playbook

This playbook validates durability beyond scenario tests by running:

1. Sustained mixed workload load (multi-tenant + prompt entropy).
2. Chaos injections (restarts, kill-style failures, autoscale churn, optional disk pressure).
3. Evidence export (latency curves, hit-rate drift, chaos timeline, system pressure).

## Entry points

- Orchestrator: `scripts/soak_chaos.py`
- Wrapper: `scripts/run_soak_chaos.sh`

## Quick smoke run (5-15 min)

```bash
cd /home/korith/korith

export KORITH_SOAK_BACKEND_ID=korith_local
export KORITH_SOAK_MODEL_ID=llama-8b-local
export KORITH_SOAK_MODEL_PATH=/home/korith/korith/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

./scripts/run_soak_chaos.sh \
  --duration-s 900 \
  --rps 0.2 \
  --concurrency 1 \
  --n-ctx 4096 \
  --max-tokens 32 \
  --prompt-tokens-max 256 \
  --chaos-min-interval-s 60 \
  --chaos-max-interval-s 120
```

## 24h durability run

```bash
cd /home/korith/korith

export KORITH_SOAK_BACKEND_ID=korith_local
export KORITH_SOAK_MODEL_ID=llama-8b-local
export KORITH_SOAK_MODEL_PATH=/home/korith/korith/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

./scripts/run_soak_chaos.sh \
  --duration-s 86400 \
  --rps 0.25 \
  --concurrency 2 \
  --initial-workers 1 \
  --max-workers 2 \
  --chaos-min-interval-s 7200 \
  --chaos-max-interval-s 14400 \
  --report-bucket-s 900
```

## Optional heavy storage churn

Set `--disk-pressure-bytes` to force file-system pressure events:

```bash
./scripts/run_soak_chaos.sh --disk-pressure-bytes 2147483648
```

## Output artifacts

Each run writes into `platform_data/soak/<timestamp>/`:

- `run_config.json`: full run configuration.
- `data/results.jsonl`: per-request records (status + metrics rollups).
- `data/chaos_events.jsonl`: chaos timeline.
- `data/system_metrics.csv`: sampled memory/disk/GPU/process health.
- `report/summary.json`: machine-readable final report.
- `report/summary.md`: human-readable summary.
- `report/curves.csv`: p50/p95/p99 + hit-rate series over time.

## Pass criteria guidance

- No corruption events.
- Stable or improving hit-rate over time.
- p95/p99 remain bounded after chaos events.
- No unbounded disk growth.
- Request failures remain within expected chaos windows and recover after restarts.
