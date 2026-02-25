#!/usr/bin/env bash
set -euo pipefail

URL="${KORITH_URL:-http://127.0.0.1:8000}"
JOBSPEC="${1:-demo/jobs/ticket_triage.json}"
RUNS="${2:-3}"
TIMEOUT_S="${3:-30}"
API_KEY="${KORITH_API_KEY:-}"

if [[ -z "${API_KEY}" ]]; then
  echo "KORITH_API_KEY is required"
  exit 1
fi

if [[ ! -f "${JOBSPEC}" ]]; then
  echo "jobspec not found: ${JOBSPEC}"
  exit 1
fi

python3 - "$URL" "$JOBSPEC" "$RUNS" "$TIMEOUT_S" "$API_KEY" <<'PY'
import json
import subprocess
import sys
import time

url, jobspec, runs_s, timeout_s, api_key = sys.argv[1:]
runs = int(runs_s)
timeout = float(timeout_s)

rows = []

for i in range(1, runs + 1):
    job_id = subprocess.check_output(
        [
            "python3",
            "platform/korith_platform.py",
            "submit",
            jobspec,
            "--url",
            url,
            "--api-key",
            api_key,
        ],
        text=True,
    ).strip()

    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = json.loads(
            subprocess.check_output(
                [
                    "python3",
                    "platform/korith_platform.py",
                    "status",
                    job_id,
                    "--url",
                    url,
                    "--api-key",
                    api_key,
                ],
                text=True,
            )
        )
        if status.get("status") in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.4)

    if status.get("status") != "SUCCEEDED":
        print(f"run={i} job_id={job_id} status={status.get('status')}", flush=True)
        rows.append((i, job_id, "failed", 0.0, 0.0, 0.0))
        continue

    metrics = json.loads(
        subprocess.check_output(
            [
                "python3",
                "platform/korith_platform.py",
                "metrics",
                job_id,
                "--url",
                url,
                "--api-key",
                api_key,
            ],
            text=True,
        )
    )
    amf = metrics.get("amf", {})
    perf = metrics.get("perf", {})
    decision = str(amf.get("decision", "unknown"))
    skip_ratio = float(amf.get("skip_ratio", 0.0) or 0.0)
    prefill_ms = float(perf.get("prefill_ms", 0.0) or 0.0)
    total_ms = float(perf.get("total_ms", 0.0) or 0.0)
    restore_ms = float(amf.get("restore_ms", 0.0) or 0.0)
    rows.append((i, job_id, decision, skip_ratio, prefill_ms, total_ms))
    print(
        f"run={i} job_id={job_id} amf={decision} skip_ratio={skip_ratio:.3f} "
        f"prefill_ms={prefill_ms:.2f} restore_ms={restore_ms:.2f} total_ms={total_ms:.2f}",
        flush=True,
    )

hits = sum(1 for _, _, d, *_ in rows if d == "hit")
ok = [r for r in rows if r[2] != "failed"]
avg_total = (sum(r[5] for r in ok) / len(ok)) if ok else 0.0
print(f"summary runs={runs} succeeded={len(ok)} hits={hits} hit_rate={(hits/len(ok) if ok else 0.0):.3f} avg_total_ms={avg_total:.2f}")
PY

