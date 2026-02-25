#!/usr/bin/env bash
set -euo pipefail

URL="${KORITH_URL:-http://127.0.0.1:8000}"
JOBSPEC="${1:-demo/jobs/ticket_triage.json}"
RUNS="${2:-5}"
TIMEOUT_S="${3:-60}"
DRAFT_TOKENS="${4:-6}"
API_KEY="${KORITH_API_KEY:-}"

if [[ -z "${API_KEY}" ]]; then
  echo "KORITH_API_KEY is required"
  exit 1
fi
if [[ ! -f "${JOBSPEC}" ]]; then
  echo "jobspec not found: ${JOBSPEC}"
  exit 1
fi

python3 - "$URL" "$JOBSPEC" "$RUNS" "$TIMEOUT_S" "$DRAFT_TOKENS" "$API_KEY" <<'PY'
import json
import subprocess
import sys
import time

url, jobspec_path, runs_s, timeout_s, draft_tokens_s, api_key = sys.argv[1:]
runs = int(runs_s)
timeout = float(timeout_s)
draft_tokens = int(draft_tokens_s)

jobspec = json.load(open(jobspec_path, "r", encoding="utf-8"))
jobspec.setdefault("policy", {})
jobspec["policy"]["allow_spec"] = True
jobspec["policy"]["allow_amf_reuse"] = bool(jobspec["policy"].get("allow_amf_reuse", True))
jobspec["spec_cfg"] = {
    "enabled": True,
    "draft_max_tokens": draft_tokens,
    "verify_batch": draft_tokens,
    "max_spec_steps": 32,
}

rows = []
for i in range(1, runs + 1):
    payload = "/tmp/korith_spec_jobspec.json"
    with open(payload, "w", encoding="utf-8") as f:
        json.dump(jobspec, f)
    job_id = subprocess.check_output(
        ["python3", "platform/korith_platform.py", "submit", payload, "--url", url, "--api-key", api_key],
        text=True,
    ).strip()
    deadline = time.time() + timeout
    status = "QUEUED"
    while time.time() < deadline:
        rec = json.loads(subprocess.check_output(
            ["python3", "platform/korith_platform.py", "status", job_id, "--url", url, "--api-key", api_key],
            text=True,
        ))
        status = rec.get("status", "")
        if status in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.25)
    metrics = json.loads(subprocess.check_output(
        ["python3", "platform/korith_platform.py", "metrics", job_id, "--url", url, "--api-key", api_key],
        text=True,
    ))
    amf = metrics.get("amf", {})
    spec = metrics.get("spec", {})
    perf = metrics.get("perf", {})
    rows.append({
        "job_id": job_id,
        "status": status,
        "lane": metrics.get("scheduling", {}).get("lane", ""),
        "amf": amf.get("decision", "unavailable"),
        "spec_enabled": bool(spec.get("enabled", False)),
        "acceptance_rate": float(spec.get("acceptance_rate", 0.0) or 0.0),
        "total_ms": float(perf.get("total_ms", 0.0) or 0.0),
        "avg_tps": float(perf.get("avg_tps", 0.0) or 0.0),
    })
    print(
        f"run={i} job_id={job_id} status={status} lane={rows[-1]['lane']} amf={rows[-1]['amf']} "
        f"spec={int(rows[-1]['spec_enabled'])} accept={rows[-1]['acceptance_rate']:.3f} "
        f"total_ms={rows[-1]['total_ms']:.2f} tps={rows[-1]['avg_tps']:.2f}",
        flush=True,
    )

ok = [r for r in rows if r["status"] == "SUCCEEDED"]
avg_total = sum(r["total_ms"] for r in ok) / len(ok) if ok else 0.0
avg_tps = sum(r["avg_tps"] for r in ok) / len(ok) if ok else 0.0
avg_accept = sum(r["acceptance_rate"] for r in ok) / len(ok) if ok else 0.0
enabled_runs = sum(1 for r in ok if r["spec_enabled"])
print(
    f"summary runs={runs} succeeded={len(ok)} spec_enabled_runs={enabled_runs} "
    f"avg_acceptance_rate={avg_accept:.3f} avg_total_ms={avg_total:.2f} avg_tps={avg_tps:.2f}"
)
PY

