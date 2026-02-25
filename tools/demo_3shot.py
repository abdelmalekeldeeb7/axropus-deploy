#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from platform.runtime.service import build_runtime
from platform.ledger import db as ledger


DEFAULT_JOB = Path("demo/jobs/ticket_triage.json")


def wait_done(db_path: Path, job_id: str, timeout_s: float = 600.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rec = ledger.get_job(db_path, job_id)
        if rec and rec["status"] in ("SUCCEEDED", "FAILED"):
            return rec
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} timed out")


def main():
    db_path = Path(os.environ.get("KORITH_PLATFORM_DB", "./platform_data/ledger.sqlite")).resolve()
    artifacts = Path(os.environ.get("KORITH_PLATFORM_ARTIFACTS", "./platform_data/artifacts")).resolve()
    gpus = os.environ.get("KORITH_PLATFORM_GPUS", "0")
    gpu_ids = [int(x.strip()) for x in gpus.split(",") if x.strip()]

    if not DEFAULT_JOB.exists():
        raise SystemExit("demo/jobs/ticket_triage.json not found")
    jobspec = json.loads(DEFAULT_JOB.read_text(encoding="utf-8"))

    router = build_runtime(db_path, artifacts, gpu_ids)

    job_ids = []
    for _ in range(3):
        job_id = router.submit(jobspec)
        wait_done(db_path, job_id)
        job_ids.append(job_id)

    hits = 0
    skip_sum = 0.0
    roi_sum = 0.0
    roi_count = 0
    for job_id in job_ids:
        run = ledger.get_latest_run(db_path, job_id)
        metrics = json.loads(Path(run["metrics_path"]).read_text(encoding="utf-8"))
        if metrics["amf"]["decision"] == "hit":
            hits += 1
            skip_sum += metrics["amf"]["skip_ratio"]
            roi_sum += metrics["amf"]["roi"]
            roi_count += 1

    hit_rate = hits / len(job_ids)
    avg_skip_ratio = (skip_sum / roi_count) if roi_count else 0.0
    avg_roi = (roi_sum / roi_count) if roi_count else 0.0

    print(f"[KORITH_DEMO_SUMMARY] hit_rate={hit_rate:.2f} avg_skip_ratio={avg_skip_ratio:.2f} avg_roi={avg_roi:.2f}")


if __name__ == "__main__":
    main()
