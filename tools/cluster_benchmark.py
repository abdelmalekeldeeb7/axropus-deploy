from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="./platform_data/artifacts")
    parser.add_argument("--job-ids", default="")
    args = parser.parse_args()

    allow_job_ids = set(x.strip() for x in args.job_ids.split(",") if x.strip())
    artifacts = Path(args.artifacts_dir)
    reports = []
    hit_count = 0
    skip_vals = []
    roi_vals = []
    for metrics_path in artifacts.rglob("metrics.json"):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        job_id = data["ids"]["job_id"]
        if allow_job_ids and job_id not in allow_job_ids:
            continue
        decision = data["amf"]["decision"]
        if decision == "hit":
            hit_count += 1
        skip_vals.append(float(data["amf"]["skip_ratio"]))
        roi_vals.append(float(data["amf"]["roi"]))
        reports.append({
            "job_id": job_id,
            "amf_decision": decision,
            "skip_ratio": data["amf"]["skip_ratio"],
            "roi": data["amf"]["roi"],
        })
    print(json.dumps({"reports": reports}, indent=2))
    total = len(reports)
    if total > 0:
        hit_rate = hit_count / total
        avg_skip = sum(skip_vals) / len(skip_vals)
        avg_roi = sum(roi_vals) / len(roi_vals)
    else:
        hit_rate = 0.0
        avg_skip = 0.0
        avg_roi = 0.0
    print(
        f"[KORITH_DEMO_SUMMARY] hit_rate={hit_rate:.2f} "
        f"avg_skip_ratio={avg_skip:.2f} avg_roi={avg_roi:.2f}"
    )


if __name__ == "__main__":
    main()
