#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from urllib import request


def _http(method: str, url: str, payload: dict | None, api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Request-Id": str(uuid.uuid4()),
    }
    req = request.Request(url, method=method, data=data, headers=headers)
    with request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def wait_terminal(base_url: str, job_id: str, api_key: str, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        last = _http("GET", f"{base_url}/v1/jobs/{job_id}", None, api_key)
        if last.get("status") in ("SUCCEEDED", "FAILED"):
            return last
        time.sleep(0.25)
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--out", default="platform_data/pilot_run_summary.json")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    jobs = dataset.get("jobs", dataset)
    if not isinstance(jobs, list):
        raise ValueError("dataset must be a list or an object with jobs[]")

    results = []
    for item in jobs:
        submit = _http("POST", f"{args.url}/v1/jobs", item, args.api_key)
        job_id = submit["job_id"]
        wait_terminal(args.url, job_id, args.api_key, args.timeout_s)
        metrics = _http("GET", f"{args.url}/v1/jobs/{job_id}/metrics", None, args.api_key)
        results.append({"job_id": job_id, "metrics": metrics})

    hits = 0
    skip_sum = 0.0
    roi_sum = 0.0
    saved_ms = 0.0
    for row in results:
        amf = row["metrics"].get("amf", {})
        if amf.get("decision") == "hit":
            hits += 1
        skip_sum += float(amf.get("skip_ratio", 0.0) or 0.0)
        roi_sum += float(amf.get("roi", 0.0) or 0.0)
        saved_ms += float(amf.get("saved_ms", 0.0) or 0.0)

    n = len(results)
    summary = {
        "runs": n,
        "hit_rate": (hits / n) if n else 0.0,
        "avg_skip_ratio": (skip_sum / n) if n else 0.0,
        "avg_roi": (roi_sum / n) if n else 0.0,
        "saved_ms_total": saved_ms,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"results": results, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
