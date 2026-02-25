#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.environ.get("KORITH_PLATFORM_DIR", ROOT / "platform_data")).resolve()
DEFAULT_BENCH_DIR = DEFAULT_DATA_DIR / "benchmarks"
DEFAULT_LEDGER = DEFAULT_DATA_DIR / "ledger.jsonl"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_job(job_id: str) -> dict | None:
    if not DEFAULT_LEDGER.exists():
        return None
    with DEFAULT_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("job_id") == job_id:
                return rec
    return None


def run_submit(job_json: Path, env: dict) -> str:
    cmd = [str(ROOT / "platform" / "korith_platform.py"), "submit", str(job_json)]
    out = subprocess.check_output(cmd, env=env, text=True).strip()
    return out


def make_job(prompt_text: str, model_path: str, max_tokens: int) -> dict:
    return {
        "schema_version": "korith.job.v1",
        "owner": {"user_id": "bench", "org_id": "local"},
        "workload": {
            "type": "ticket_triage",
            "input": {
                "content_type": "text/plain",
                "source": "inline",
                "data": prompt_text,
            },
            "instructions": "Classify priority, summarize, add tags, and suggest assignee.",
            "constraints": {"max_tokens": max_tokens, "format": "priority, summary, tags, assignee"},
        },
        "runtime": {
            "model_id": Path(model_path).stem,
            "model_path": model_path,
            "seed": 1234,
            "n_ctx": 8192,
            "n_batch": 512,
        },
        "policy": {
            "mode": "deterministic",
            "allow_amf_reuse": True,
            "fail_closed": True,
            "log_level": "info",
            "governance": {
                "record_prompts": True,
                "record_outputs": True,
                "redact_pii": False,
            },
        },
        "tags": ["bench"],
    }


def run_case(case: dict, env: dict) -> dict:
    prompt_file = ROOT / case["prompt_file"]
    prompt_text = prompt_file.read_text(encoding="utf-8")
    model_path = case["model_path"]
    max_tokens = int(case.get("max_tokens", 256))
    shots = int(case.get("shots", 3))

    results = []
    for _ in range(shots):
        job = make_job(prompt_text, model_path, max_tokens)
        tmp_job = ROOT / "platform_data" / "tmp_job.json"
        tmp_job.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        job_id = run_submit(tmp_job, env)
        rec = find_job(job_id)
        results.append({"job_id": job_id, "record": rec})

    return {"case": case, "runs": results}


def summarize_case(case_result: dict) -> dict:
    hits = 0
    roi = []
    skip = []
    tps = []
    for run in case_result["runs"]:
        rec = run.get("record") or {}
        metrics = rec.get("metrics", {})
        if metrics.get("amf_hit"):
            hits += 1
        if "amf_roi" in metrics:
            roi.append(float(metrics["amf_roi"]))
        if "skip_ratio" in metrics:
            skip.append(float(metrics["skip_ratio"]))
        if "avg_tps" in metrics:
            tps.append(float(metrics["avg_tps"]))
    total = len(case_result["runs"])
    return {
        "shots": total,
        "hit_rate": hits / total if total else 0.0,
        "avg_roi": sum(roi) / len(roi) if roi else 0.0,
        "avg_skip_ratio": sum(skip) / len(skip) if skip else 0.0,
        "avg_tps": sum(tps) / len(tps) if tps else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(ROOT / "platform" / "benchmark_suite.json"))
    ap.add_argument("--amf-path", default=str(ROOT / "platform_data" / "bench_amf_store"))
    args = ap.parse_args()

    DEFAULT_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["KORITH_AMF_PATH"] = args.amf_path
    env["KORITH_BENCHMARK_MODE"] = "1"

    suite = read_json(Path(args.suite))
    run_id = str(uuid.uuid4())
    run_out = {"run_id": run_id, "suite_id": suite.get("suite_id"), "cases": []}

    for case in suite.get("cases", []):
        case_result = run_case(case, env)
        summary = summarize_case(case_result)
        run_out["cases"].append({"case": case, "summary": summary, "runs": case_result["runs"]})

    out_path = DEFAULT_BENCH_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(run_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"run_id": run_id, "output": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
