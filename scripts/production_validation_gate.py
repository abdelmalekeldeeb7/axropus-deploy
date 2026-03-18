#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from platform.economics.tenant_meter import TenantMeter  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _run_cmd(cmd: List[str], *, cwd: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    duration_s = time.perf_counter() - started
    return {
        "cmd": cmd,
        "rc": int(proc.returncode),
        "duration_s": float(duration_s),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _find_latest_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _analyze_soak_run(
    soak_dir: Path,
    *,
    min_succeeded: int,
    require_chaos: bool,
    max_hit_rate_drift: float,
    max_p95_drift_ms: float,
    hit_rate_tolerance: float,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []

    summary_path = soak_dir / "report" / "summary.json"
    results_path = soak_dir / "data" / "results.jsonl"
    if not summary_path.exists():
        return {
            "status": "fail",
            "failures": [f"missing soak summary: {summary_path}"],
            "warnings": [],
            "soak_dir": str(soak_dir),
        }

    report = _read_json(summary_path)
    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}
    chaos = report.get("chaos", {}) if isinstance(report.get("chaos", {}), dict) else {}
    drift = summary.get("drift", {}) if isinstance(summary.get("drift", {}), dict) else {}
    per_org_report = report.get("per_org", {}) if isinstance(report.get("per_org", {}), dict) else {}

    succeeded = _safe_int(summary.get("requests_succeeded"), 0)
    if succeeded < min_succeeded:
        failures.append(f"requests_succeeded={succeeded} < min_succeeded={min_succeeded}")

    events_total = _safe_int(chaos.get("events_total"), 0)
    if require_chaos and events_total <= 0:
        failures.append("chaos events missing (events_total=0)")

    hit_rate_delta = abs(_safe_float(drift.get("hit_rate_delta"), 0.0))
    if hit_rate_delta > max_hit_rate_drift:
        failures.append(
            f"hit_rate_drift={hit_rate_delta:.4f} exceeds max_hit_rate_drift={max_hit_rate_drift:.4f}"
        )
    p95_delta = abs(_safe_float(drift.get("p95_ms_delta"), 0.0))
    if p95_delta > max_p95_drift_ms:
        failures.append(f"p95_ms_drift={p95_delta:.3f} exceeds max_p95_drift_ms={max_p95_drift_ms:.3f}")

    rows = _read_jsonl(results_path)
    per_org_meter: Dict[str, TenantMeter] = {}
    for row in rows:
        if str(row.get("status", "")).upper() != "SUCCEEDED":
            continue
        org_id = str(row.get("org_id", "") or "default")
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
        meter = per_org_meter.get(org_id)
        if meter is None:
            meter = TenantMeter(tenant_id=org_id)
            per_org_meter[org_id] = meter
        meter.accumulate(metrics)

    reconciled: Dict[str, Dict[str, Any]] = {}
    all_orgs = sorted(set(per_org_report.keys()) | set(per_org_meter.keys()))
    for org_id in all_orgs:
        rep = per_org_report.get(org_id, {}) if isinstance(per_org_report.get(org_id, {}), dict) else {}
        meter = per_org_meter.get(org_id, TenantMeter(tenant_id=org_id))
        meter_summary = meter.summary()
        expected_count = _safe_int(rep.get("count"), 0)
        expected_hit_rate = _safe_float(rep.get("hit_rate"), 0.0)
        actual_count = _safe_int(meter.total_requests, 0)
        actual_hit_rate = _safe_float(meter_summary.get("amf_hit_rate"), 0.0)
        count_match = expected_count == actual_count
        hit_rate_match = abs(expected_hit_rate - actual_hit_rate) <= hit_rate_tolerance
        if not count_match:
            failures.append(
                f"billing_reconcile[{org_id}] count mismatch report={expected_count} actual={actual_count}"
            )
        if not hit_rate_match:
            failures.append(
                f"billing_reconcile[{org_id}] hit_rate mismatch report={expected_hit_rate:.6f} actual={actual_hit_rate:.6f}"
            )
        reconciled[org_id] = {
            "report_count": expected_count,
            "actual_count": actual_count,
            "report_hit_rate": expected_hit_rate,
            "actual_hit_rate": actual_hit_rate,
            "count_match": count_match,
            "hit_rate_match": hit_rate_match,
        }

    if not rows:
        warnings.append("results.jsonl is empty; billing reconciliation confidence is low.")

    return {
        "status": "pass" if not failures else "fail",
        "soak_dir": str(soak_dir),
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "requests_succeeded": succeeded,
        "chaos_events_total": events_total,
        "drift": {
            "hit_rate_delta": hit_rate_delta,
            "p95_ms_delta": p95_delta,
        },
        "billing_reconciliation": reconciled,
        "failures": failures,
        "warnings": warnings,
    }


def _launch_soak(
    *,
    output_root: Path,
    duration_s: int,
    rps: float,
    concurrency: int,
) -> Tuple[Dict[str, Any], Path | None]:
    before = {p.resolve() for p in output_root.iterdir()} if output_root.exists() else set()
    cmd = [
        "./scripts/run_soak_chaos.sh",
        "--duration-s",
        str(max(1, int(duration_s))),
        "--rps",
        str(float(rps)),
        "--concurrency",
        str(max(1, int(concurrency))),
    ]
    result = _run_cmd(cmd, cwd=REPO_ROOT)
    after = {p.resolve() for p in output_root.iterdir()} if output_root.exists() else set()
    new_dirs = sorted([p for p in (after - before) if p.is_dir()], key=lambda p: p.stat().st_mtime)
    soak_dir = new_dirs[-1] if new_dirs else _find_latest_dir(output_root)
    return result, soak_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production validation gate: consistency matrix + soak/fault checks + billing reconciliation."
    )
    parser.add_argument("--skip-consistency", action="store_true", help="Skip consistency matrix execution.")
    parser.add_argument("--consistency-profile", choices=("quick", "full"), default="full")
    parser.add_argument("--run-soak", action="store_true", help="Launch soak run before analysis.")
    parser.add_argument(
        "--soak-run-dir",
        default="",
        help="Existing soak run directory to analyze. Used when --run-soak is not set.",
    )
    parser.add_argument("--soak-output-root", default="platform_data/soak", help="Soak output root directory.")
    parser.add_argument("--soak-duration-s", type=int, default=900, help="Duration when launching soak.")
    parser.add_argument("--soak-rps", type=float, default=0.25, help="Target RPS when launching soak.")
    parser.add_argument("--soak-concurrency", type=int, default=2, help="Concurrency when launching soak.")
    parser.add_argument("--min-succeeded", type=int, default=1, help="Minimum succeeded requests in soak summary.")
    parser.add_argument("--require-chaos", action="store_true", help="Fail when soak run has zero chaos events.")
    parser.add_argument("--max-hit-rate-drift", type=float, default=0.20)
    parser.add_argument("--max-p95-drift-ms", type=float, default=5000.0)
    parser.add_argument("--hit-rate-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", default="", help="Optional JSON output path for gate report.")
    args = parser.parse_args()

    output_root = Path(args.soak_output_root).expanduser().resolve()

    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "consistency": {},
        "soak": {},
        "status": "pass",
        "failures": [],
    }

    if not args.skip_consistency:
        cmd = [
            sys.executable,
            "scripts/consistency_matrix.py",
            "--profile",
            str(args.consistency_profile),
            "--fail-on-any",
        ]
        consistency = _run_cmd(cmd, cwd=REPO_ROOT)
        report["consistency"] = consistency
        if consistency["rc"] != 0:
            report["failures"].append("consistency_matrix_failed")
    else:
        report["consistency"] = {"status": "skipped"}

    soak_dir: Path | None = None
    if args.run_soak:
        soak_cmd_result, soak_dir = _launch_soak(
            output_root=output_root,
            duration_s=int(args.soak_duration_s),
            rps=float(args.soak_rps),
            concurrency=int(args.soak_concurrency),
        )
        report["soak_launch"] = soak_cmd_result
        if soak_cmd_result["rc"] != 0:
            report["failures"].append("soak_launch_failed")
    elif args.soak_run_dir:
        soak_dir = Path(args.soak_run_dir).expanduser().resolve()
    else:
        soak_dir = _find_latest_dir(output_root)

    if soak_dir is None:
        report["soak"] = {"status": "fail", "failures": ["no_soak_run_dir_found"], "warnings": []}
        report["failures"].append("no_soak_run_dir_found")
    else:
        soak_report = _analyze_soak_run(
            soak_dir=soak_dir,
            min_succeeded=max(0, int(args.min_succeeded)),
            require_chaos=bool(args.require_chaos),
            max_hit_rate_drift=max(0.0, float(args.max_hit_rate_drift)),
            max_p95_drift_ms=max(0.0, float(args.max_p95_drift_ms)),
            hit_rate_tolerance=max(0.0, float(args.hit_rate_tolerance)),
        )
        report["soak"] = soak_report
        if soak_report.get("status") != "pass":
            report["failures"].extend(soak_report.get("failures", []))

    if report["failures"]:
        report["status"] = "fail"

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
