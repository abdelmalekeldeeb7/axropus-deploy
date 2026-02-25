from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ledger import db as ledger
from ..ledger.postgres_store import PostgresLedgerStore
from .model import extract_savings_components


def _parse_targets(raw: str) -> List[float]:
    vals: List[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        v = float(s)
        if v <= 0.0 or v >= 1.0:
            raise ValueError(f"target must be in (0,1): {s}")
        vals.append(v)
    if not vals:
        raise ValueError("at least one target required")
    return sorted(set(vals))


def _safe_float(val: Any) -> float:
    try:
        return float(val or 0.0)
    except Exception:
        return 0.0


def _load_jobs(db_ref: str, limit: int, org_id: Optional[str]) -> List[Dict[str, Any]]:
    if db_ref.startswith("postgres://") or db_ref.startswith("postgresql://"):
        store = PostgresLedgerStore(dsn=db_ref)
        return store.list_jobs(limit=limit, org_id=org_id)
    return ledger.list_jobs(Path(db_ref).resolve(), limit=limit, org_id=org_id)


def _load_latest_run(db_ref: str, job_id: str) -> Optional[Dict[str, Any]]:
    if db_ref.startswith("postgres://") or db_ref.startswith("postgresql://"):
        store = PostgresLedgerStore(dsn=db_ref)
        return store.get_latest_run(job_id)
    return ledger.get_latest_run(Path(db_ref).resolve(), job_id)


def _compute_report(db_ref: str, targets: List[float], org_id: Optional[str], limit: int) -> Dict[str, Any]:
    jobs = _load_jobs(db_ref, limit=limit, org_id=org_id)

    prefill_runtime_sum = 0.0
    decode_runtime_sum = 0.0
    other_runtime_sum = 0.0
    prefill_saved_sum = 0.0
    decode_saved_sum = 0.0
    baseline_total_sum = 0.0
    observed_total_sum = 0.0
    rows = 0

    for job in jobs:
        run = _load_latest_run(db_ref, str(job["job_id"]))
        if not run:
            continue
        metrics_path = str(run.get("metrics_path") or "")
        if not metrics_path:
            continue
        p = Path(metrics_path)
        if not p.exists():
            continue

        try:
            metrics = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        perf = metrics.get("perf", {})
        components = extract_savings_components(metrics)

        total_ms = _safe_float(perf.get("total_ms"))
        prefill_ms = _safe_float(perf.get("prefill_ms"))
        decode_ms = _safe_float(perf.get("decode_ms"))
        prefill_saved_ms = _safe_float(components.get("prefill_saved_ms"))
        decode_saved_ms = _safe_float(components.get("spec_saved_ms")) + _safe_float(
            components.get("kernels_saved_ms")
        )

        other_ms = max(0.0, total_ms - prefill_ms - decode_ms)
        baseline_prefill_ms = prefill_ms + prefill_saved_ms
        baseline_decode_ms = decode_ms + decode_saved_ms
        baseline_total_ms = baseline_prefill_ms + baseline_decode_ms + other_ms

        rows += 1
        prefill_runtime_sum += prefill_ms
        decode_runtime_sum += decode_ms
        other_runtime_sum += other_ms
        prefill_saved_sum += prefill_saved_ms
        decode_saved_sum += decode_saved_ms
        observed_total_sum += total_ms
        baseline_total_sum += baseline_total_ms

    if rows == 0 or baseline_total_sum <= 0.0:
        return {
            "summary": {
                "jobs_analyzed": rows,
                "error": "no_usable_metrics",
            },
            "targets": [],
        }

    prefill_baseline_sum = prefill_runtime_sum + prefill_saved_sum
    decode_baseline_sum = decode_runtime_sum + decode_saved_sum
    current_total_saved = prefill_saved_sum + decode_saved_sum

    prefill_share = prefill_baseline_sum / baseline_total_sum if baseline_total_sum > 0 else 0.0
    decode_share = decode_baseline_sum / baseline_total_sum if baseline_total_sum > 0 else 0.0
    other_share = max(0.0, 1.0 - prefill_share - decode_share)

    prefill_cut = prefill_saved_sum / prefill_baseline_sum if prefill_baseline_sum > 0 else 0.0
    decode_cut = decode_saved_sum / decode_baseline_sum if decode_baseline_sum > 0 else 0.0
    blended_cut = current_total_saved / baseline_total_sum if baseline_total_sum > 0 else 0.0

    target_rows: List[Dict[str, Any]] = []
    for t in targets:
        required_total_saved = t * baseline_total_sum
        required_decode_saved = max(0.0, required_total_saved - prefill_saved_sum)
        required_decode_cut = (
            required_decode_saved / decode_baseline_sum if decode_baseline_sum > 0 else 0.0
        )
        additional_decode_saved = max(0.0, required_decode_saved - decode_saved_sum)
        additional_decode_cut = max(0.0, required_decode_cut - decode_cut)
        feasible = required_decode_cut <= 1.0 + 1e-9
        target_rows.append(
            {
                "target_savings_pct": t * 100.0,
                "required_decode_cut_pct": required_decode_cut * 100.0,
                "additional_decode_cut_pct": additional_decode_cut * 100.0,
                "additional_decode_saved_ms": additional_decode_saved,
                "feasible": bool(feasible),
            }
        )

    return {
        "summary": {
            "jobs_analyzed": rows,
            "org_id": org_id or "all",
            "baseline_total_ms": baseline_total_sum,
            "observed_total_ms": observed_total_sum,
            "current_total_saved_ms": current_total_saved,
            "current_savings_pct": blended_cut * 100.0,
            "prefill_share_pct": prefill_share * 100.0,
            "decode_share_pct": decode_share * 100.0,
            "other_share_pct": other_share * 100.0,
            "prefill_cut_pct": prefill_cut * 100.0,
            "decode_cut_pct": decode_cut * 100.0,
        },
        "targets": target_rows,
    }


def generate_target_report(
    db_path: str | Path,
    out_path: Path,
    targets_csv: str = "0.5,0.6,0.7",
    org_id: Optional[str] = None,
    limit: int = 5000,
) -> Dict[str, Any]:
    db_ref = str(db_path)
    targets = _parse_targets(targets_csv)
    report = _compute_report(db_ref=db_ref, targets=targets, org_id=org_id, limit=limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
