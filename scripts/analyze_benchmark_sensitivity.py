#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _pct(numer: float, denom: float) -> float:
    if denom <= 0.0:
        return 0.0
    return (numer / denom) * 100.0


def _load_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _is_hit_row(row: Dict[str, Any]) -> bool:
    if int(row.get("summary_hit", 0) or 0) == 1:
        return True
    return str(row.get("key", "")).strip().lower() == "hit"


def _is_cold_row(row: Dict[str, Any]) -> bool:
    if int(row.get("summary_hit", 0) or 0) == 0:
        return True
    key = str(row.get("key", "")).strip().lower()
    return key in {"miss", "first_run", "cold"}


def analyze(payload: Dict[str, Any], drop_warm_runs: int, overhead_warn_pct: float) -> Dict[str, Any]:
    rows = _load_rows(payload)
    warm_rows = [r for r in rows if _is_hit_row(r)]
    cold_rows = [r for r in rows if _is_cold_row(r)]

    cold_row = cold_rows[0] if cold_rows else None
    cold_elapsed_ms = _safe_float((cold_row or {}).get("elapsed_ms"), 0.0)
    warm_elapsed = [_safe_float(r.get("elapsed_ms"), 0.0) for r in warm_rows]
    warm_elapsed = [v for v in warm_elapsed if v > 0.0]

    warm_avg_ms = statistics.mean(warm_elapsed) if warm_elapsed else 0.0
    steady_rows = warm_rows[max(0, int(drop_warm_runs)) :]
    steady_elapsed = [_safe_float(r.get("elapsed_ms"), 0.0) for r in steady_rows]
    steady_elapsed = [v for v in steady_elapsed if v > 0.0]
    steady_avg_ms = statistics.mean(steady_elapsed) if steady_elapsed else 0.0

    warm_overhead_ms: List[float] = []
    warm_overhead_share_pct: List[float] = []
    for row in warm_rows:
        elapsed = _safe_float(row.get("elapsed_ms"), 0.0)
        restore = _safe_float(row.get("hit_restore_ms"), 0.0)
        if elapsed <= 0.0 or restore <= 0.0:
            continue
        overhead = max(0.0, elapsed - restore)
        warm_overhead_ms.append(overhead)
        warm_overhead_share_pct.append(_pct(overhead, elapsed))

    first_warm = warm_rows[0] if warm_rows else {}
    first_warm_elapsed = _safe_float(first_warm.get("elapsed_ms"), 0.0)
    first_warm_restore = _safe_float(first_warm.get("hit_restore_ms"), 0.0)
    first_warm_overhead = max(0.0, first_warm_elapsed - first_warm_restore)
    first_warm_overhead_share_pct = _pct(first_warm_overhead, first_warm_elapsed)

    summary_block = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
    hit_rate = _safe_float(summary_block.get("hit_rate"), 0.0)
    if hit_rate <= 0.0 and rows:
        hit_rate = _safe_float(len(warm_rows), 0.0) / _safe_float(len(rows), 1.0)

    e2e_reduction_all = _pct(cold_elapsed_ms - warm_avg_ms, cold_elapsed_ms)
    e2e_reduction_steady = _pct(cold_elapsed_ms - steady_avg_ms, cold_elapsed_ms)

    startup_entries_vals = [int(r.get("startup_entries", 0) or 0) for r in rows]
    startup_entries_avg = statistics.mean(startup_entries_vals) if startup_entries_vals else 0.0

    median_overhead_share = statistics.median(warm_overhead_share_pct) if warm_overhead_share_pct else 0.0
    startup_sensitive = first_warm_overhead_share_pct > (median_overhead_share + overhead_warn_pct)

    warnings: List[str] = []
    if not cold_row:
        warnings.append("No cold row detected; cold-vs-warm reduction may be invalid.")
    if not warm_rows:
        warnings.append("No warm rows detected; cannot estimate warm steady-state behavior.")
    if startup_sensitive:
        warnings.append(
            "First warm run overhead share is materially above steady warm median; "
            "startup/I-O effects are likely dominating observed e2e."
        )

    return {
        "input_rows": len(rows),
        "cold_rows": len(cold_rows),
        "warm_rows": len(warm_rows),
        "hit_rate": hit_rate,
        "cold_elapsed_ms": cold_elapsed_ms,
        "warm_elapsed_avg_ms": warm_avg_ms,
        "steady_warm_elapsed_avg_ms": steady_avg_ms,
        "drop_warm_runs": int(drop_warm_runs),
        "e2e_reduction_pct_cold_vs_warm_avg": e2e_reduction_all,
        "e2e_reduction_pct_cold_vs_steady_warm": e2e_reduction_steady,
        "startup_entries_avg": startup_entries_avg,
        "warm_overhead_ms_median": statistics.median(warm_overhead_ms) if warm_overhead_ms else 0.0,
        "warm_overhead_share_pct_median": median_overhead_share,
        "first_warm_overhead_ms": first_warm_overhead,
        "first_warm_overhead_share_pct": first_warm_overhead_share_pct,
        "startup_sensitive": startup_sensitive,
        "warnings": warnings,
    }


def _print_human(report: Dict[str, Any]) -> None:
    print(f"rows={report.get('input_rows', 0)} cold={report.get('cold_rows', 0)} warm={report.get('warm_rows', 0)}")
    print(
        "cold_ms={:.3f} warm_avg_ms={:.3f} steady_warm_avg_ms={:.3f} hit_rate={:.4f}".format(
            _safe_float(report.get("cold_elapsed_ms", 0.0)),
            _safe_float(report.get("warm_elapsed_avg_ms", 0.0)),
            _safe_float(report.get("steady_warm_elapsed_avg_ms", 0.0)),
            _safe_float(report.get("hit_rate", 0.0)),
        )
    )
    print(
        "e2e_reduction_all={:.4f}% e2e_reduction_steady={:.4f}%".format(
            _safe_float(report.get("e2e_reduction_pct_cold_vs_warm_avg", 0.0)),
            _safe_float(report.get("e2e_reduction_pct_cold_vs_steady_warm", 0.0)),
        )
    )
    print(
        "overhead_median={:.3f}ms overhead_median_share={:.3f}% first_warm_overhead_share={:.3f}% startup_sensitive={}".format(
            _safe_float(report.get("warm_overhead_ms_median", 0.0)),
            _safe_float(report.get("warm_overhead_share_pct_median", 0.0)),
            _safe_float(report.get("first_warm_overhead_share_pct", 0.0)),
            bool(report.get("startup_sensitive", False)),
        )
    )
    for warning in report.get("warnings", []):
        print(f"warning: {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark startup/I-O sensitivity from Axropus benchmark summary JSON."
    )
    parser.add_argument("--input", required=True, help="Path to benchmark summary JSON.")
    parser.add_argument("--output", default="", help="Optional path to write analysis JSON.")
    parser.add_argument("--drop-warm-runs", type=int, default=1, help="Warm runs to drop for steady-state average.")
    parser.add_argument(
        "--overhead-warn-pct",
        type=float,
        default=10.0,
        help="Warn when first warm overhead share exceeds median warm overhead share by this many percent points.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress human-readable stdout summary.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    report = analyze(
        payload=payload,
        drop_warm_runs=max(0, int(args.drop_warm_runs)),
        overhead_warn_pct=max(0.0, float(args.overhead_warn_pct)),
    )
    report["input_path"] = str(input_path)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        _print_human(report)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
