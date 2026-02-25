from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import csv

from ..ledger import db as ledger
from ..ledger.postgres_store import PostgresLedgerStore
from .model import estimate_savings


def _write_simple_pdf(path: Path, lines: List[str]) -> None:
    content = ["BT", "/F1 12 Tf", "72 760 Td"]
    for i, line in enumerate(lines):
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if i > 0:
            content.append("0 -16 Td")
        content.append(f"({safe}) Tj")
    content.append("ET")
    stream = "\n".join(content).encode("utf-8")

    objects: List[bytes] = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("utf-8") + stream + b"\nendstream endobj\n"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(out))
        out.extend(obj)
    xref_start = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n".encode("utf-8"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("utf-8"))
    out.extend(
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("utf-8")
    )
    path.write_bytes(bytes(out))


def _split_period(ts: str) -> Tuple[str, str]:
    if "T" in ts:
        day = ts.split("T", 1)[0]
    else:
        day = ts[:10]
    month = day[:7]
    return day, month


def generate_report(db_path: str | Path, out_path: Path, gpu_hourly_cost: float) -> None:
    store = None
    db_ref = str(db_path)
    sqlite_path = Path(db_ref).resolve()
    jobs = ledger.list_jobs(sqlite_path, limit=1000)
    if db_ref.startswith("postgres://") or db_ref.startswith("postgresql://"):
        store = PostgresLedgerStore(dsn=db_ref)
        jobs = store.list_jobs(limit=1000)
    report_rows: List[Dict] = []
    total_saved = 0.0
    total_cost = 0.0
    total_tokens_skipped = 0
    total_tokens_out = 0
    by_day: Dict[str, Dict[str, float]] = {}
    by_month: Dict[str, Dict[str, float]] = {}
    by_gpu: Dict[str, Dict[str, float]] = {}
    by_workload: Dict[str, Dict[str, float]] = {}

    for job in jobs:
        job_id = job["job_id"]
        job_rec = ledger.get_job(sqlite_path, job_id)
        if store is not None:
            latest = store.get_latest_run(job_id)
            job_rec = store.get_job(job_id)
        else:
            latest = ledger.get_latest_run(sqlite_path, job_id)
        if not latest:
            continue
        metrics_path = latest.get("metrics_path")
        if not metrics_path or not Path(metrics_path).exists():
            continue
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        savings = estimate_savings(metrics, gpu_hourly_cost)
        amf = metrics.get("amf", {})
        perf = metrics.get("perf", {})
        sched = metrics.get("scheduling", {})
        ids = metrics.get("ids", {})
        finished_at = ids.get("finished_at", latest.get("finished_at", ""))
        day, month = _split_period(finished_at)
        gpu_id = str(sched.get("gpu_id", "unknown"))
        workload = "unknown"
        if isinstance(job_rec, dict):
            jobspec = job_rec.get("jobspec", {})
            workload = str(jobspec.get("workload", {}).get("type", "unknown")) if isinstance(jobspec, dict) else "unknown"
        if workload == "unknown":
            workload = "default"

        tokens_skipped = int(amf.get("skipped_tokens", 0) or 0)
        tokens_out = int(perf.get("tokens_out", 0) or 0)
        total_tokens_skipped += tokens_skipped
        total_tokens_out += tokens_out

        report_rows.append({
            "job_id": job_id,
            "workload": workload,
            "gpu_id": gpu_id,
            "day": day,
            "month": month,
            "saved_ms": savings["saved_ms"],
            "total_ms": savings["total_ms"],
            "restore_ms": float(amf.get("restore_ms", 0.0) or 0.0),
            "tokens_skipped": tokens_skipped,
            "tokens_out": tokens_out,
            "cost_saved_usd": savings["cost_saved_usd"],
            "cost_total_usd": savings["cost_total_usd"],
            "utilization_uplift": (savings["saved_ms"] / savings["total_ms"]) if savings["total_ms"] > 0 else 0.0,
        })
        total_saved += savings["cost_saved_usd"]
        total_cost += savings["cost_total_usd"]
        for bucket, key in ((by_day, day), (by_month, month), (by_gpu, gpu_id), (by_workload, workload)):
            if key not in bucket:
                bucket[key] = {"jobs": 0.0, "saved_ms": 0.0, "cost_saved_usd": 0.0}
            bucket[key]["jobs"] += 1.0
            bucket[key]["saved_ms"] += savings["saved_ms"]
            bucket[key]["cost_saved_usd"] += savings["cost_saved_usd"]

    report = {
        "summary": {
            "jobs": len(report_rows),
            "total_compute_saved_ms": sum(r["saved_ms"] for r in report_rows),
            "total_tokens_skipped": total_tokens_skipped,
            "total_tokens_out": total_tokens_out,
            "cost_saved_usd": total_saved,
            "cost_total_usd": total_cost,
            "cost_reduction_pct": (total_saved / total_cost * 100.0) if total_cost > 0 else 0.0,
            "avg_replay_roi": (
                sum(float(r.get("saved_ms", 0.0)) for r in report_rows) /
                max(1.0, sum(float(r.get("restore_ms", 0.0)) for r in report_rows))
            ),
        },
        "jobs": report_rows,
        "daily_savings": by_day,
        "monthly_savings": by_month,
        "per_gpu_savings": by_gpu,
        "per_workload_savings": by_workload,
        "totals": {
            "cost_saved_usd": total_saved,
            "cost_total_usd": total_cost,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_id",
                "workload",
                "gpu_id",
                "day",
                "month",
                "saved_ms",
                "total_ms",
                "restore_ms",
                "tokens_skipped",
                "tokens_out",
                "cost_saved_usd",
                "cost_total_usd",
                "utilization_uplift",
            ],
        )
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    summary_path = out_path.with_suffix(".summary.txt")
    summary_path.write_text(
        f"cost_saved_usd={total_saved:.4f}\n"
        f"cost_total_usd={total_cost:.4f}\n",
        encoding="utf-8",
    )

    pdf_path = out_path.parent / "executive_summary.pdf"
    _write_simple_pdf(
        pdf_path,
        [
            "Korith Savings Executive Summary",
            f"Jobs analyzed: {len(report_rows)}",
            f"Total compute saved (ms): {report['summary']['total_compute_saved_ms']:.2f}",
            f"Total cost saved (USD): {total_saved:.4f}",
            f"Total baseline cost (USD): {total_cost:.4f}",
            f"Cost reduction (%): {report['summary']['cost_reduction_pct']:.2f}",
            f"Total skipped tokens: {total_tokens_skipped}",
        ],
    )
