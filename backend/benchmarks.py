from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])

DATA_DIR = Path(__file__).resolve().parent / "benchmark_data"
SUMMARY_PATH = DATA_DIR / "summary.json"
RUNS_PATH = DATA_DIR / "Raw_Metrics_817runs_128k.csv"
MIXED_PATH = DATA_DIR / "Mixed_Workload_Model.csv"


def _parse_cell(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return ""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark file missing: {path.name}",
        )


@lru_cache(maxsize=1)
def _load_summary() -> dict[str, Any]:
    _require_file(SUMMARY_PATH)
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_runs() -> list[dict[str, Any]]:
    _require_file(RUNS_PATH)
    with RUNS_PATH.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [{k: _parse_cell(v) for k, v in row.items()} for row in reader]


@lru_cache(maxsize=1)
def _load_mixed() -> list[dict[str, Any]]:
    _require_file(MIXED_PATH)
    with MIXED_PATH.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [{k: _parse_cell(v) for k, v in row.items()} for row in reader]


@router.get("/summary")
def benchmark_summary() -> dict[str, Any]:
    return _load_summary()


@router.get("/runs")
def benchmark_runs() -> dict[str, Any]:
    rows = _load_runs()
    return {
        "count": len(rows),
        "rows": rows,
    }


@router.get("/mixed")
def benchmark_mixed() -> dict[str, Any]:
    rows = _load_mixed()
    return {
        "count": len(rows),
        "rows": rows,
    }
