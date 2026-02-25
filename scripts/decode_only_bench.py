#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


def _http(method: str, url: str, payload: dict | None = None, api_key: str = "") -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4())}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _build_shapes(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            a, b = token.split(":", 1)
        else:
            a, b = token, "256"
        out.append((max(1, int(a)), max(1, int(b))))
    if not out:
        out = [(256, 256)]
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    idx = max(0, min(len(v) - 1, int(math.ceil(len(v) * p)) - 1))
    return float(v[idx])


def _make_jobspec(
    base: dict[str, Any],
    *,
    prompt_tokens: int,
    max_tokens: int,
    shape_idx: int,
    allow_amf_reuse: bool,
) -> dict[str, Any]:
    jobspec = copy.deepcopy(base)
    jobspec.setdefault("policy", {})
    jobspec["policy"]["allow_amf_reuse"] = bool(allow_amf_reuse)
    jobspec["policy"]["allow_spec"] = False
    jobspec.setdefault("deterministic_cfg", {})
    jobspec["deterministic_cfg"]["max_tokens"] = int(max_tokens)
    data = " ".join(["tok"] * int(prompt_tokens))
    jobspec["input"] = {"content": f"{data}\nshape={shape_idx}"}
    return jobspec


def _run_one(
    base_jobspec: dict[str, Any],
    *,
    url: str,
    api_key: str,
    timeout_s: float,
    prompt_tokens: int,
    max_tokens: int,
    shape_idx: int,
    allow_amf_reuse: bool,
) -> dict[str, Any]:
    jobspec = _make_jobspec(
        base_jobspec,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        shape_idx=shape_idx,
        allow_amf_reuse=allow_amf_reuse,
    )
    t0 = time.time()
    try:
        submit = _http("POST", f"{url}/v1/jobs", payload=jobspec, api_key=api_key)
        job_id = str(submit["job_id"])
    except (HTTPError, URLError) as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            body = str(exc)
        return {
            "job_id": "",
            "status": "SUBMIT_FAILED",
            "shape_idx": int(shape_idx),
            "prompt_tokens_cfg": int(prompt_tokens),
            "max_tokens_cfg": int(max_tokens),
            "elapsed_ms_wall": (time.time() - t0) * 1000.0,
            "error": body,
            "metrics": {},
        }

    deadline = time.time() + timeout_s
    status = {}
    while time.time() < deadline:
        status = _http("GET", f"{url}/v1/jobs/{job_id}", api_key=api_key)
        if str(status.get("status", "")) in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.2)
    metrics = _http("GET", f"{url}/v1/jobs/{job_id}/metrics", api_key=api_key)
    return {
        "job_id": job_id,
        "status": str(status.get("status", "UNKNOWN")),
        "shape_idx": int(shape_idx),
        "prompt_tokens_cfg": int(prompt_tokens),
        "max_tokens_cfg": int(max_tokens),
        "elapsed_ms_wall": (time.time() - t0) * 1000.0,
        "metrics": metrics,
    }


def _validate_row_metrics(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if str(row.get("status", "")) != "SUCCEEDED":
        return out
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        return ["metrics_missing"]
    savings = metrics.get("savings", {}) if isinstance(metrics.get("savings", {}), dict) else {}
    kernels = metrics.get("kernels", {}) if isinstance(metrics.get("kernels", {}), dict) else {}
    spec = metrics.get("spec", {}) if isinstance(metrics.get("spec", {}), dict) else {}
    measurement = metrics.get("measurement", {}) if isinstance(metrics.get("measurement", {}), dict) else {}

    prefill_saved = float(savings.get("prefill_saved_ms", 0.0) or 0.0)
    spec_saved = float(savings.get("spec_saved_ms", 0.0) or 0.0)
    kernels_saved = float(savings.get("kernels_saved_ms", 0.0) or 0.0)
    total_saved = float(savings.get("total_saved_ms", 0.0) or 0.0)
    if min(prefill_saved, spec_saved, kernels_saved, total_saved) < -1e-6:
        out.append("negative_savings_component")
    expected_total = prefill_saved + spec_saved + kernels_saved
    if abs(total_saved - expected_total) > 1e-6:
        out.append("savings_double_count_or_mismatch")
    savings_comparable = bool(savings.get("comparable", measurement.get("savings_comparable", False)))
    if total_saved > 0.0 and not savings_comparable:
        out.append("savings_not_comparable")

    kernels_ms_saved = float(kernels.get("ms_saved", 0.0) or 0.0)
    kernels_applied = bool(kernels.get("kernels_applied", False))
    kernels_comparable = bool(kernels.get("comparable", False))
    if kernels_ms_saved > 0.0 and not kernels_applied:
        out.append("kernel_saved_without_apply")
    if kernels_ms_saved > 0.0 and not kernels_comparable:
        out.append("kernel_saved_not_comparable")

    spec_saved_raw = float(spec.get("saved_ms", 0.0) or 0.0)
    if spec_saved_raw > 0.0 and total_saved <= 0.0:
        out.append("spec_saved_unaccounted")
    spec_comparable = bool(spec.get("comparable", measurement.get("spec_comparable", False)))
    if spec_saved_raw > 0.0 and not spec_comparable:
        out.append("spec_saved_not_comparable")
    return out


def _summarize_rows(rows: list[dict[str, Any]], *, shapes: list[tuple[int, int]]) -> tuple[dict[str, Any], dict[str, dict[str, float]], list[dict[str, Any]]]:
    ok = [r for r in rows if r["status"] == "SUCCEEDED"]
    decode_ms = [float(r["metrics"].get("perf", {}).get("decode_ms", 0.0) or 0.0) for r in ok]
    total_ms = [float(r["metrics"].get("perf", {}).get("total_ms", 0.0) or 0.0) for r in ok]
    tps = [float(r["metrics"].get("perf", {}).get("avg_tps", 0.0) or 0.0) for r in ok]
    tokens_out = [int(r["metrics"].get("perf", {}).get("tokens_out", 0) or 0) for r in ok]

    shape_summary: dict[str, dict[str, float]] = {}
    for idx, (pt, mt) in enumerate(shapes):
        key = f"{pt}:{mt}"
        subset = [r for r in ok if int(r["shape_idx"]) == idx]
        if not subset:
            continue
        d = [float(r["metrics"].get("perf", {}).get("decode_ms", 0.0) or 0.0) for r in subset]
        t = [float(r["metrics"].get("perf", {}).get("total_ms", 0.0) or 0.0) for r in subset]
        shape_summary[key] = {
            "runs": float(len(subset)),
            "decode_ms_avg": float(sum(d) / len(d)),
            "decode_ms_p95": _percentile(d, 0.95),
            "total_ms_avg": float(sum(t) / len(t)),
            "total_ms_p95": _percentile(t, 0.95),
        }

    row_issues = []
    for row in rows:
        issues = _validate_row_metrics(row)
        if issues:
            row_issues.append({"job_id": str(row.get("job_id", "")), "status": str(row.get("status", "")), "issues": issues})

    summary = {
        "succeeded": len(ok),
        "failed": max(0, len(rows) - len(ok)),
        "decode_ms_avg": (sum(decode_ms) / len(decode_ms)) if decode_ms else 0.0,
        "decode_ms_p50": _percentile(decode_ms, 0.5),
        "decode_ms_p95": _percentile(decode_ms, 0.95),
        "total_ms_avg": (sum(total_ms) / len(total_ms)) if total_ms else 0.0,
        "total_ms_p50": _percentile(total_ms, 0.5),
        "total_ms_p95": _percentile(total_ms, 0.95),
        "avg_tps": (sum(tps) / len(tps)) if tps else 0.0,
        "tokens_out_avg": (sum(tokens_out) / len(tokens_out)) if tokens_out else 0.0,
        "invalid_metric_rows": len(row_issues),
    }
    return summary, shape_summary, row_issues


def _warmup_shapes(
    base_jobspec: dict[str, Any],
    *,
    url: str,
    api_key: str,
    timeout_s: float,
    shapes: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    warm_rows = []
    for idx, (prompt_tokens, max_tokens) in enumerate(shapes):
        warm_rows.append(
            _run_one(
                base_jobspec,
                url=url,
                api_key=api_key,
                timeout_s=timeout_s,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                shape_idx=idx,
                allow_amf_reuse=True,
            )
        )
    return warm_rows


def _scenario_id(*, replay_mode: str, concurrency: int) -> str:
    return f"{replay_mode}_c{int(concurrency)}"


def _collect_baseline_map(baseline_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(baseline_json.get("scenarios"), list):
        for item in baseline_json.get("scenarios", []):
            if not isinstance(item, dict):
                continue
            sid = str(item.get("scenario_id", "") or "")
            summary = item.get("summary", {}) if isinstance(item.get("summary", {}), dict) else {}
            if sid and summary:
                out[sid] = summary
    summary_top = baseline_json.get("summary", {}) if isinstance(baseline_json.get("summary", {}), dict) else {}
    if summary_top:
        out.setdefault("default", summary_top)
    return out


def _apply_improvement_gates(
    scenarios: list[dict[str, Any]],
    *,
    baseline_map: dict[str, dict[str, Any]],
    min_improve_pct: float,
) -> dict[str, Any]:
    gate_reasons: list[str] = []
    comparisons: list[dict[str, Any]] = []
    required_metrics = ("decode_ms_p50", "decode_ms_p95", "total_ms_p50", "total_ms_p95")
    for item in scenarios:
        sid = str(item.get("scenario_id", "") or "")
        summary = item.get("summary", {}) if isinstance(item.get("summary", {}), dict) else {}
        baseline = baseline_map.get(sid) or baseline_map.get("default")
        if not baseline:
            gate_reasons.append(f"{sid}:baseline_missing")
            continue
        cmp_row: dict[str, Any] = {"scenario_id": sid, "metrics": {}}
        for key in required_metrics:
            base_val = float(baseline.get(key, 0.0) or 0.0)
            cur_val = float(summary.get(key, 0.0) or 0.0)
            if base_val <= 0.0:
                gate_reasons.append(f"{sid}:{key}:baseline_nonpositive")
                continue
            improve_pct = ((base_val - cur_val) / base_val) * 100.0
            cmp_row["metrics"][key] = {
                "baseline": base_val,
                "current": cur_val,
                "improve_pct": improve_pct,
            }
            if improve_pct < float(min_improve_pct):
                gate_reasons.append(f"{sid}:{key}:improve_pct={improve_pct:.3f}<min={min_improve_pct:.3f}")
        comparisons.append(cmp_row)
    return {
        "passed": len(gate_reasons) == 0,
        "min_improve_pct": float(min_improve_pct),
        "comparisons": comparisons,
        "reasons": gate_reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode-only mixed-shape benchmark")
    ap.add_argument("--jobspec", required=True, help="Base jobspec JSON path")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--matrix-concurrency", default="", help="comma-separated concurrency matrix, e.g. 2,4,8")
    ap.add_argument(
        "--replay-modes",
        default="miss",
        help="comma-separated replay modes: miss,hit,mixed",
    )
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument(
        "--shapes",
        default="256:256,1024:256,4096:256,8192:256",
        help="comma-separated prompt:max_tokens entries",
    )
    ap.add_argument("--baseline", default="", help="Optional baseline JSON for must-pass gates")
    ap.add_argument("--gate-min-improve-pct", type=float, default=0.0)
    ap.add_argument(
        "--strict-metrics",
        default="1",
        help="If true, fail when non-comparable or double-counted metrics are detected",
    )
    ap.add_argument(
        "--require-gates",
        default="0",
        help="If true, non-passing gates return non-zero exit code",
    )
    ap.add_argument("--out", default="", help="Optional JSON output path")
    args = ap.parse_args()

    base_jobspec = json.loads(Path(args.jobspec).read_text(encoding="utf-8"))
    shapes = _build_shapes(args.shapes)

    conc_tokens = [t.strip() for t in str(args.matrix_concurrency or "").split(",") if t.strip()]
    conc_values = [max(1, int(t)) for t in conc_tokens] if conc_tokens else [max(1, int(args.concurrency))]
    replay_modes = [m.strip().lower() for m in str(args.replay_modes).split(",") if m.strip()]
    replay_modes = [m for m in replay_modes if m in ("miss", "hit", "mixed")]
    if not replay_modes:
        replay_modes = ["miss"]

    strict_metrics = str(args.strict_metrics).strip().lower() in ("1", "true", "yes", "on")
    require_gates = str(args.require_gates).strip().lower() in ("1", "true", "yes", "on")

    scenarios: list[dict[str, Any]] = []
    for replay_mode in replay_modes:
        allow_amf_reuse = replay_mode in ("hit", "mixed")
        warmup_rows: list[dict[str, Any]] = []
        if replay_mode == "hit":
            warmup_rows = _warmup_shapes(
                base_jobspec,
                url=args.url.rstrip("/"),
                api_key=args.api_key,
                timeout_s=float(args.timeout_s),
                shapes=shapes,
            )
        for conc in conc_values:
            futures = []
            rows = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(conc))) as ex:
                for i in range(max(1, int(args.runs))):
                    prompt_tokens, max_tokens = shapes[i % len(shapes)]
                    fut = ex.submit(
                        _run_one,
                        base_jobspec,
                        url=args.url.rstrip("/"),
                        api_key=args.api_key,
                        timeout_s=float(args.timeout_s),
                        prompt_tokens=prompt_tokens,
                        max_tokens=max_tokens,
                        shape_idx=i % len(shapes),
                        allow_amf_reuse=allow_amf_reuse,
                    )
                    futures.append(fut)
                for fut in concurrent.futures.as_completed(futures):
                    rows.append(fut.result())
            summary, shape_summary, row_issues = _summarize_rows(rows, shapes=shapes)
            scenarios.append(
                {
                    "scenario_id": _scenario_id(replay_mode=replay_mode, concurrency=conc),
                    "replay_mode": replay_mode,
                    "concurrency": int(conc),
                    "warmup_rows": warmup_rows if replay_mode == "hit" else [],
                    "summary": summary,
                    "shape_summary": shape_summary,
                    "metric_issues": row_issues,
                    "rows": rows,
                }
            )

    gate = {
        "passed": True,
        "reasons": [],
        "comparisons": [],
    }
    if args.baseline:
        baseline_json = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline_map = _collect_baseline_map(baseline_json)
        gate = _apply_improvement_gates(
            scenarios,
            baseline_map=baseline_map,
            min_improve_pct=float(args.gate_min_improve_pct),
        )

    invalid_rows_total = sum(int((s.get("summary", {}) or {}).get("invalid_metric_rows", 0) or 0) for s in scenarios)
    if strict_metrics and invalid_rows_total > 0:
        gate["passed"] = False
        gate.setdefault("reasons", []).append(f"invalid_metric_rows={invalid_rows_total}")

    out = {
        "config": {
            "runs": int(args.runs),
            "concurrency": int(args.concurrency),
            "matrix_concurrency": [int(c) for c in conc_values],
            "replay_modes": replay_modes,
            "shapes": [f"{a}:{b}" for a, b in shapes],
            "decode_only": True,
            "strict_metrics": bool(strict_metrics),
            "baseline": str(args.baseline or ""),
            "gate_min_improve_pct": float(args.gate_min_improve_pct),
        },
        "scenarios": scenarios,
        "gate": gate,
    }

    if len(scenarios) == 1:
        only = scenarios[0]
        out["summary"] = only.get("summary", {})
        out["shape_summary"] = only.get("shape_summary", {})
        out["rows"] = only.get("rows", [])

    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if require_gates and not bool(gate.get("passed", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
