#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path
from typing import Any, Dict
from urllib import request


def _http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _http_no_body(method: str, url: str) -> None:
    req = request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method=method)
    with request.urlopen(req, timeout=120):
        return


def _event_category(name: str) -> str:
    n = str(name or "").lower()
    if not n:
        return "other"
    if "cudalaunchkernel" in n or "cuda graph launch" in n or "cudagraphlaunch" in n:
        return "kernel_launch_overhead"
    if any(k in n for k in ("flash_attn", "flashattn", "attention", "paged_attention", "unified_attention")):
        return "attention"
    if any(k in n for k in ("mlp", "ffn", "gate_up", "down_proj", "silu", "gelu", "fused_moe", "gemm", "fused_mul_mat", "mul_mat")):
        return "mlp"
    if any(k in n for k in ("sample", "sampler", "topk", "multinomial", "argmax", "logits")):
        return "sampling"
    if any(k in n for k in ("kv", "cache_update", "slot_mapping", "append", "unified_kv_cache_update")):
        return "kv_update"
    if "schedule" in n or "scheduler" in n:
        return "scheduler"
    return "other"


def _dur_to_ms(token: str) -> float:
    raw = str(token or "").strip().lower()
    if not raw:
        return 0.0
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(us|ms|s)$", raw)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "us":
        return val / 1000.0
    if unit == "ms":
        return val
    return val * 1000.0


def _load_trace_file(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_trace(profiler_dir: Path) -> Path:
    candidates = sorted(
        [p for p in profiler_dir.rglob("*") if p.name.endswith(".pt.trace.json") or p.name.endswith(".pt.trace.json.gz")],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no torch profiler traces under {profiler_dir}")
    return candidates[-1]


def _extract_trace_breakdown(trace: dict) -> dict:
    totals_ms: Dict[str, float] = {
        "attention": 0.0,
        "mlp": 0.0,
        "sampling": 0.0,
        "kv_update": 0.0,
        "scheduler": 0.0,
        "kernel_launch_overhead": 0.0,
        "other": 0.0,
    }
    events = trace.get("traceEvents", [])
    if not isinstance(events, list):
        return totals_ms
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ph") != "X":
            continue
        name = str(ev.get("name", "") or "")
        dur_us = float(ev.get("dur", 0.0) or 0.0)
        if dur_us <= 0.0:
            continue
        cat = _event_category(name)
        totals_ms[cat] += (dur_us / 1000.0)
    return totals_ms


def _extract_profiler_table_breakdown(table_path: Path) -> dict:
    totals_ms: Dict[str, float] = {
        "attention": 0.0,
        "mlp": 0.0,
        "sampling": 0.0,
        "kv_update": 0.0,
        "scheduler": 0.0,
        "kernel_launch_overhead": 0.0,
        "other": 0.0,
    }
    if not table_path.exists():
        return totals_ms
    for line in table_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.rstrip()
        if not line or line.startswith("-") or line.startswith("Self CPU time total"):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 3:
            continue
        name = parts[0]
        self_cpu_time = parts[2]
        ms = _dur_to_ms(self_cpu_time)
        if ms <= 0.0:
            continue
        cat = _event_category(name)
        totals_ms[cat] += ms
    return totals_ms


def _scheduler_ms_from_trace_file(path: Path, decode_steps: int) -> float:
    if not path.exists():
        return 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    calls = int(payload.get("calls", 0) or 0)
    avg_ms = float(payload.get("avg_ms", 0.0) or 0.0)
    if calls <= 0 or avg_ms <= 0.0:
        return 0.0
    effective_calls = max(1, min(int(calls), int(max(1, decode_steps))))
    return float(effective_calls) * float(avg_ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18001")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--profiler-dir", default="/tmp/axropus_vllm_prof")
    ap.add_argument("--scheduler-trace", default="/tmp/axropus_sched_trace.json")
    ap.add_argument("--output", default="decode_bottleneck_report.json")
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    args = ap.parse_args()

    profiler_dir = Path(args.profiler_dir)
    profiler_dir.mkdir(parents=True, exist_ok=True)

    sched_path = Path(args.scheduler_trace)
    if sched_path.exists():
        sched_path.unlink()

    prompt = " ".join(["tok"] * max(1, int(args.prompt_tokens)))
    payload = {
        "model": args.model_id,
        "prompt": prompt,
        "max_tokens": int(max(1, args.decode_steps)),
        "temperature": 0.0,
        "top_p": 1.0,
    }

    # Warm-up one request to avoid compile/startup pollution.
    _ = _http_json("POST", f"{args.endpoint}/v1/completions", payload=payload)

    # Profile exactly one decode-focused request.
    _http_no_body("POST", f"{args.endpoint}/start_profile")
    t0 = time.time()
    _ = _http_json("POST", f"{args.endpoint}/v1/completions", payload=payload)
    wall_ms = (time.time() - t0) * 1000.0
    _http_no_body("POST", f"{args.endpoint}/stop_profile")

    trace_file = _find_latest_trace(profiler_dir)
    trace = _load_trace_file(trace_file)
    # Prefer profiler_out summary when available (stable op-level view).
    profiler_out = profiler_dir / "profiler_out_0.txt"
    totals_ms = _extract_profiler_table_breakdown(profiler_out)
    if sum(totals_ms.values()) <= 0.0:
        totals_ms = _extract_trace_breakdown(trace)

    # Prefer scheduler timing from our scheduler-side trace when available.
    sched_ms_override = _scheduler_ms_from_trace_file(sched_path, decode_steps=int(max(1, args.decode_steps)))
    if sched_ms_override > 0.0:
        totals_ms["scheduler"] = sched_ms_override

    total_ms = sum(float(v) for v in totals_ms.values())
    if total_ms <= 0.0:
        total_ms = float(wall_ms)

    decode_steps = max(1, int(args.decode_steps))
    breakdown = {}
    for key, val in totals_ms.items():
        pct = (float(val) / float(total_ms)) if total_ms > 0 else 0.0
        breakdown[key] = {
            "total_ms": float(val),
            "per_step_ms": float(val) / float(decode_steps),
            "pct": float(pct),
        }

    ranked = sorted(((k, v["pct"]) for k, v in breakdown.items()), key=lambda kv: kv[1], reverse=True)
    top3 = [{"component": k, "pct": float(p)} for k, p in ranked[:3]]

    report = {
        "decode_steps": int(decode_steps),
        "wall_ms_profiled_request": float(wall_ms),
        "trace_file": str(trace_file),
        "scheduler_trace_file": str(sched_path),
        "breakdown": breakdown,
        "attention_pct": breakdown["attention"]["pct"],
        "mlp_pct": breakdown["mlp"]["pct"],
        "sampling_pct": breakdown["sampling"]["pct"],
        "kv_update_pct": breakdown["kv_update"]["pct"],
        "scheduler_pct": breakdown["scheduler"]["pct"],
        "kernel_launch_overhead_pct": breakdown["kernel_launch_overhead"]["pct"],
        "other_pct": breakdown["other"]["pct"],
        "top3_bottlenecks": top3,
    }

    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(out), "trace": str(trace_file)}, indent=2))


if __name__ == "__main__":
    main()
