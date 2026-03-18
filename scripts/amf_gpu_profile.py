#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class GpuProfile:
    name: str
    n_ctx: int
    n_batch: int
    cuda_dtype: str
    amf_max_bytes: int
    amf_storage_budget_gb: int


PROFILES: Dict[str, GpuProfile] = {
    "h100": GpuProfile(
        name="h100",
        n_ctx=131072,
        n_batch=2048,
        cuda_dtype="bf16",
        amf_max_bytes=64 * 1024**3,
        amf_storage_budget_gb=512,
    ),
    "a100-80g": GpuProfile(
        name="a100-80g",
        n_ctx=131072,
        n_batch=1536,
        cuda_dtype="bf16",
        amf_max_bytes=48 * 1024**3,
        amf_storage_budget_gb=384,
    ),
    "a100-40g": GpuProfile(
        name="a100-40g",
        n_ctx=65536,
        n_batch=1024,
        cuda_dtype="fp16",
        amf_max_bytes=24 * 1024**3,
        amf_storage_budget_gb=192,
    ),
    "consumer-16g": GpuProfile(
        name="consumer-16g",
        n_ctx=8192,
        n_batch=512,
        cuda_dtype="fp16",
        amf_max_bytes=8 * 1024**3,
        amf_storage_budget_gb=64,
    ),
}


def detect_gpu_name_mem() -> Tuple[str, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return "", 0
    if not out:
        return "", 0
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return first, 0
    name = parts[0]
    try:
        mem_mb = int(parts[1])
    except Exception:
        mem_mb = 0
    return name, mem_mb


def choose_profile(profile: str) -> GpuProfile:
    if profile != "auto":
        if profile not in PROFILES:
            raise ValueError(f"unknown profile: {profile}")
        return PROFILES[profile]

    gpu_name, mem_mb = detect_gpu_name_mem()
    name_upper = gpu_name.upper()
    if "H100" in name_upper:
        return PROFILES["h100"]
    if "A100" in name_upper and mem_mb >= 70000:
        return PROFILES["a100-80g"]
    if "A100" in name_upper:
        return PROFILES["a100-40g"]
    return PROFILES["consumer-16g"]


def run_benchmark(
    *,
    model_path: Path,
    prompt_file: Path,
    runs: int,
    max_tokens: int,
    output_root: Path,
    profile: GpuProfile,
) -> Dict[str, object]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"amf_{profile.name}_{stamp}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "amf_store").mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for i in range(1, runs + 1):
        tag = f"run_{i:04d}"
        env = os.environ.copy()
        env["KORITH_ENABLE_AMF"] = "1"
        env["KORITH_ENABLE_MF"] = "1"
        env["KORITH_PROMPT_FILE"] = str(prompt_file)
        env["KORITH_MAX_TOKENS"] = str(max_tokens)
        env["KORITH_N_CTX"] = str(profile.n_ctx)
        env["KORITH_N_BATCH"] = str(profile.n_batch)
        env["KORITH_CUDA_DTYPE"] = profile.cuda_dtype
        env["KORITH_AMF_PATH"] = str(run_dir / "amf_store")
        env["KORITH_AMF_MAX_BYTES"] = str(profile.amf_max_bytes)
        env["KORITH_AMF_STORAGE_BUDGET_GB"] = str(profile.amf_storage_budget_gb)
        env["KORITH_AMF_DISK_GUARD"] = "1"
        env["KORITH_ENGINE_METRICS_PATH"] = str(run_dir / "metrics" / f"{tag}.json")
        env["KORITH_ENGINE_EVENTS_PATH"] = str(run_dir / "metrics" / f"{tag}.events.jsonl")

        t0 = time.perf_counter()
        proc = subprocess.run(
            ["./build/korith_dynamic", str(model_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        (run_dir / "logs" / f"{tag}.out").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "logs" / f"{tag}.err").write_text(proc.stderr, encoding="utf-8")

        row: Dict[str, object] = {
            "run": i,
            "rc": proc.returncode,
            "elapsed_ms": elapsed_ms,
        }
        mpath = run_dir / "metrics" / f"{tag}.json"
        if mpath.exists():
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                m = {}
            amf = m.get("amf", {}) if isinstance(m.get("amf", {}), dict) else {}
            perf = m.get("perf", {}) if isinstance(m.get("perf", {}), dict) else {}
            row["decision"] = amf.get("decision", "")
            row["skip_ratio"] = amf.get("skip_ratio", 0.0)
            row["restore_ms"] = amf.get("restore_ms", 0.0)
            row["prefill_ms"] = perf.get("prefill_ms", 0.0)
            row["total_ms_metric"] = perf.get("total_ms", 0.0)
        rows.append(row)

    cold = [r for r in rows if str(r.get("decision", "")) != "hit"]
    warm = [r for r in rows if str(r.get("decision", "")) == "hit"]
    cold_elapsed = float(cold[0]["elapsed_ms"]) if cold else 0.0
    warm_elapsed = [float(r["elapsed_ms"]) for r in warm]
    warm_avg = (sum(warm_elapsed) / len(warm_elapsed)) if warm_elapsed else 0.0
    hit_rate = (len(warm) / len(rows)) if rows else 0.0
    e2e_reduction = ((cold_elapsed - warm_avg) / cold_elapsed * 100.0) if cold_elapsed > 0 and warm_avg > 0 else 0.0

    summary: Dict[str, object] = {
        "profile": profile.name,
        "gpu_name": detect_gpu_name_mem()[0],
        "runs_requested": runs,
        "runs_ok": sum(1 for r in rows if int(r.get("rc", 1)) == 0),
        "hit_count": len(warm),
        "hit_rate": hit_rate,
        "cold_elapsed_ms": cold_elapsed,
        "warm_elapsed_avg_ms": warm_avg,
        "e2e_reduction_pct_cold_vs_warm_avg": e2e_reduction,
    }
    out = {"rows": rows, "summary": summary}
    out_path = run_dir / "summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "summary_path": str(out_path), "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="AMF GPU profile selector and benchmark runner.")
    parser.add_argument("--profile", default="auto", choices=["auto", *PROFILES.keys()])
    parser.add_argument("--print-env", action="store_true", help="Print export commands for selected profile.")
    parser.add_argument("--bench", action="store_true", help="Run benchmark loop with selected profile.")
    parser.add_argument("--model", default="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--prompt-file", default="workloads/long_prompt.txt")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-root", default="platform_data/bench_profiles")
    args = parser.parse_args()

    profile = choose_profile(args.profile)
    model_path = Path(args.model).resolve()
    prompt_path = Path(args.prompt_file).resolve()

    if args.print_env:
        print(f"export KORITH_N_CTX={profile.n_ctx}")
        print(f"export KORITH_N_BATCH={profile.n_batch}")
        print(f"export KORITH_CUDA_DTYPE={profile.cuda_dtype}")
        print(f"export KORITH_AMF_MAX_BYTES={profile.amf_max_bytes}")
        print(f"export KORITH_AMF_STORAGE_BUDGET_GB={profile.amf_storage_budget_gb}")
        print("export KORITH_AMF_DISK_GUARD=1")

    if args.bench:
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
        result = run_benchmark(
            model_path=model_path,
            prompt_file=prompt_path,
            runs=max(1, int(args.runs)),
            max_tokens=max(1, int(args.max_tokens)),
            output_root=Path(args.output_root).resolve(),
            profile=profile,
        )
        print(json.dumps(result, indent=2))
    else:
        gpu_name, mem_mb = detect_gpu_name_mem()
        print(
            json.dumps(
                {
                    "selected_profile": profile.name,
                    "gpu_name": gpu_name,
                    "gpu_mem_mb": mem_mb,
                    "n_ctx": profile.n_ctx,
                    "n_batch": profile.n_batch,
                    "cuda_dtype": profile.cuda_dtype,
                    "amf_max_bytes": profile.amf_max_bytes,
                    "amf_storage_budget_gb": profile.amf_storage_budget_gb,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
