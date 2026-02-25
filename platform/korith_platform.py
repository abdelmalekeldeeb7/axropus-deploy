#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure local `platform` package shadows stdlib `platform` module.
PKG_ROOT = ROOT / "platform"
if "platform" in sys.modules and not hasattr(sys.modules["platform"], "__path__"):
    del sys.modules["platform"]
if "platform" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "platform",
        PKG_ROOT / "__init__.py",
        submodule_search_locations=[str(PKG_ROOT)],
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["platform"] = module
        spec.loader.exec_module(module)

from platform.gateway.auth import hash_api_key, issue_api_key  # noqa: E402
from platform.runtime.config import apply_config_to_env, build_ledger, load_platform_config  # noqa: E402
from platform.runtime.router_service import run_router, run_node_router  # noqa: E402
from platform.runtime.worker_service import run_worker  # noqa: E402
from platform.economics.report import generate_report  # noqa: E402
from platform.economics.targets import generate_target_report  # noqa: E402


DEFAULT_DB = Path(os.environ.get("KORITH_PLATFORM_DB", "./platform_data/ledger.sqlite")).resolve()
DEFAULT_ARTIFACTS = Path(os.environ.get("KORITH_PLATFORM_ARTIFACTS", "./platform_data/artifacts")).resolve()


def _http(method: str, url: str, payload: dict | None = None, api_key: str | None = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4())}
    key = api_key or os.environ.get("KORITH_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(req) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _load_config(config_path: str | None) -> None:
    apply_config_to_env(load_platform_config(config_path or None))


def cmd_submit(args: argparse.Namespace) -> None:
    _load_config(args.config)
    job = json.loads(Path(args.jobspec).read_text(encoding="utf-8"))
    res = _http("POST", f"{args.url}/v1/jobs", job, api_key=args.api_key)
    print(res["job_id"])


def cmd_status(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/jobs/{args.job_id}", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_metrics(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/jobs/{args.job_id}/metrics", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_events(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/jobs/{args.job_id}/events", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_logs(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/jobs/{args.job_id}/logs", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_history(args: argparse.Namespace) -> None:
    _load_config(args.config)
    query = f"limit={args.limit}"
    if args.org:
        query += f"&org={args.org}"
    res = _http("GET", f"{args.url}/v1/jobs?{query}", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_restore(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("POST", f"{args.url}/v1/jobs/{args.job_id}/restore", {}, api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_capabilities(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/capabilities", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_replay_status(args: argparse.Namespace) -> None:
    _load_config(args.config)
    if args.url:
        res = _http("GET", f"{args.url}/v1/replay/status?fingerprint={args.fingerprint}", api_key=args.api_key)
    else:
        ledger = build_ledger()
        ledger.init()
        res = ledger.get_replay_governance(args.fingerprint) or {"fingerprint_hash": args.fingerprint, "state": "active"}
    print(json.dumps(res, indent=2))

def cmd_spec_status(args: argparse.Namespace) -> None:
    _load_config(args.config)
    if args.url:
        res = _http("GET", f"{args.url}/v1/spec/status?fingerprint={args.fingerprint}", api_key=args.api_key)
    else:
        ledger = build_ledger()
        ledger.init()
        row = ledger.get_spec_governance(args.fingerprint)
        if not row:
            res = {"fingerprint_hash": args.fingerprint, "state": "active"}
        else:
            res = dict(row)
            res["state"] = "disabled" if int(row.get("spec_disabled", 0) or 0) else "active"
    print(json.dumps(res, indent=2))

def cmd_kernels_status(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/kernels/status", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_kpi_summary(args: argparse.Namespace) -> None:
    _load_config(args.config)
    query = f"limit={int(args.limit)}&gpu_hourly_cost={float(args.gpu_hourly_cost)}"
    res = _http("GET", f"{args.url}/v1/kpi/summary?{query}", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_cluster_nodes(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/cluster/nodes", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_cluster_routing(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/cluster/routing?fingerprint={args.fingerprint}", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_snapshot_locations(args: argparse.Namespace) -> None:
    _load_config(args.config)
    res = _http("GET", f"{args.url}/v1/snapshots/locations?fingerprint={args.fingerprint}", api_key=args.api_key)
    print(json.dumps(res, indent=2))


def cmd_node_status(args: argparse.Namespace) -> None:
    _load_config(args.config)
    target = args.url.rstrip("/")
    if args.node_url:
        target = args.node_url.rstrip("/")
    elif args.node_id:
        cluster = _http("GET", f"{args.url}/v1/cluster/nodes", api_key=args.api_key)
        nodes = cluster.get("nodes", [])
        for node in nodes:
            if node.get("node_id") == args.node_id:
                target = f"http://{node.get('host')}:{int(node.get('router_port', 0) or 0)}"
                break
    res = _http("GET", f"{target}/v1/node/status", api_key=args.api_key if target == args.url.rstrip("/") else None)
    print(json.dumps(res, indent=2))


def cmd_key_create(args: argparse.Namespace) -> None:
    _load_config(args.config)
    ledger = build_ledger()
    ledger.init()
    key_id = args.key_id or str(uuid.uuid4())
    raw = issue_api_key(key_id)
    key_hash = hash_api_key(raw)
    permissions_json = args.permissions_json
    from datetime import datetime, timezone
    created_at = args.created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger.create_api_key(
        key_id=key_id,
        key_hash=key_hash,
        org_id=args.org,
        created_at=created_at,
        rate_limit_tpm=args.rate_limit_tpm,
        rate_limit_rpm=args.rate_limit_rpm,
        permissions_json=permissions_json,
    )
    print(json.dumps({"key_id": key_id, "api_key": raw, "org_id": args.org}, indent=2))


def cmd_key_revoke(args: argparse.Namespace) -> None:
    _load_config(args.config)
    ledger = build_ledger()
    ledger.init()
    from datetime import datetime, timezone
    revoked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ok = ledger.revoke_api_key(args.key_id, revoked_at=revoked_at)
    print(json.dumps({"revoked": ok, "key_id": args.key_id}, indent=2))


def cmd_key_list(args: argparse.Namespace) -> None:
    _load_config(args.config)
    ledger = build_ledger()
    ledger.init()
    rows = ledger.list_api_keys(org_id=args.org)
    print(json.dumps({"keys": rows}, indent=2))


def cmd_pilot_run(args: argparse.Namespace) -> None:
    _load_config(args.config)
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    jobs = dataset.get("jobs", []) if isinstance(dataset, dict) else dataset
    if not isinstance(jobs, list):
        raise ValueError("dataset must be list or {\"jobs\": [...]} structure")

    results: list[Dict[str, Any]] = []
    for item in jobs:
        res = _http("POST", f"{args.url}/v1/jobs", item, api_key=args.api_key)
        job_id = res["job_id"]
        deadline = time.time() + float(args.timeout_s)
        status = {}
        while time.time() < deadline:
            status = _http("GET", f"{args.url}/v1/jobs/{job_id}", api_key=args.api_key)
            if status.get("status") in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(0.25)
        metrics = _http("GET", f"{args.url}/v1/jobs/{job_id}/metrics", api_key=args.api_key)
        results.append({"job_id": job_id, "metrics": metrics})

    hits = 0
    total = len(results)
    skip_sum = 0.0
    roi_sum = 0.0
    for row in results:
        amf = row["metrics"].get("amf", {})
        if amf.get("decision") == "hit":
            hits += 1
        skip_sum += float(amf.get("skip_ratio", 0.0) or 0.0)
        roi_sum += float(amf.get("roi", 0.0) or 0.0)

    summary = {
        "runs": total,
        "hit_rate": (hits / total) if total else 0.0,
        "avg_skip_ratio": (skip_sum / total) if total else 0.0,
        "avg_roi": (roi_sum / total) if total else 0.0,
    }
    print(json.dumps({"results": results, "summary": summary}, indent=2))


def cmd_bench_spec(args: argparse.Namespace) -> None:
    _load_config(args.config)
    jobspec = json.loads(Path(args.jobspec).read_text(encoding="utf-8"))
    jobspec.setdefault("policy", {})
    if "allow_spec" in jobspec["policy"]:
        jobspec["policy"]["allow_spec"] = bool(jobspec["policy"]["allow_spec"])
    else:
        jobspec["policy"]["allow_spec"] = os.environ.get("KORITH_SPEC_ENABLED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    jobspec["policy"]["allow_amf_reuse"] = bool(jobspec["policy"].get("allow_amf_reuse", True))

    rows: list[Dict[str, Any]] = []
    runs = max(1, int(args.runs))
    for i in range(runs):
        print(f"bench_spec run {i+1}/{runs}: submit", flush=True)
        res = _http("POST", f"{args.url}/v1/jobs", jobspec, api_key=args.api_key)
        job_id = str(res["job_id"])
        deadline = time.time() + float(args.timeout_s)
        status = {}
        last_status = None
        last_print = 0.0
        while time.time() < deadline:
            status = _http("GET", f"{args.url}/v1/jobs/{job_id}", api_key=args.api_key)
            cur = str(status.get("status", "UNKNOWN"))
            now = time.time()
            if cur != last_status or (now - last_print) >= 2.0:
                print(f"bench_spec run {i+1}/{runs}: job={job_id} status={cur}", flush=True)
                last_status = cur
                last_print = now
            if status.get("status") in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(0.25)
        metrics = _http("GET", f"{args.url}/v1/jobs/{job_id}/metrics", api_key=args.api_key)
        lane_reason = str(status.get("routing_decision", {}).get("lane_reason", "") or "")
        spec_disable_reason = str(metrics.get("spec", {}).get("disable_reason", "") or "")
        print(
            f"bench_spec run {i+1}/{runs}: done status={status.get('status', 'UNKNOWN')} "
            f"lane={metrics.get('scheduling', {}).get('lane', '')} "
            f"spec_enabled={bool(metrics.get('spec', {}).get('enabled', False))} "
            f"lane_reason={lane_reason} "
            f"spec_disable_reason={spec_disable_reason}",
            flush=True,
        )
        rows.append({"job_id": job_id, "status": status.get("status", "UNKNOWN"), "metrics": metrics})

    succeeded = [r for r in rows if r["status"] == "SUCCEEDED"]
    total = len(rows)
    tps = 0.0
    acc = 0.0
    speedup = 0.0
    enabled = 0
    for row in rows:
        m = row["metrics"]
        tps += float(m.get("perf", {}).get("avg_tps", 0.0) or 0.0)
        spec = m.get("spec", {})
        if bool(spec.get("enabled", False)):
            enabled += 1
        acc += float(spec.get("acceptance_rate", 0.0) or 0.0)
        speedup += float(spec.get("speedup_est", 0.0) or 0.0)

    summary = {
        "runs": total,
        "succeeded": len(succeeded),
        "spec_enabled_runs": enabled,
        "avg_tps": (tps / total) if total else 0.0,
        "avg_acceptance_rate": (acc / total) if total else 0.0,
        "avg_speedup_est": (speedup / total) if total else 0.0,
    }
    print(json.dumps({"rows": rows, "summary": summary}, indent=2))


def cmd_savings_target(args: argparse.Namespace) -> None:
    _load_config(args.config)
    report = generate_target_report(
        db_path=args.ledger,
        out_path=Path(args.out),
        targets_csv=args.targets,
        org_id=(args.org or None),
        limit=int(args.limit),
    )
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common_http = argparse.ArgumentParser(add_help=False)
    common_http.add_argument("--url", default="http://127.0.0.1:8000")
    common_http.add_argument("--api-key", default=os.environ.get("KORITH_API_KEY", ""))
    common_http.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))

    s = sub.add_parser("router")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--mode", choices=["cluster", "node"], default=os.environ.get("KORITH_ROUTER_MODE", "cluster"))
    s.add_argument("--node-id", default=os.environ.get("KORITH_NODE_ID", ""))
    s.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))
    s.set_defaults(
        func=lambda args: run_router(args.host, args.port, args.config or None)
        if args.mode == "cluster"
        else run_node_router(args.host, args.port, args.config or None, node_id=args.node_id)
    )

    s = sub.add_parser("worker")
    s.add_argument("--worker-id", default=os.environ.get("KORITH_WORKER_ID", "worker-0"))
    s.add_argument("--gpu-id", type=int, default=int(os.environ.get("KORITH_GPU_ID", "0")))
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=9000)
    s.add_argument("--node-id", default=os.environ.get("KORITH_NODE_ID", ""))
    s.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))
    s.set_defaults(func=lambda args: run_worker(args.worker_id, args.gpu_id, args.host, args.port, args.config or None, node_id=args.node_id))

    s = sub.add_parser("submit", parents=[common_http])
    s.add_argument("jobspec")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("status", parents=[common_http])
    s.add_argument("job_id")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("metrics", parents=[common_http])
    s.add_argument("job_id")
    s.set_defaults(func=cmd_metrics)

    s = sub.add_parser("events", parents=[common_http])
    s.add_argument("job_id")
    s.set_defaults(func=cmd_events)

    s = sub.add_parser("logs", parents=[common_http])
    s.add_argument("job_id")
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("history", parents=[common_http])
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--org", default="")
    s.set_defaults(func=cmd_history)

    s = sub.add_parser("restore", parents=[common_http])
    s.add_argument("job_id")
    s.set_defaults(func=cmd_restore)

    s = sub.add_parser("capabilities", parents=[common_http])
    s.set_defaults(func=cmd_capabilities)

    s = sub.add_parser("report")
    s.add_argument("--ledger", default=str(DEFAULT_DB))
    s.add_argument("--out", default="platform_data/savings_report.json")
    s.add_argument("--gpu_hourly_cost", type=float, default=2.5)
    s.set_defaults(func=lambda args: generate_report(args.ledger, Path(args.out), args.gpu_hourly_cost))

    s = sub.add_parser("savings-target")
    s.add_argument("--ledger", default=str(DEFAULT_DB))
    s.add_argument("--out", default="platform_data/savings_target_report.json")
    s.add_argument("--targets", default="0.5,0.6,0.7")
    s.add_argument("--org", default="")
    s.add_argument("--limit", type=int, default=5000)
    s.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))
    s.set_defaults(func=cmd_savings_target)

    keys = sub.add_parser("keys")
    keys_sub = keys.add_subparsers(dest="keys_cmd", required=True)

    s = keys_sub.add_parser("create")
    s.add_argument("--org", required=True)
    s.add_argument("--key-id", default="")
    s.add_argument("--rate-limit-tpm", type=int, default=120000)
    s.add_argument("--rate-limit-rpm", type=int, default=600)
    s.add_argument("--permissions-json", default="{}")
    s.add_argument("--created-at", default="")
    s.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))
    s.set_defaults(func=cmd_key_create)

    s = keys_sub.add_parser("revoke")
    s.add_argument("key_id")
    s.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))
    s.set_defaults(func=cmd_key_revoke)

    s = keys_sub.add_parser("list")
    s.add_argument("--org", default="")
    s.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))
    s.set_defaults(func=cmd_key_list)

    replay = sub.add_parser("replay")
    replay_sub = replay.add_subparsers(dest="replay_cmd", required=True)
    s = replay_sub.add_parser("status", parents=[common_http])
    s.add_argument("fingerprint")
    s.set_defaults(func=cmd_replay_status)

    spec = sub.add_parser("spec")
    spec_sub = spec.add_subparsers(dest="spec_cmd", required=True)
    s = spec_sub.add_parser("status", parents=[common_http])
    s.add_argument("fingerprint")
    s.set_defaults(func=cmd_spec_status)

    kernels = sub.add_parser("kernels")
    kernels_sub = kernels.add_subparsers(dest="kernels_cmd", required=True)
    s = kernels_sub.add_parser("status", parents=[common_http])
    s.set_defaults(func=cmd_kernels_status)

    kpi = sub.add_parser("kpi")
    kpi_sub = kpi.add_subparsers(dest="kpi_cmd", required=True)
    s = kpi_sub.add_parser("summary", parents=[common_http])
    s.add_argument("--limit", type=int, default=500)
    s.add_argument("--gpu-hourly-cost", type=float, default=2.5)
    s.set_defaults(func=cmd_kpi_summary)

    cluster = sub.add_parser("cluster")
    cluster_sub = cluster.add_subparsers(dest="cluster_cmd", required=True)
    s = cluster_sub.add_parser("nodes", parents=[common_http])
    s.set_defaults(func=cmd_cluster_nodes)
    s = cluster_sub.add_parser("routing", parents=[common_http])
    s.add_argument("--fingerprint", required=True)
    s.set_defaults(func=cmd_cluster_routing)

    snapshots = sub.add_parser("snapshots")
    snapshots_sub = snapshots.add_subparsers(dest="snapshots_cmd", required=True)
    s = snapshots_sub.add_parser("locations", parents=[common_http])
    s.add_argument("--fingerprint", required=True)
    s.set_defaults(func=cmd_snapshot_locations)

    node = sub.add_parser("node")
    node_sub = node.add_subparsers(dest="node_cmd", required=True)
    s = node_sub.add_parser("status", parents=[common_http])
    s.add_argument("--node-id", default="")
    s.add_argument("--node-url", default="")
    s.set_defaults(func=cmd_node_status)

    pilot = sub.add_parser("pilot")
    pilot_sub = pilot.add_subparsers(dest="pilot_cmd", required=True)
    s = pilot_sub.add_parser("run", parents=[common_http])
    s.add_argument("dataset")
    s.add_argument("--timeout-s", type=float, default=120.0)
    s.set_defaults(func=cmd_pilot_run)

    bench = sub.add_parser("bench")
    bench_sub = bench.add_subparsers(dest="bench_cmd", required=True)
    s = bench_sub.add_parser("spec", parents=[common_http])
    s.add_argument("--jobspec", required=True)
    s.add_argument("--runs", type=int, default=10)
    s.add_argument("--timeout-s", type=float, default=180.0)
    s.set_defaults(func=cmd_bench_spec)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
