#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class Check:
    dimension: str
    label: str
    description: str
    cmd: List[str]
    required_paths: List[Path]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def unittest_cmd(*cases: str) -> List[str]:
    return [sys.executable, "-m", "unittest", *cases]


def build_checks(profile: str, repo_root: Path, include_cpp: bool) -> List[Check]:
    checks: List[Check] = [
        Check(
            dimension="traffic_entropy",
            label="Traffic entropy",
            description="Routing remains stable across queue/load entropy and lane-specific affinity.",
            cmd=unittest_cmd(
                "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_routing_preference_worker_affinity_then_locality",
                "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_single_node_miss_lane_uses_shape_affinity",
                "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_single_node_miss_lane_shape_affinity_yields_when_overloaded",
            ),
            required_paths=[],
        ),
        Check(
            dimension="multi_tenant_isolation",
            label="Multi-tenant isolation",
            description="Snapshot index and decode cache remain isolated by org/tenant boundaries.",
            cmd=unittest_cmd(
                "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_snapshot_index_org_isolation",
                "platform.tests.test_decode_cache.DecodeCacheStoreTests.test_decode_cache_isolated_by_org_id",
            ),
            required_paths=[],
        ),
        Check(
            dimension="slight_prompt_mutations",
            label="Slight prompt mutations",
            description="Prompt/template canonicalization keeps deterministic cache identity under small formatting changes.",
            cmd=unittest_cmd(
                "platform.tests.test_prompt_canonicalization",
                "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_render_prompt_can_canonicalize_template_json_inputs",
            ),
            required_paths=[],
        ),
        Check(
            dimension="rolling_restarts",
            label="Rolling restarts",
            description="Snapshot lookup survives node-id changes and restart locality shifts.",
            cmd=unittest_cmd(
                "platform.tests.test_phase6.Phase6Tests.test_resolve_snapshot_input_path_accepts_existing_path_when_node_id_changes",
            ),
            required_paths=[],
        ),
        Check(
            dimension="autoscaling_nodes",
            label="Autoscaling nodes",
            description="Node heartbeat and routing selection adapt to live/stale node states.",
            cmd=unittest_cmd(
                "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_node_registry_heartbeat_and_selection",
                "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_node_locality_prefers_live_worker_when_snapshot_worker_is_stale",
            ),
            required_paths=[],
        ),
        Check(
            dimension="storage_churn",
            label="Storage churn",
            description="Tiered snapshot cache enforces budgets and evicts old files under churn.",
            cmd=unittest_cmd(
                "platform.tests.test_phase6.Phase6Tests.test_snapshot_vram_budget_evicts_oldest_cached_file",
            ),
            required_paths=[],
        ),
        Check(
            dimension="long_time_horizons",
            label="Long time horizons",
            description="TTL/cooldown governance handles stale affinity and replay/spec safety over time.",
            cmd=unittest_cmd(
                "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_fingerprint_affinity_ttl_can_disable_stale_affinity",
                "platform.tests.test_phase6.Phase6Tests.test_spec_governance_cooldown_transitions",
            ),
            required_paths=[],
        ),
    ]

    if include_cpp:
        checks.append(
            Check(
                dimension="partial_prefix_overlap",
                label="Partial prefix overlap",
                description="AMF longest-prefix matching hits when a stored prefix is a strict prompt prefix.",
                cmd=["./build/amf_prefix_test"],
                required_paths=[repo_root / "build" / "amf_prefix_test"],
            )
        )

    if profile == "full":
        checks.extend(
            [
                Check(
                    dimension="traffic_entropy_full",
                    label="Traffic entropy (full)",
                    description="Extended lane-role and transfer-vs-recompute routing checks.",
                    cmd=unittest_cmd(
                        "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_single_node_lane_role_filters_worker_selection",
                        "platform.tests.test_phase5_cluster.Phase5ClusterTests.test_transfer_vs_recompute_decision",
                    ),
                    required_paths=[],
                ),
                Check(
                    dimension="rolling_restarts_full",
                    label="Rolling restarts (full)",
                    description="Same-node snapshot preference check after index changes.",
                    cmd=unittest_cmd(
                        "platform.tests.test_phase6.Phase6Tests.test_resolve_snapshot_input_path_prefers_same_node_without_reindex",
                    ),
                    required_paths=[],
                ),
                Check(
                    dimension="storage_churn_full",
                    label="Storage churn (full)",
                    description="Snapshot copy sidecar consistency and tier-promotion behavior.",
                    cmd=unittest_cmd(
                        "platform.tests.test_phase6.Phase6Tests.test_snapshot_cache_copy_keeps_metadata_sidecar",
                        "platform.tests.test_phase6.Phase6Tests.test_promote_snapshot_input_moves_nvme_to_vram_and_records_index",
                    ),
                    required_paths=[],
                ),
                Check(
                    dimension="long_time_horizons_full",
                    label="Long time horizons (full)",
                    description="Decode-governor cooldown behavior under repeated bad streaks.",
                    cmd=unittest_cmd(
                        "platform.tests.test_phase6.Phase6Tests.test_decode_governor_spec_cooldown_after_bad_streak",
                    ),
                    required_paths=[],
                ),
            ]
        )

    return checks


def run_check(check: Check, repo_root: Path, out_dir: Path) -> dict:
    missing = [str(p) for p in check.required_paths if not p.exists()]
    log_path = out_dir / "logs" / f"{check.dimension}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if missing:
        payload = {
            "dimension": check.dimension,
            "label": check.label,
            "description": check.description,
            "status": "skip",
            "reason": "missing_required_paths",
            "missing_paths": missing,
            "command": check.cmd,
            "duration_s": 0.0,
            "return_code": None,
            "log_path": str(log_path),
        }
        log_path.write_text(
            "STATUS: SKIP\n"
            f"MISSING: {json.dumps(missing)}\n"
            f"COMMAND: {' '.join(shlex.quote(part) for part in check.cmd)}\n",
            encoding="utf-8",
        )
        return payload

    env = os.environ.copy()
    existing = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = str(repo_root) if not existing else f"{str(repo_root)}{os.pathsep}{existing}"

    started = time.perf_counter()
    proc = subprocess.run(
        check.cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    duration_s = time.perf_counter() - started
    status = "pass" if proc.returncode == 0 else "fail"
    log_path.write_text(
        "STATUS: "
        + status.upper()
        + "\nCOMMAND: "
        + " ".join(shlex.quote(part) for part in check.cmd)
        + f"\nRETURN_CODE: {proc.returncode}\nDURATION_S: {duration_s:.3f}\n\n=== STDOUT ===\n"
        + (proc.stdout or "")
        + "\n=== STDERR ===\n"
        + (proc.stderr or ""),
        encoding="utf-8",
    )
    return {
        "dimension": check.dimension,
        "label": check.label,
        "description": check.description,
        "status": status,
        "command": check.cmd,
        "duration_s": duration_s,
        "return_code": proc.returncode,
        "log_path": str(log_path),
    }


def write_markdown(results: List[dict], out_path: Path) -> None:
    lines = [
        "# Consistency Matrix",
        "",
        f"Generated at: {utc_now()}",
        "",
        "| Dimension | Status | Duration (s) | Command |",
        "|---|---|---:|---|",
    ]
    for row in results:
        cmd = " ".join(shlex.quote(part) for part in row.get("command", []))
        lines.append(
            f"| {row.get('label','')} | {str(row.get('status','')).upper()} | "
            f"{float(row.get('duration_s', 0.0)):.2f} | `{cmd}` |"
        )
    lines.append("")
    lines.append("## Notes")
    for row in results:
        notes = row.get("reason", "")
        if notes:
            lines.append(f"- {row.get('label', row.get('dimension'))}: {notes}")
    lines.append("")
    lines.append("## Per-check logs")
    for row in results:
        lines.append(f"- {row.get('label', row.get('dimension'))}: `{row.get('log_path', '')}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Axropus consistency checks across core stability dimensions.")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--output-root",
        default="platform_data/consistency",
        help="Directory to store timestamped consistency reports.",
    )
    parser.add_argument("--no-cpp", action="store_true", help="Skip C++ binary checks (e.g., build/amf_prefix_test).")
    parser.add_argument("--fail-on-any", action="store_true", help="Exit non-zero when any check fails.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (repo_root / args.output_root / stamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = build_checks(profile=str(args.profile), repo_root=repo_root, include_cpp=not bool(args.no_cpp))
    results = [run_check(check, repo_root=repo_root, out_dir=out_dir) for check in checks]

    report = {
        "generated_at": utc_now(),
        "profile": str(args.profile),
        "output_dir": str(out_dir),
        "summary": {
            "total": len(results),
            "pass": sum(1 for r in results if r.get("status") == "pass"),
            "fail": sum(1 for r in results if r.get("status") == "fail"),
            "skip": sum(1 for r in results if r.get("status") == "skip"),
        },
        "results": results,
    }
    (out_dir / "consistency_matrix.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(results, out_dir / "consistency_matrix.md")

    print(f"Consistency report: {out_dir}")
    for row in results:
        print(
            f"{str(row.get('status', '')).upper():<5} "
            f"{row.get('label', row.get('dimension', '')):<30} "
            f"{float(row.get('duration_s', 0.0)):>7.2f}s"
        )

    failed = report["summary"]["fail"] > 0
    return 1 if (failed and bool(args.fail_on_any)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
