from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..adapters.registry import AdapterRegistry
from ..artifacts.adapter import ArtifactStoreAdapter
from ..cluster.node_registry import NodeRegistry
from ..cluster.snapshot_index import SnapshotIndex
from ..cluster.snapshot_transfer import estimate_transfer_ms
from ..economics.model import extract_savings_components
from ..economics.tenant_meter import TenantMeter
from ..ledger.store import LedgerStore
from ..observability.metrics import GLOBAL_METRICS
from ..observability.platform_logging import emit_log
from ..queue.base import QueueBase
from .prompt_canonicalization import canonicalize_prompt_text, canonicalize_template_data
from .registry import WorkerRegistry
from .restore_store import RestoreStore

_TRUTHY = ("1", "true", "yes", "on")


class RouterRequestError(ValueError):
    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = str(error_code or "BAD_REQUEST")
        self.status_code = int(status_code)
        self.details = details or {}

    def to_payload(self) -> Dict[str, Any]:
        payload = {"error": str(self), "code": self.error_code}
        payload.update(self.details)
        return payload


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_str(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def fnv1a64_str(data: str) -> str:
    h = 1469598103934665603
    for b in data.encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def _bucket_ceil(value: int, bucket: int) -> int:
    if bucket <= 0:
        return max(0, int(value))
    v = max(0, int(value))
    return ((v + bucket - 1) // bucket) * bucket


class ClusterRouter:
    def __init__(
        self,
        ledger: LedgerStore,
        artifacts: ArtifactStoreAdapter,
        queue: QueueBase,
        registry: WorkerRegistry,
        restore_store: RestoreStore,
        adapter_registry: AdapterRegistry,
        node_registry: Optional[NodeRegistry] = None,
        snapshot_index: Optional[SnapshotIndex] = None,
        amf_coordinator_client = None,
        transfer_bandwidth_mbps: float = 200.0,
        transfer_rtt_ms: float = 5.0,
        transfer_threshold: float = 0.8,
    ) -> None:
        self.ledger = ledger
        self.artifacts = artifacts
        self.queue = queue
        self.registry = registry
        self.restore_store = restore_store
        self.adapter_registry = adapter_registry
        self.node_registry = node_registry
        self.snapshot_index = snapshot_index
        self.amf_coordinator_client = amf_coordinator_client
        self.transfer_bandwidth_mbps = max(1.0, float(transfer_bandwidth_mbps))
        self.transfer_rtt_ms = max(0.0, float(transfer_rtt_ms))
        self.transfer_threshold = max(0.1, float(transfer_threshold))
        self._session_affinity: Dict[str, Dict[str, str]] = {}
        self._fingerprint_affinity: Dict[str, Dict[str, str]] = {}
        self._shape_affinity: Dict[str, Dict[str, str]] = {}
        self._worker_amf_ready: Dict[str, bool] = {}
        self._rr = 0

    def submit(
        self,
        jobspec: Dict[str, Any],
        org_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        route_start = time.time()
        self._validate_jobspec(jobspec)
        resolved_org_id = org_id or self._extract_org_id(jobspec)
        resolved_tenant_id = tenant_id or self._extract_tenant_id(jobspec, resolved_org_id)
        request_id = request_id or str(uuid.uuid4())

        idempotency_key = jobspec.get("idempotency_key")
        if idempotency_key:
            existing = self.ledger.find_job_by_idempotency(idempotency_key, org_id=resolved_org_id)
            if existing:
                return existing

        job_id = jobspec.get("job_id") or str(uuid.uuid4())
        created_at = utc_now()
        prompt = self._render_prompt(jobspec)
        prompt_hash = sha256_str(prompt)
        amf_lookup_hash = fnv1a64_str(f"{resolved_tenant_id}:{prompt}")
        sampling_hash = sha256_str(json.dumps(jobspec.get("deterministic_cfg", {}), sort_keys=True))

        adapter = self.adapter_registry.get_adapter(jobspec)
        prompt_tokens = int(adapter.tokenize(prompt) or 0)
        self._validate_context_budget(jobspec, prompt, prompt_tokens)
        fingerprint = self._fingerprint(jobspec, adapter)
        fingerprint_hash = sha256_str(json.dumps(fingerprint, sort_keys=True))
        shape_key = self._routing_shape_key(jobspec=jobspec, prompt_tokens=prompt_tokens)
        predicted_lane, lane_reason = self._select_lane_with_reason(jobspec, adapter, fingerprint_hash, resolved_org_id)
        deterministic_cfg = jobspec.get("deterministic_cfg", {}) if isinstance(jobspec.get("deterministic_cfg", {}), dict) else {}
        try:
            target_tokens = max(1, int(deterministic_cfg.get("max_tokens", 256) or 256))
        except Exception:
            target_tokens = 256

        routing = self._select_route(
            jobspec=jobspec,
            fingerprint_hash=fingerprint_hash,
            org_id=resolved_org_id,
            tenant_id=resolved_tenant_id,
            amf_lookup_hash=amf_lookup_hash,
            predicted_lane=predicted_lane,
            shape_key=shape_key,
        )
        lane = routing.get("final_lane", predicted_lane)
        execution = self._resolve_execution_target(jobspec=jobspec, lane=str(lane).upper())
        execution_backend_id = str(execution.get("backend_id", jobspec["backend_id"]))
        execution_model = execution.get("model", jobspec.get("model", {}))
        execution_reason = str(execution.get("reason", "requested_backend"))
        exec_jobspec = dict(jobspec)
        exec_jobspec["backend_id"] = execution_backend_id
        exec_jobspec["model"] = execution_model
        execution_fingerprint = fingerprint
        execution_backend_version = adapter.backend_version
        if execution_backend_id != jobspec["backend_id"] or execution_model != jobspec.get("model", {}):
            try:
                exec_adapter = self.adapter_registry.get_adapter(exec_jobspec)
                execution_backend_version = exec_adapter.backend_version
                execution_fingerprint = self._fingerprint(exec_jobspec, exec_adapter)
            except Exception:
                execution_backend_id = jobspec["backend_id"]
                execution_model = jobspec.get("model", {})
                execution_reason = "requested_backend_fallback"
                execution_fingerprint = fingerprint
                execution_backend_version = adapter.backend_version
        chosen_node_id = routing.get("chosen_node_id", "")
        chosen_worker_id = routing.get("chosen_worker_id", "")

        job = {
            "job_id": job_id,
            "created_at": created_at,
            "org_id": resolved_org_id,
            "tenant_id": resolved_tenant_id,
            "request_id": request_id,
            "backend_id": jobspec["backend_id"],
            "backend_version": adapter.backend_version,
            "model": jobspec["model"],
            "execution_backend_id": execution_backend_id,
            "execution_backend_version": execution_backend_version,
            "execution_model": execution_model,
            "prompt_rendered": prompt,
            "prompt_hash": prompt_hash,
            "sampling_hash": sampling_hash,
            "prompt_tokens": prompt_tokens,
            "fingerprint": fingerprint,
            "fingerprint_hash": fingerprint_hash,
            "execution_fingerprint": execution_fingerprint,
            "deterministic_cfg": jobspec.get("deterministic_cfg", {}),
            "policy": jobspec.get("policy", {}),
            "spec_cfg": jobspec.get("spec_cfg", {}),
            "parent_job_id": jobspec.get("parent_job_id"),
            "session_id": jobspec.get("session_id"),
            "routing_decision": {
                "chosen_node_id": chosen_node_id,
                "chosen_worker_id": chosen_worker_id,
                "reason": routing.get("reason", "least_loaded"),
                "predicted_lane": predicted_lane,
                "final_lane": lane,
                "shape_key": shape_key,
                "target_tokens": int(target_tokens),
                "prompt_tokens": int(prompt_tokens),
                "contract_version": 1,
                "lane_reason": lane_reason,
                "transfer_requested": bool(routing.get("transfer_requested", False)),
                "transfer_from_node_id": routing.get("transfer_from_node_id", ""),
                "estimated_transfer_ms": float(routing.get("estimated_transfer_ms", 0.0) or 0.0),
                "estimated_baseline_prefix_ms": float(routing.get("estimated_baseline_prefix_ms", 0.0) or 0.0),
                "snapshot_tier": str(routing.get("snapshot_tier", "")),
                "replay_local": bool(routing.get("replay_local", False)),
                "transfer_from_tier": str(routing.get("transfer_from_tier", "")),
                "execution_backend_id": execution_backend_id,
                "execution_reason": execution_reason,
                "tenant_id": resolved_tenant_id,
                "amf_lookup_hash": amf_lookup_hash,
            },
        }

        self.ledger.insert_job(
            job_id=job_id,
            created_at=created_at,
            jobspec=jobspec,
            fingerprint=fingerprint,
            prompt_hash=prompt_hash,
            fingerprint_hash=fingerprint_hash,
            idempotency_key=idempotency_key,
            status="QUEUED",
            org_id=resolved_org_id,
        )

        artifacts = self.artifacts.init_job(job_id, org_id=resolved_org_id)
        self._append_event(
            artifacts["events"],
            "ROUTER_ACCEPT",
            {"job_id": job_id, "org_id": resolved_org_id, "request_id": request_id},
        )
        self._append_event(
            artifacts["events"],
            "ROUTER_ASSIGN",
            {
                "job_id": job_id,
                "org_id": resolved_org_id,
                "tenant_id": resolved_tenant_id,
                "request_id": request_id,
                "worker_id": chosen_worker_id,
                "node_id": chosen_node_id,
                "lane": lane,
                "reason": routing.get("reason", "least_loaded"),
                "lane_reason": lane_reason,
            },
        )

        emit_log(
            "router",
            "ROUTE_DECISION",
            {
                "job_id": job_id,
                "run_id": "",
                "worker_id": chosen_worker_id,
                "session_id": jobspec.get("session_id", ""),
                "org_id": resolved_org_id,
                "tenant_id": resolved_tenant_id,
                "request_id": request_id,
                "latency_ms": (time.time() - route_start) * 1000.0,
                "lane": lane,
                "reason": routing.get("reason", "least_loaded"),
                "lane_reason": lane_reason,
                "node_id": chosen_node_id,
            },
            artifact_log=artifacts["job_dir"] / "platform.log",
        )

        payload = {
            "job": job,
            "lane": lane,
            "target_node_id": chosen_node_id,
            "target_worker_id": chosen_worker_id,
            "tenant_id": resolved_tenant_id,
            "amf_lookup_hash": amf_lookup_hash,
            "force_miss": bool(routing.get("force_miss", False)),
            "snapshot_transfer": {
                "requested": bool(routing.get("transfer_requested", False)),
                "from_node_id": routing.get("transfer_from_node_id", ""),
                "fingerprint_hash": fingerprint_hash,
                "org_id": resolved_org_id,
                "tenant_id": resolved_tenant_id,
            },
            "enqueued_at": time.time(),
            "request_id": request_id,
        }
        self.queue.enqueue(job_id, payload)

        stats = self.queue.stats()
        if isinstance(stats, dict):
            GLOBAL_METRICS.set_gauge("queue_depth_hit", float(stats.get("queue_depth_hit", 0) or 0))
            GLOBAL_METRICS.set_gauge("queue_depth_miss", float(stats.get("queue_depth_miss", 0) or 0))
        GLOBAL_METRICS.inc("router_jobs_total", 1.0, labels={"org_id": resolved_org_id})
        GLOBAL_METRICS.inc("worker_assignment_count", 1.0, labels={"org_id": resolved_org_id})
        if routing.get("affinity_hit"):
            GLOBAL_METRICS.inc("session_affinity_hits", 1.0, labels={"org_id": resolved_org_id})
        if routing.get("fingerprint_affinity_hit"):
            GLOBAL_METRICS.inc("fingerprint_affinity_hits", 1.0, labels={"org_id": resolved_org_id})
        if routing.get("shape_affinity_hit"):
            GLOBAL_METRICS.inc("shape_affinity_hits", 1.0, labels={"org_id": resolved_org_id})
        GLOBAL_METRICS.set_gauge("routing_latency_ms", (time.time() - route_start) * 1000.0)
        return job_id

    def _resolve_execution_target(self, jobspec: Dict[str, Any], lane: str) -> Dict[str, Any]:
        backend_id = str(jobspec.get("backend_id", ""))
        model = dict(jobspec.get("model", {}) or {})
        resolved = {
            "backend_id": backend_id,
            "model": model,
            "reason": "requested_backend",
        }

        decode_opt_profile = os.environ.get("KORITH_DECODE_OPT_PROFILE", "0").strip().lower() in _TRUTHY
        lane_up = str(lane).upper()
        lane_is_miss = lane_up in ("MISS", "SPEC_MISS")
        lane_is_hit = lane_up in ("HIT", "SPEC_HIT")
        lane_is_spec = lane_up.startswith("SPEC_")
        endpoint, model_id = self._resolve_vllm_target_for_lane(model=model, lane_up=lane_up)
        miss_backend_enabled = os.environ.get("KORITH_VLLM_MISS_BACKEND_ENABLED", "0").strip().lower() in _TRUTHY
        hit_backend_enabled = os.environ.get("KORITH_VLLM_HIT_BACKEND_ENABLED", "0").strip().lower() in _TRUTHY
        if decode_opt_profile and endpoint:
            # Step 3: when decode-opt profile is enabled, allow miss-lane vLLM
            # execution by default if an endpoint is configured.
            miss_backend_enabled = True

        # Keep legacy knob for compatibility while preferring the explicit
        # executor-all-lanes contract toggle.
        executor_all_lanes = os.environ.get("KORITH_VLLM_EXECUTOR_ALL_LANES", "0").strip().lower() in _TRUTHY
        all_lanes = executor_all_lanes or (
            os.environ.get("KORITH_VLLM_ALL_LANES_BACKEND_ENABLED", "0").strip().lower() in _TRUTHY
        )
        spec_backend_enabled = os.environ.get("KORITH_VLLM_SPEC_BACKEND_ENABLED", "0").strip().lower() in _TRUTHY
        if all_lanes:
            lane_enabled = True
        elif lane_is_spec and spec_backend_enabled:
            lane_enabled = True
        elif lane_is_miss:
            lane_enabled = miss_backend_enabled
        elif lane_is_hit:
            lane_enabled = hit_backend_enabled
        else:
            lane_enabled = False
        if not lane_enabled:
            return resolved

        det = jobspec.get("deterministic_cfg", {}) if isinstance(jobspec.get("deterministic_cfg", {}), dict) else {}
        try:
            max_tokens = int(det.get("max_tokens", 0) or 0)
        except Exception:
            max_tokens = 0
        threshold_default = "128" if (decode_opt_profile and lane_is_miss) else "0"
        try:
            threshold_key = "KORITH_VLLM_MISS_MIN_MAX_TOKENS" if lane_is_miss else "KORITH_VLLM_HIT_MIN_MAX_TOKENS"
            min_decode_tokens = max(0, int(os.environ.get(threshold_key, threshold_default) or 0))
        except Exception:
            min_decode_tokens = 0
        if min_decode_tokens > 0 and max_tokens > 0 and max_tokens < min_decode_tokens:
            return resolved

        allowed_sources = {
            item.strip()
            for item in os.environ.get("KORITH_VLLM_MISS_SOURCE_BACKENDS", "korith_local,korith_cuda").split(",")
            if item.strip()
        }
        if allowed_sources and backend_id not in allowed_sources:
            return resolved
        if not endpoint:
            return resolved
        if not model_id:
            return resolved
        resolved["backend_id"] = str(os.environ.get("KORITH_VLLM_BACKEND_ID", "vllm")).strip() or "vllm"
        resolved["model"] = {"model_id": model_id, "endpoint": endpoint}
        resolved["reason"] = "hit_backend_override" if lane_is_hit else "miss_backend_override"
        return resolved

    def _resolve_vllm_target_for_lane(self, *, model: Dict[str, Any], lane_up: str) -> Tuple[str, str]:
        lane = str(lane_up).upper()
        endpoint = str(os.environ.get("KORITH_VLLM_ENDPOINT", "")).strip()
        model_id = str(os.environ.get("KORITH_VLLM_MODEL_ID", "")).strip() or str(model.get("model_id", "")).strip()

        if lane in ("HIT", "SPEC_HIT"):
            endpoint = str(os.environ.get("KORITH_VLLM_HIT_ENDPOINT", "")).strip() or endpoint
            model_id = str(os.environ.get("KORITH_VLLM_HIT_MODEL_ID", "")).strip() or model_id
        elif lane in ("MISS", "SPEC_MISS"):
            endpoint = str(os.environ.get("KORITH_VLLM_MISS_ENDPOINT", "")).strip() or endpoint
            model_id = str(os.environ.get("KORITH_VLLM_MISS_MODEL_ID", "")).strip() or model_id

        if lane.startswith("SPEC_"):
            endpoint = str(os.environ.get("KORITH_VLLM_SPEC_ENDPOINT", "")).strip() or endpoint
            model_id = str(os.environ.get("KORITH_VLLM_SPEC_MODEL_ID", "")).strip() or model_id

        return endpoint, model_id

    def _select_route(
        self,
        jobspec: Dict[str, Any],
        fingerprint_hash: str,
        org_id: str,
        predicted_lane: str,
        tenant_id: str = "default",
        amf_lookup_hash: str = "",
        shape_key: str = "",
    ) -> Dict[str, Any]:
        nodes = self.node_registry.list_nodes() if self.node_registry else []
        workers = self.registry.list_workers()
        self._track_worker_amf_status(workers)
        lane_workers = self._workers_for_lane(workers, predicted_lane)
        if not lane_workers:
            if self._router_bool("KORITH_ROUTER_STRICT_WORKER_LANES", False):
                raise RuntimeError(f"no workers available for lane={predicted_lane}")
            lane_workers = workers
        is_hit_lane = predicted_lane in ("HIT", "SPEC_HIT")
        is_miss_lane = predicted_lane in ("MISS", "SPEC_MISS")
        amf_ready_preferred = False
        ready_worker_ids = {
            str(w.get("worker_id", "") or "")
            for w in workers
            if bool(self._worker_amf_health(w).get("ready", False))
        }
        if is_hit_lane and ready_worker_ids and len(ready_worker_ids) < len(workers):
            warm_lane_workers = [w for w in lane_workers if str(w.get("worker_id", "") or "") in ready_worker_ids]
            if warm_lane_workers:
                lane_workers = warm_lane_workers
                amf_ready_preferred = True
            if nodes:
                ready_node_ids = {
                    str(w.get("node_id", "") or "")
                    for w in workers
                    if str(w.get("worker_id", "") or "") in ready_worker_ids
                }
                warm_nodes = [n for n in nodes if str(n.get("node_id", "") or "") in ready_node_ids]
                if warm_nodes:
                    nodes = warm_nodes
                    amf_ready_preferred = True
        worker_ids = {str(w.get("worker_id") or "") for w in workers if str(w.get("worker_id") or "")}
        lane_worker_ids = {str(w.get("worker_id") or "") for w in lane_workers if str(w.get("worker_id") or "")}
        node_ids = {str(n.get("node_id") or "") for n in nodes if str(n.get("node_id") or "")}
        self._prune_affinity_state(worker_ids=worker_ids, node_ids=node_ids)

        locations = self.snapshot_index.get_locations(fingerprint_hash, org_id) if (self.snapshot_index and is_hit_lane) else []
        session_id = str(jobspec.get("session_id") or "")
        fingerprint_key = f"{org_id}:{fingerprint_hash}"
        use_fingerprint_affinity = is_hit_lane and self._router_bool("KORITH_ROUTER_ENABLE_FINGERPRINT_AFFINITY", True)
        shape_affinity_key = f"{org_id}:{shape_key}" if shape_key else ""
        use_shape_affinity = (
            is_miss_lane
            and self._router_bool("KORITH_ROUTER_ENABLE_SHAPE_AFFINITY", True)
            and bool(shape_affinity_key)
        )

        affinity = self._session_affinity.get(session_id, {}) if session_id else {}
        affinity_node = str(affinity.get("node_id", "") or "")
        affinity_worker = str(affinity.get("worker_id", "") or "")

        fp_affinity = self._fingerprint_affinity.get(fingerprint_key, {}) if use_fingerprint_affinity else {}
        fp_node = str(fp_affinity.get("node_id", "") or "")
        fp_worker = str(fp_affinity.get("worker_id", "") or "")
        shape_affinity = self._shape_affinity.get(shape_affinity_key, {}) if use_shape_affinity else {}
        shape_node = str(shape_affinity.get("node_id", "") or "")
        shape_worker = str(shape_affinity.get("worker_id", "") or "")

        if nodes:
            by_node = {n["node_id"]: n for n in nodes}
            chosen: Dict[str, Any]
            coordinator_rows: list[Dict[str, Any]] = []
            if (
                self.amf_coordinator_client is not None
                and bool(getattr(self.amf_coordinator_client, "enabled", False))
                and bool(amf_lookup_hash)
            ):
                try:
                    coordinator_rows = self.amf_coordinator_client.lookup(amf_lookup_hash, tenant_id)
                except Exception:
                    coordinator_rows = []
            if coordinator_rows:
                candidate_nodes = [row for row in coordinator_rows if str(row.get("node_id", "") or "") in by_node]
                if candidate_nodes:
                    candidate_nodes.sort(
                        key=lambda row: (
                            int(by_node[str(row.get("node_id", "") or "")].get("inflight", 0) or 0),
                            int(by_node[str(row.get("node_id", "") or "")].get("queue_depth_hit", 0) or 0),
                        )
                    )
                    selected = candidate_nodes[0]
                    selected_node_id = str(selected.get("node_id", "") or "")
                    selected_worker_id = self._resolve_worker_id(
                        str(selected.get("worker_id", "") or ""),
                        lane_worker_ids,
                    )
                    if not selected_worker_id:
                        selected_worker_id = self._pick_worker_for_node(selected_node_id, lane_workers)
                    if not selected_worker_id:
                        selected_worker_id = self._pick_worker_for_node(selected_node_id, workers)
                    if selected_worker_id:
                        return {
                            "chosen_node_id": selected_node_id,
                            "chosen_worker_id": selected_worker_id,
                            "reason": "amf_coordinator_cache",
                            "predicted_lane": predicted_lane,
                            "final_lane": predicted_lane,
                            "affinity_hit": False,
                            "fingerprint_affinity_hit": False,
                            "shape_affinity_hit": False,
                            "replay_local": predicted_lane in ("HIT", "SPEC_HIT"),
                            "amf_ready_preferred": bool(amf_ready_preferred),
                            "coordinator_lookup": True,
                        }
            prefer_replay_local = (
                is_hit_lane
                and bool(locations)
                and self._router_bool("KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT", False)
            )
            affinity_has_local_snapshot = bool(
                affinity_node
                and any(str(loc.get("node_id") or "") == affinity_node for loc in locations)
            )
            defer_affinity_for_replay_local = bool(
                affinity_node
                and affinity_node in by_node
                and prefer_replay_local
                and not affinity_has_local_snapshot
            )
            if affinity_node and affinity_node in by_node and not defer_affinity_for_replay_local:
                affinity_worker_id = self._resolve_worker_id(affinity_worker, lane_worker_ids)
                if not affinity_worker_id:
                    affinity_worker_id = self._resolve_worker_id(affinity_worker, worker_ids)
                chosen = {
                    "chosen_node_id": affinity_node,
                    "chosen_worker_id": affinity_worker_id,
                    "reason": "worker_affinity",
                    "predicted_lane": predicted_lane,
                    "final_lane": predicted_lane,
                    "affinity_hit": True,
                    "fingerprint_affinity_hit": False,
                }
            else:
                local_location = None
                for loc in locations:
                    node_id = str(loc.get("node_id") or "")
                    if node_id in by_node:
                        local_location = loc
                        break
                if local_location is not None:
                    local_node = str(local_location.get("node_id") or "")
                    local_worker = self._resolve_worker_id(str(local_location.get("worker_id", "") or ""), lane_worker_ids)
                    if not local_worker:
                        local_worker = self._pick_worker_for_node(local_node, lane_workers)
                    if not local_worker:
                        local_worker = self._resolve_worker_id(str(local_location.get("worker_id", "") or ""), worker_ids)
                    if not local_worker:
                        local_worker = self._pick_worker_for_node(local_node, workers)
                    chosen = {
                        "chosen_node_id": local_node,
                        "chosen_worker_id": local_worker,
                        "reason": "node_locality",
                        "snapshot_tier": str(local_location.get("storage_tier", "unknown") or "unknown"),
                        "predicted_lane": predicted_lane,
                        "final_lane": predicted_lane,
                        "affinity_hit": False,
                        "fingerprint_affinity_hit": False,
                    }
                elif fp_node and fp_node in by_node:
                    fp_worker_id = self._resolve_worker_id(fp_worker, lane_worker_ids)
                    if not fp_worker_id:
                        fp_worker_id = self._resolve_worker_id(fp_worker, worker_ids)
                    chosen = {
                        "chosen_node_id": fp_node,
                        "chosen_worker_id": fp_worker_id,
                        "reason": "fingerprint_affinity",
                        "predicted_lane": predicted_lane,
                        "final_lane": predicted_lane,
                        "affinity_hit": False,
                        "fingerprint_affinity_hit": True,
                    }
                elif shape_node and shape_node in by_node:
                    shape_worker_id = self._resolve_worker_id(shape_worker, lane_worker_ids)
                    if not shape_worker_id:
                        shape_worker_id = self._resolve_worker_id(shape_worker, worker_ids)
                    if not shape_worker_id:
                        shape_worker_id = self._pick_worker_for_node(shape_node, lane_workers)
                    if not shape_worker_id:
                        shape_worker_id = self._pick_worker_for_node(shape_node, workers)
                    if self._shape_affinity_is_overloaded(shape_node, nodes):
                        node = self._least_loaded_node(nodes, predicted_lane)
                        node_id = str(node.get("node_id", "") or "")
                        chosen_worker = self._pick_worker_for_node(node_id, lane_workers)
                        if not chosen_worker:
                            chosen_worker = self._pick_worker_for_node(node_id, workers)
                        chosen = {
                            "chosen_node_id": node_id,
                            "chosen_worker_id": chosen_worker,
                            "reason": "shape_affinity_overloaded",
                            "predicted_lane": predicted_lane,
                            "final_lane": predicted_lane,
                            "affinity_hit": False,
                            "fingerprint_affinity_hit": False,
                            "shape_affinity_hit": False,
                        }
                    else:
                        chosen = {
                            "chosen_node_id": shape_node,
                            "chosen_worker_id": shape_worker_id,
                            "reason": "shape_affinity",
                            "predicted_lane": predicted_lane,
                            "final_lane": predicted_lane,
                            "affinity_hit": False,
                            "fingerprint_affinity_hit": False,
                            "shape_affinity_hit": True,
                        }
                else:
                    # Least loaded node fallback.
                    node = self._least_loaded_node(nodes, predicted_lane)
                    node_id = str(node.get("node_id", "") or "")
                    chosen_worker = self._pick_worker_for_node(node_id, lane_workers)
                    if not chosen_worker:
                        chosen_worker = self._pick_worker_for_node(node_id, workers)
                    chosen = {
                        "chosen_node_id": node_id,
                        "chosen_worker_id": chosen_worker,
                        "reason": "least_loaded",
                        "predicted_lane": predicted_lane,
                        "final_lane": predicted_lane,
                        "affinity_hit": False,
                        "fingerprint_affinity_hit": False,
                        "shape_affinity_hit": False,
                    }

            # If predicted HIT and no local snapshot on chosen node, compare transfer vs recompute.
            if is_hit_lane and locations:
                chosen_node = str(chosen.get("chosen_node_id") or "")
                has_local = any(str(loc.get("node_id") or "") == chosen_node for loc in locations)
                if not has_local:
                    remote, transfer_ms = self._pick_transfer_candidate(locations=locations, chosen_node=chosen_node)
                    if remote is not None:
                        remote_node = str(remote.get("node_id") or "")
                        baseline_prefix_ms = self._estimate_baseline_prefix_ms(fingerprint_hash, org_id)
                        chosen["estimated_transfer_ms"] = transfer_ms
                        chosen["estimated_baseline_prefix_ms"] = baseline_prefix_ms
                        if transfer_ms < (baseline_prefix_ms * self.transfer_threshold):
                            chosen["transfer_requested"] = True
                            chosen["transfer_from_node_id"] = remote_node
                            chosen["transfer_from_tier"] = str(remote.get("storage_tier", "unknown") or "unknown")
                        else:
                            chosen["force_miss"] = True
                            chosen["final_lane"] = "SPEC_MISS" if predicted_lane == "SPEC_HIT" else "MISS"
                            chosen["reason"] = "transfer_too_expensive"
                    else:
                        chosen["force_miss"] = True
                        chosen["final_lane"] = "SPEC_MISS" if predicted_lane == "SPEC_HIT" else "MISS"
                        chosen["reason"] = "transfer_candidate_unavailable"
            if session_id:
                self._session_affinity[session_id] = {
                    "node_id": chosen["chosen_node_id"],
                    "worker_id": chosen["chosen_worker_id"],
                    "ts": str(time.time()),
                }
            if use_fingerprint_affinity and chosen.get("final_lane") in ("HIT", "SPEC_HIT"):
                self._remember_fingerprint_affinity(
                    fingerprint_key,
                    node_id=str(chosen.get("chosen_node_id") or ""),
                    worker_id=str(chosen.get("chosen_worker_id") or ""),
                )
            if use_shape_affinity and chosen.get("final_lane") in ("MISS", "SPEC_MISS"):
                self._remember_shape_affinity(
                    shape_affinity_key,
                    node_id=str(chosen.get("chosen_node_id") or ""),
                    worker_id=str(chosen.get("chosen_worker_id") or ""),
                )
            final_lane = str(chosen.get("final_lane", predicted_lane) or predicted_lane).upper()
            replay_local = False
            if final_lane in ("HIT", "SPEC_HIT"):
                reason_local = str(chosen.get("reason", "") or "")
                replay_local = reason_local in ("node_locality", "worker_affinity", "fingerprint_affinity")
                if bool(chosen.get("transfer_requested", False)):
                    replay_local = False
                if bool(chosen.get("force_miss", False)):
                    replay_local = False
            chosen["replay_local"] = bool(replay_local)
            chosen.setdefault("shape_affinity_hit", False)
            chosen["amf_ready_preferred"] = bool(amf_ready_preferred)
            return chosen

        # Backward-compatible single-node/worker mode.
        worker, reason = self._select_worker_single_node(
            jobspec=jobspec,
            workers=lane_workers,
            fingerprint_hash=fingerprint_hash,
            org_id=org_id,
            predicted_lane=predicted_lane,
            shape_key=shape_affinity_key,
        )
        chosen = {
            "chosen_node_id": "",
            "chosen_worker_id": worker["worker_id"],
            "reason": reason,
            "predicted_lane": predicted_lane,
            "final_lane": predicted_lane,
            "affinity_hit": reason == "worker_affinity",
            "fingerprint_affinity_hit": reason == "fingerprint_affinity",
            "shape_affinity_hit": reason == "shape_affinity",
            "amf_ready_preferred": bool(amf_ready_preferred),
        }
        if session_id:
            self._session_affinity[session_id] = {
                "node_id": "",
                "worker_id": chosen["chosen_worker_id"],
                "ts": str(time.time()),
            }
        if use_fingerprint_affinity and chosen.get("final_lane") in ("HIT", "SPEC_HIT"):
            self._remember_fingerprint_affinity(
                fingerprint_key,
                node_id="",
                worker_id=str(chosen.get("chosen_worker_id") or ""),
            )
        if use_shape_affinity and chosen.get("final_lane") in ("MISS", "SPEC_MISS"):
            self._remember_shape_affinity(
                shape_affinity_key,
                node_id="",
                worker_id=str(chosen.get("chosen_worker_id") or ""),
            )
        final_lane = str(chosen.get("final_lane", predicted_lane) or predicted_lane).upper()
        chosen["replay_local"] = bool(final_lane in ("HIT", "SPEC_HIT"))
        return chosen

    def _workers_for_lane(self, workers: list, lane: str) -> list:
        lane_up = str(lane).upper()
        return [w for w in workers if self._worker_supports_lane(w, lane_up)]

    def _worker_amf_health(self, worker: Dict[str, Any]) -> Dict[str, Any]:
        caps = worker.get("capabilities", {}) if isinstance(worker.get("capabilities", {}), dict) else {}
        try:
            cache_entries = int(caps.get("amf_cache_entries", 0) or 0)
        except Exception:
            cache_entries = 0
        try:
            hit_rate = float(caps.get("amf_hit_rate", 0.0) or 0.0)
        except Exception:
            hit_rate = 0.0
        try:
            warm_ratio = float(caps.get("amf_warm_ratio", 0.0) or 0.0)
        except Exception:
            warm_ratio = 0.0
        prewarm_complete = bool(caps.get("amf_prewarm_complete", False))
        ready = bool(caps.get("amf_ready", False) or prewarm_complete or hit_rate > 0.5)
        return {
            "ready": ready,
            "cache_entries": max(0, cache_entries),
            "hit_rate": max(0.0, min(1.0, hit_rate)),
            "warm_ratio": max(0.0, min(1.0, warm_ratio)),
        }

    def _track_worker_amf_status(self, workers: list) -> None:
        current_ids = set()
        for worker in workers:
            worker_id = str(worker.get("worker_id", "") or "")
            if not worker_id:
                continue
            current_ids.add(worker_id)
            health = self._worker_amf_health(worker)
            ready = bool(health.get("ready", False))
            prev = self._worker_amf_ready.get(worker_id)
            if prev is None or prev != ready:
                node_id = str(worker.get("node_id", "") or "")
                status = "warm" if ready else "warming"
                print(
                    f"[ROUTER_AMF] node={node_id} status={status} "
                    f"entries={int(health.get('cache_entries', 0))} "
                    f"hit_rate={float(health.get('hit_rate', 0.0)):.3f}"
                )
            self._worker_amf_ready[worker_id] = ready
        for stale_id in list(self._worker_amf_ready.keys()):
            if stale_id not in current_ids:
                self._worker_amf_ready.pop(stale_id, None)

    def _worker_supports_lane(self, worker: Dict[str, Any], lane: str) -> bool:
        caps = worker.get("capabilities", {}) if isinstance(worker.get("capabilities", {}), dict) else {}
        lanes_raw = caps.get("lanes", [])
        allowed = set()
        if isinstance(lanes_raw, list):
            allowed = {str(item).strip().upper() for item in lanes_raw if str(item).strip()}
        elif isinstance(lanes_raw, str):
            allowed = {str(item).strip().upper() for item in lanes_raw.split(",") if str(item).strip()}
        if "ALL" in allowed or "*" in allowed:
            return True
        lane_up = str(lane).upper()
        if allowed:
            return lane_up in allowed

        role = str(caps.get("lane_role", "all") or "all").strip().lower()
        if role in ("hit", "hit_only", "hot"):
            return lane_up in ("HIT", "SPEC_HIT")
        if role in ("miss", "miss_only", "cold"):
            return lane_up in ("MISS", "SPEC_MISS")
        return True

    def _estimate_baseline_prefix_ms(self, fingerprint_hash: str, org_id: str) -> float:
        default_ms = float(50.0)
        prior = self.ledger.find_job_by_fingerprint(fingerprint_hash, org_id=org_id)
        if not prior:
            return default_ms
        run = self.ledger.get_latest_run(prior)
        if not run:
            return default_ms
        metrics_path = run.get("metrics_path")
        if not metrics_path:
            return default_ms
        try:
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            amf = metrics.get("amf", {})
            perf = metrics.get("perf", {})
            baseline = float(amf.get("baseline_prefix_ms", 0.0) or 0.0)
            if baseline > 0.0:
                return baseline
            prefill = float(perf.get("prefill_ms", 0.0) or 0.0)
            return prefill if prefill > 0.0 else default_ms
        except Exception:
            return default_ms

    def get_job(self, job_id: str, org_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        rec = self.ledger.get_job(job_id)
        if not rec:
            return None
        if org_id and rec.get("org_id") != org_id:
            return None
        latest = self.ledger.get_latest_run(job_id)
        if latest:
            rec.update(latest)
        return rec

    def list_jobs(self, limit: int = 20, org_id: Optional[str] = None):
        return self.ledger.list_jobs(limit, org_id=org_id)

    def restore(self, job_id: str, org_id: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
        rec = self.ledger.get_job(job_id)
        if not rec:
            GLOBAL_METRICS.inc("mf_restore_failures", 1.0)
            return {"restored": False, "reason": "job_not_found"}
        if org_id and rec.get("org_id") != org_id:
            GLOBAL_METRICS.inc("mf_restore_failures", 1.0, labels={"org_id": org_id})
            return {"restored": False, "reason": "forbidden"}
        snap = self.ledger.get_snapshot_for_job(job_id)
        if not snap:
            GLOBAL_METRICS.inc("mf_restore_failures", 1.0, labels={"org_id": rec.get("org_id", "default")})
            return {"restored": False, "reason": "snapshot_not_found"}
        artifacts = self.artifacts.init_job(job_id, org_id=rec.get("org_id", "default"))

        if snap["fingerprint_hash"] != rec.get("fingerprint_hash"):
            self._append_event(
                artifacts["events"],
                "MF_RESTORE_FAILED",
                {
                    "job_id": job_id,
                    "org_id": rec.get("org_id", "default"),
                    "request_id": request_id,
                    "reason": "fingerprint_mismatch",
                },
            )
            self.ledger.upsert_replay_governance(
                fingerprint_hash=rec.get("fingerprint_hash", ""),
                replay_disabled=1,
                disabled_reason="corruption_detected",
                disabled_at=utc_now(),
                cooldown_until=0.0,
                negative_roi_streak=0,
                corruption_detected=1,
                restore_guard_disabled=0,
                updated_at=utc_now(),
            )
            GLOBAL_METRICS.inc("mf_restore_failures", 1.0, labels={"org_id": rec.get("org_id", "default")})
            GLOBAL_METRICS.inc("corruption_detected_events", 1.0, labels={"org_id": rec.get("org_id", "default")})
            return {"restored": False, "reason": "fingerprint_mismatch"}

        self.restore_store.set(snap["fingerprint_hash"], snap["snapshot_path"])
        self._append_event(
            artifacts["events"],
            "MF_RESTORE",
            {
                "job_id": job_id,
                "org_id": rec.get("org_id", "default"),
                "request_id": request_id,
                "snapshot_path": snap["snapshot_path"],
            },
        )
        return {"restored": True, "snapshot_path": snap["snapshot_path"]}

    def get_workers(self):
        return self.registry.list_workers()

    def queue_status(self):
        stats = self.queue.stats()
        queued = stats.get("QUEUED", 0) if isinstance(stats, dict) else 0
        hit_depth = 0
        miss_depth = 0
        if isinstance(stats, dict):
            hit_depth = int(stats.get("queue_depth_hit", 0) or 0)
            miss_depth = int(stats.get("queue_depth_miss", 0) or 0)
        GLOBAL_METRICS.set_gauge("queue_depth", float(queued))
        GLOBAL_METRICS.set_gauge("queue_depth_hit", float(hit_depth))
        GLOBAL_METRICS.set_gauge("queue_depth_miss", float(miss_depth))
        return stats

    def capabilities(self):
        return self.adapter_registry.list_capabilities()

    def cluster_nodes(self):
        if not self.node_registry:
            return []
        return self.node_registry.list_nodes()

    def routing_debug(self, fingerprint_hash: str, org_id: str = "default") -> Dict[str, Any]:
        nodes = self.node_registry.list_nodes() if self.node_registry else []
        locations = self.snapshot_index.get_locations(fingerprint_hash, org_id) if self.snapshot_index else []
        return {
            "fingerprint_hash": fingerprint_hash,
            "org_id": org_id,
            "nodes": nodes,
            "snapshot_locations": locations,
        }

    def snapshot_locations(self, fingerprint_hash: str, org_id: str = "default") -> Dict[str, Any]:
        if not self.snapshot_index:
            return {"locations": []}
        return {"locations": self.snapshot_index.get_locations(fingerprint_hash, org_id)}

    def spec_status(self, fingerprint_hash: str, org_id: str = "default") -> Dict[str, Any]:
        row = self.ledger.get_spec_governance(fingerprint_hash)
        if not row:
            return {
                "fingerprint_hash": fingerprint_hash,
                "org_id": org_id,
                "state": "active",
                "spec_disabled": 0,
                "bad_accept_streak": 0,
                "cooldown_until": 0.0,
            }
        state = "disabled" if int(row.get("spec_disabled", 0) or 0) else "active"
        out = dict(row)
        out["state"] = state
        return out

    def kernels_status(self) -> Dict[str, Any]:
        workers = self.registry.list_workers()
        return {
            "kernels_enabled": os.environ.get("KORITH_KERNELS", "0"),
            "kernel_backend": os.environ.get("KORITH_KERNEL_BACKEND", "none"),
            "kernel_verify": os.environ.get("KORITH_KERNEL_VERIFY", "0"),
            "accel_enabled": os.environ.get("KORITH_ACCEL_ENABLED", "0"),
            "workers": workers,
            "capabilities": self.adapter_registry.list_capabilities(),
        }

    def kpi_summary(
        self,
        *,
        org_id: str = "default",
        limit: int = 500,
        gpu_hourly_cost: float = 2.5,
    ) -> Dict[str, Any]:
        jobs = self.ledger.list_jobs(limit=max(1, int(limit)), org_id=org_id)
        rows = 0
        succeeded = 0

        observed_total_ms = 0.0
        prefill_runtime_ms = 0.0
        decode_runtime_ms = 0.0
        other_runtime_ms = 0.0
        tokens_out = 0

        prefill_saved_ms = 0.0
        spec_saved_ms = 0.0
        kernels_saved_ms = 0.0
        total_saved_ms = 0.0

        prefill_baseline_ms = 0.0
        decode_baseline_ms = 0.0
        baseline_total_ms = 0.0

        amf_supported_rows = 0
        amf_hit_rows = 0
        amf_skip_ratio_sum = 0.0
        spec_enabled_rows = 0
        kernels_applied_rows = 0
        decode_cache_checked_rows = 0
        decode_cache_hit_rows = 0
        decode_cache_saved_est_ms = 0.0
        decode_cache_tokens_out = 0

        def _safe_float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except Exception:
                return 0.0

        for job in jobs:
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            run = self.ledger.get_latest_run(job_id)
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

            rows += 1
            if int(run.get("exit_code", 0) or 0) == 0:
                succeeded += 1

            perf = metrics.get("perf", {}) if isinstance(metrics.get("perf", {}), dict) else {}
            amf = metrics.get("amf", {}) if isinstance(metrics.get("amf", {}), dict) else {}
            spec = metrics.get("spec", {}) if isinstance(metrics.get("spec", {}), dict) else {}
            kernels = metrics.get("kernels", {}) if isinstance(metrics.get("kernels", {}), dict) else {}
            health = metrics.get("health", {}) if isinstance(metrics.get("health", {}), dict) else {}

            total_ms = max(0.0, _safe_float(perf.get("total_ms", 0.0)))
            prefill_ms = max(0.0, _safe_float(perf.get("prefill_ms", 0.0)))
            decode_ms = max(0.0, _safe_float(perf.get("decode_ms", 0.0)))
            other_ms = max(0.0, total_ms - prefill_ms - decode_ms)

            observed_total_ms += total_ms
            prefill_runtime_ms += prefill_ms
            decode_runtime_ms += decode_ms
            other_runtime_ms += other_ms
            tokens_out += int(perf.get("tokens_out", 0) or 0)

            savings = extract_savings_components(metrics)
            prefill_saved = max(0.0, _safe_float(savings.get("prefill_saved_ms", 0.0)))
            spec_saved = max(0.0, _safe_float(savings.get("spec_saved_ms", 0.0)))
            kernels_saved = max(0.0, _safe_float(savings.get("kernels_saved_ms", 0.0)))
            saved_total = max(0.0, _safe_float(savings.get("total_saved_ms", 0.0)))
            decode_saved = spec_saved + kernels_saved

            prefill_saved_ms += prefill_saved
            spec_saved_ms += spec_saved
            kernels_saved_ms += kernels_saved
            total_saved_ms += saved_total

            baseline_prefill = prefill_ms + prefill_saved
            baseline_decode = decode_ms + decode_saved
            baseline_total = baseline_prefill + baseline_decode + other_ms
            prefill_baseline_ms += baseline_prefill
            decode_baseline_ms += baseline_decode
            baseline_total_ms += baseline_total

            if bool(amf.get("supported", False)):
                amf_supported_rows += 1
                if str(amf.get("decision", "")) == "hit":
                    amf_hit_rows += 1
                sr = _safe_float(amf.get("skip_ratio", 0.0))
                amf_skip_ratio_sum += min(1.0, max(0.0, sr))

            if bool(spec.get("enabled", False)):
                spec_enabled_rows += 1
            if bool(kernels.get("kernels_applied", False)):
                kernels_applied_rows += 1
            decode_cache = health.get("decode_cache", {}) if isinstance(health.get("decode_cache", {}), dict) else {}
            if bool(decode_cache.get("enabled", False)):
                decode_cache_checked_rows += 1
                if bool(decode_cache.get("hit", False)):
                    decode_cache_hit_rows += 1
                    decode_cache_tokens_out += int(perf.get("tokens_out", 0) or 0)
                    decode_cache_saved_est_ms += max(0.0, _safe_float(decode_cache.get("saved_decode_ms_est", 0.0)))

        gpu_cost_per_ms = (float(gpu_hourly_cost) / 3_600_000.0) if gpu_hourly_cost > 0 else 0.0
        baseline_positive = baseline_total_ms > 0.0
        prefill_baseline_positive = prefill_baseline_ms > 0.0
        decode_baseline_positive = decode_baseline_ms > 0.0

        blended_savings_pct = (total_saved_ms / baseline_total_ms * 100.0) if baseline_positive else 0.0
        prefill_share_pct = (prefill_baseline_ms / baseline_total_ms * 100.0) if baseline_positive else 0.0
        decode_share_pct = (decode_baseline_ms / baseline_total_ms * 100.0) if baseline_positive else 0.0
        other_share_pct = max(0.0, 100.0 - prefill_share_pct - decode_share_pct) if baseline_positive else 0.0
        prefill_cut_pct = (prefill_saved_ms / prefill_baseline_ms * 100.0) if prefill_baseline_positive else 0.0
        decode_saved_ms = spec_saved_ms + kernels_saved_ms
        decode_cut_pct = (decode_saved_ms / decode_baseline_ms * 100.0) if decode_baseline_positive else 0.0

        def _decode_cut_needed(target_pct: float) -> Dict[str, Any]:
            if not baseline_positive or not decode_baseline_positive:
                return {"target_savings_pct": target_pct, "required_decode_cut_pct": 0.0, "feasible": False}
            required_total_saved = (target_pct / 100.0) * baseline_total_ms
            required_decode_saved = max(0.0, required_total_saved - prefill_saved_ms)
            required_decode_cut = (required_decode_saved / decode_baseline_ms) * 100.0
            return {
                "target_savings_pct": target_pct,
                "required_decode_cut_pct": required_decode_cut,
                "additional_decode_cut_pct": max(0.0, required_decode_cut - decode_cut_pct),
                "feasible": required_decode_cut <= 100.0 + 1e-9,
            }

        amf_hit_rate_pct = (amf_hit_rows / amf_supported_rows * 100.0) if amf_supported_rows > 0 else 0.0
        avg_skip_ratio = (amf_skip_ratio_sum / amf_supported_rows) if amf_supported_rows > 0 else 0.0
        spec_enabled_rate_pct = (spec_enabled_rows / rows * 100.0) if rows > 0 else 0.0
        kernels_applied_rate_pct = (kernels_applied_rows / rows * 100.0) if rows > 0 else 0.0
        decode_cache_hit_rate_pct = (
            decode_cache_hit_rows / decode_cache_checked_rows * 100.0
        ) if decode_cache_checked_rows > 0 else 0.0

        return {
            "org_id": org_id,
            "generated_at": utc_now(),
            "jobs_seen": len(jobs),
            "jobs_analyzed": rows,
            "jobs_succeeded": succeeded,
            "kpi": {
                "performance": {
                    "tokens_out_total": tokens_out,
                    "observed_total_ms": observed_total_ms,
                    "prefill_runtime_ms": prefill_runtime_ms,
                    "decode_runtime_ms": decode_runtime_ms,
                    "other_runtime_ms": other_runtime_ms,
                },
                "amf": {
                    "supported_rows": amf_supported_rows,
                    "hit_rows": amf_hit_rows,
                    "hit_rate_pct": amf_hit_rate_pct,
                    "avg_skip_ratio": avg_skip_ratio,
                },
                "acceleration": {
                    "spec_enabled_rate_pct": spec_enabled_rate_pct,
                    "kernels_applied_rate_pct": kernels_applied_rate_pct,
                },
                "decode_cache": {
                    "checked_rows": decode_cache_checked_rows,
                    "hit_rows": decode_cache_hit_rows,
                    "hit_rate_pct": decode_cache_hit_rate_pct,
                    "saved_decode_ms_est": decode_cache_saved_est_ms,
                    "avoided_decode_calls": decode_cache_hit_rows,
                    "tokens_served_from_cache": decode_cache_tokens_out,
                },
                "savings": {
                    "prefill_saved_ms": prefill_saved_ms,
                    "spec_saved_ms": spec_saved_ms,
                    "kernels_saved_ms": kernels_saved_ms,
                    "total_saved_ms": total_saved_ms,
                    "baseline_total_ms": baseline_total_ms,
                    "blended_savings_pct": blended_savings_pct,
                    "prefill_share_pct": prefill_share_pct,
                    "decode_share_pct": decode_share_pct,
                    "other_share_pct": other_share_pct,
                    "prefill_cut_pct": prefill_cut_pct,
                    "decode_cut_pct": decode_cut_pct,
                    "decode_targets": [
                        _decode_cut_needed(70.0),
                        _decode_cut_needed(80.0),
                    ],
                },
                "economics": {
                    "gpu_hourly_cost_usd": float(gpu_hourly_cost),
                    "gpu_cost_per_ms_usd": gpu_cost_per_ms,
                    "cost_observed_usd": observed_total_ms * gpu_cost_per_ms,
                    "cost_saved_usd": total_saved_ms * gpu_cost_per_ms,
                    "cost_baseline_usd": baseline_total_ms * gpu_cost_per_ms,
                },
            },
        }

    def tenant_metrics(
        self,
        tenant_id: str,
        *,
        org_id: Optional[str] = None,
        limit: int = 5000,
        gpu_hourly_cost: float = 2.5,
        period_start: str = "",
        period_end: str = "",
    ) -> Dict[str, Any]:
        resolved_org_id = str(org_id or "default")
        tenant_safe = self._extract_tenant_id({"tenant_id": tenant_id}, resolved_org_id)
        meter = TenantMeter(tenant_id=tenant_safe)
        jobs = self.ledger.list_jobs(limit=max(1, int(limit)), org_id=resolved_org_id)

        min_started = period_start.strip()
        max_finished = period_end.strip()
        for job in jobs:
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            rec = self.ledger.get_job(job_id) or {}
            jobspec = rec.get("jobspec", {}) if isinstance(rec.get("jobspec", {}), dict) else {}
            job_tenant = self._extract_tenant_id(jobspec, str(rec.get("org_id", resolved_org_id)))
            if job_tenant != tenant_safe:
                continue
            run = self.ledger.get_latest_run(job_id)
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
            meter.accumulate(metrics)
            started = str(run.get("started_at") or "")
            finished = str(run.get("finished_at") or "")
            if started and (not min_started or started < min_started):
                min_started = started
            if finished and (not max_finished or finished > max_finished):
                max_finished = finished

        return meter.summary(
            gpu_hourly_cost=float(gpu_hourly_cost),
            period_start=min_started,
            period_end=max_finished,
        )

    def billing_report(
        self,
        *,
        org_id: str = "default",
        tenant_id: str = "",
        limit: int = 5000,
        gpu_hourly_cost: float = 2.5,
    ) -> Dict[str, Any]:
        org = str(org_id or "default")
        tenants: set[str] = set()
        if tenant_id:
            tenants.add(self._extract_tenant_id({"tenant_id": tenant_id}, org))
        else:
            jobs = self.ledger.list_jobs(limit=max(1, int(limit)), org_id=org)
            for job in jobs:
                job_id = str(job.get("job_id") or "")
                if not job_id:
                    continue
                rec = self.ledger.get_job(job_id) or {}
                jobspec = rec.get("jobspec", {}) if isinstance(rec.get("jobspec", {}), dict) else {}
                tenants.add(self._extract_tenant_id(jobspec, org))
        reports = [
            self.tenant_metrics(
                tenant,
                org_id=org,
                limit=limit,
                gpu_hourly_cost=gpu_hourly_cost,
            )
            for tenant in sorted(tenants)
        ]
        return {
            "generated_at": utc_now(),
            "org_id": org,
            "reports": reports,
        }

    def create_key(
        self,
        key_id: str,
        key_hash: str,
        org_id: str,
        rate_limit_tpm: int,
        rate_limit_rpm: int,
        permissions_json: str,
    ) -> None:
        self.ledger.create_api_key(
            key_id=key_id,
            key_hash=key_hash,
            org_id=org_id,
            created_at=utc_now(),
            rate_limit_tpm=rate_limit_tpm,
            rate_limit_rpm=rate_limit_rpm,
            permissions_json=permissions_json,
        )

    def revoke_key(self, key_id: str) -> bool:
        return self.ledger.revoke_api_key(key_id=key_id, revoked_at=utc_now())

    def list_keys(self, org_id: Optional[str] = None):
        return self.ledger.list_api_keys(org_id=org_id)

    def replay_status(self, fingerprint_hash: str) -> Dict[str, Any]:
        row = self.ledger.get_replay_governance(fingerprint_hash)
        if not row:
            return {"fingerprint_hash": fingerprint_hash, "state": "active"}
        state = "disabled" if int(row.get("replay_disabled", 0)) else "active"
        return {"fingerprint_hash": fingerprint_hash, "state": state, **row}

    def _render_prompt(self, jobspec: Dict[str, Any]) -> str:
        if jobspec.get("prompt"):
            return canonicalize_prompt_text(str(jobspec["prompt"]))
        template = jobspec.get("prompt_template", "")
        data = jobspec.get("input", {}) or {}
        template = canonicalize_prompt_text(str(template))
        data = canonicalize_template_data(data)
        try:
            rendered = template.format(**data)
        except Exception:
            rendered = template
        return canonicalize_prompt_text(rendered)

    def _extract_org_id(self, jobspec: Dict[str, Any]) -> str:
        owner = jobspec.get("owner") or {}
        org_id = owner.get("org_id") or jobspec.get("org_id") or "default"
        return str(org_id)

    def _extract_tenant_id(self, jobspec: Dict[str, Any], default_org_id: str = "default") -> str:
        owner = jobspec.get("owner") or {}
        tenant_id = owner.get("tenant_id") or jobspec.get("tenant_id") or default_org_id or "default"
        tenant = str(tenant_id or "default").strip() or "default"
        # Keep namespace path-safe and deterministic across services.
        safe = "".join(
            ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_"
            for ch in tenant
        )
        if not safe:
            safe = "default"
        return safe[:128]

    def _fingerprint(self, jobspec: Dict[str, Any], adapter) -> Dict[str, Any]:
        fp = adapter.get_fingerprint()
        cfg = jobspec.get("deterministic_cfg", {})
        fp.update(
            {
                "backend_id": jobspec["backend_id"],
                "model_id": jobspec["model"]["model_id"],
                "model_path": jobspec["model"].get("model_path", ""),
                "endpoint": jobspec["model"].get("endpoint", ""),
                "n_ctx": cfg.get("n_ctx", 0),
                "n_batch": cfg.get("n_batch", 0),
                "sampling_hash": sha256_str(json.dumps(cfg, sort_keys=True)),
            }
        )
        return fp

    def _validate_jobspec(self, jobspec: Dict[str, Any]) -> None:
        required = ["schema_version", "backend_id", "model", "deterministic_cfg", "policy"]
        for key in required:
            if key not in jobspec:
                raise ValueError(f"missing jobspec.{key}")
        if jobspec["schema_version"] != "korith.jobspec.v1":
            raise ValueError("schema_version must be korith.jobspec.v1")
        model = jobspec.get("model", {})
        if not model.get("model_id"):
            raise ValueError("model.model_id required")
        if jobspec["backend_id"] in ("korith_local", "korith_cuda") and not model.get("model_path"):
            raise ValueError("model.model_path required for korith_local/korith_cuda")
        if jobspec["backend_id"] == "hf_transformers" and not (model.get("model_path") or model.get("model_id")):
            raise ValueError("model.model_id or model_path required for hf_transformers")
        if jobspec["backend_id"] in ("openai_compatible", "vllm", "vllm_openai") and not model.get("endpoint"):
            raise ValueError("model.endpoint required for openai_compatible/vllm")
        if not jobspec.get("prompt") and not jobspec.get("prompt_template"):
            raise ValueError("prompt or prompt_template required")
        det = jobspec.get("deterministic_cfg", {})
        for key in ("seed", "n_ctx", "n_batch"):
            if key not in det:
                raise ValueError(f"deterministic_cfg.{key} required")
        policy = jobspec.get("policy", {})
        if "allow_amf_reuse" not in policy or "allow_spec" not in policy:
            raise ValueError("policy.allow_amf_reuse and policy.allow_spec required")

    def _parse_nonnegative_int(self, raw: Any, default: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default

    def _parse_positive_float(self, raw: Any, default: float) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return value if value > 0.0 else default

    def _estimate_prompt_tokens_for_budget(self, prompt: str, prompt_tokens: int) -> int:
        prompt_tokens = max(1, int(prompt_tokens or 0))
        multiplier = self._parse_positive_float(
            os.environ.get("KORITH_CONTEXT_PROMPT_MULTIPLIER", "1.35"),
            1.35,
        )
        chars_per_token_floor = self._parse_positive_float(
            os.environ.get("KORITH_CONTEXT_CHARS_PER_TOKEN_FLOOR", "3.5"),
            3.5,
        )
        char_estimate = int(math.ceil(float(len(prompt)) / chars_per_token_floor))
        mult_estimate = int(math.ceil(float(prompt_tokens) * multiplier))
        frag_scale = self._parse_positive_float(
            os.environ.get("KORITH_CONTEXT_FRAGMENTATION_SCALE", "3.0"),
            3.0,
        )
        frag_max_multiplier = self._parse_positive_float(
            os.environ.get("KORITH_CONTEXT_FRAGMENTATION_MAX_MULTIPLIER", "4.0"),
            4.0,
        )
        min_frag_len = self._parse_nonnegative_int(
            os.environ.get("KORITH_CONTEXT_FRAGMENTATION_MIN_TOKEN_LEN", "12"),
            12,
        )
        undercount_ratio = self._parse_positive_float(
            os.environ.get("KORITH_CONTEXT_FRAGMENTATION_UNDERCOUNT_RATIO", "0.80"),
            0.80,
        )
        words = [w for w in prompt.split() if w]
        risky = 0
        for token in words:
            has_digit = any(ch.isdigit() for ch in token)
            if "_" in token or "-" in token or has_digit or len(token) >= min_frag_len:
                risky += 1
        risky_ratio = float(risky) / float(max(1, len(words)))
        frag_multiplier = min(frag_max_multiplier, 1.0 + (risky_ratio * frag_scale))
        fragment_estimate = prompt_tokens
        # Apply fragmentation boost only when base count is likely underestimating.
        if float(prompt_tokens) < (float(char_estimate) * undercount_ratio):
            fragment_estimate = int(math.ceil(float(prompt_tokens) * frag_multiplier))
        return max(prompt_tokens, char_estimate, mult_estimate, fragment_estimate)

    def _validate_context_budget(self, jobspec: Dict[str, Any], prompt: str, prompt_tokens: int) -> None:
        det = jobspec.get("deterministic_cfg", {}) or {}
        n_ctx = self._parse_nonnegative_int(det.get("n_ctx", 0), 0)
        if n_ctx <= 0:
            return
        max_tokens = self._parse_nonnegative_int(det.get("max_tokens", 256), 256)
        reserve_tokens = self._parse_nonnegative_int(
            os.environ.get("KORITH_CONTEXT_RESERVE_TOKENS", "64"),
            64,
        )
        prompt_tokens_est = self._estimate_prompt_tokens_for_budget(prompt, prompt_tokens)
        required_ctx = prompt_tokens_est + max_tokens + reserve_tokens
        if required_ctx < n_ctx:
            return
        raise RouterRequestError(
            error_code="CONTEXT_OVERFLOW",
            message=(
                f"context_overflow required_ctx={required_ctx} "
                f"available_ctx={n_ctx} prompt_tokens_est={prompt_tokens_est}"
            ),
            status_code=400,
            details={
                "required_ctx": int(required_ctx),
                "available_ctx": int(n_ctx),
                "prompt_tokens": int(max(1, prompt_tokens)),
                "prompt_tokens_est": int(prompt_tokens_est),
                "max_tokens": int(max_tokens),
                "reserve_tokens": int(reserve_tokens),
            },
        )

    def _select_lane(self, jobspec: Dict[str, Any], adapter, fingerprint_hash: str, org_id: str) -> str:
        lane, _ = self._select_lane_with_reason(jobspec, adapter, fingerprint_hash, org_id)
        return lane

    def _select_lane_with_reason(self, jobspec: Dict[str, Any], adapter, fingerprint_hash: str, org_id: str) -> Tuple[str, str]:
        caps = adapter.get_capabilities()
        policy = jobspec.get("policy", {})
        spec_cfg = jobspec.get("spec_cfg", {}) or {}
        spec_requested = bool(policy.get("allow_spec", False))
        spec_reason = "policy_spec_off" if not spec_requested else "requested"

        def _env_bool(name: str, default: bool = False) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in ("1", "true", "yes", "on")

        # Route into SPEC lanes only when runtime spec is enabled on this router.
        runtime_spec_enabled = _env_bool("KORITH_SPEC_ENABLED", False)
        spec_miss_only = _env_bool("KORITH_SPEC_MISS_ONLY", False)
        spec_ignore_governance = _env_bool("KORITH_SPEC_IGNORE_GOVERNANCE", False)

        def _finalize_lane(lane_value: str, reason_value: str) -> Tuple[str, str]:
            lane_up = str(lane_value).upper()
            reason = str(reason_value or "")
            if spec_miss_only and lane_up == "SPEC_HIT":
                return "HIT", "spec_miss_only"
            return lane_up, reason

        if not runtime_spec_enabled:
            spec_requested = False
            spec_reason = "runtime_spec_disabled"
        if spec_requested:
            max_tokens = int((jobspec.get("deterministic_cfg", {}) or {}).get("max_tokens", 256) or 256)
            min_spec_tokens = max(1, int(os.environ.get("KORITH_SPEC_MIN_OUTPUT_TOKENS", "128") or 128))
            if max_tokens < min_spec_tokens:
                spec_requested = False
                spec_reason = "min_output_tokens_gate"
        if spec_requested:
            spec_require_distinct = _env_bool("KORITH_SPEC_REQUIRE_DISTINCT_DRAFT", True)
            spec_cache_only = _env_bool("KORITH_SPEC_CACHE_ONLY", False)
            spec_auto_cache_only = _env_bool("KORITH_SPEC_AUTO_CACHE_ONLY", False)
            if spec_require_distinct and not spec_cache_only and not spec_auto_cache_only:
                verify_model = str((jobspec.get("model", {}) or {}).get("model_path", "")).strip()
                draft_model = str(
                    spec_cfg.get("draft_model")
                    or os.environ.get("KORITH_DRAFT_MODEL_PATH", "")
                    or os.environ.get("KORITH_DRAFT_MODEL", "")
                ).strip()
                if not draft_model:
                    spec_requested = False
                    spec_reason = "draft_model_missing"
                elif verify_model:
                    try:
                        if Path(draft_model).resolve() == Path(verify_model).resolve():
                            spec_requested = False
                            spec_reason = "draft_same_as_verify"
                        else:
                            ratio_limit = max(0.1, float(os.environ.get("KORITH_SPEC_MAX_DRAFT_SIZE_RATIO", "0.75") or 0.75))
                            v_size = Path(verify_model).stat().st_size
                            d_size = Path(draft_model).stat().st_size
                            if v_size > 0 and (float(d_size) / float(v_size)) > ratio_limit:
                                spec_requested = False
                                spec_reason = "draft_too_large"
                    except Exception:
                        # Keep routing safe if model files are unavailable to router.
                        spec_requested = False
                        spec_reason = "draft_stat_error"
        spec_capable = bool(getattr(caps, "draft_supported", False) and (getattr(caps, "verify_tokens", False) or getattr(caps, "logits_access", False)))
        if not spec_capable:
            spec_reason = "capability_missing"
        if spec_requested and spec_capable and not spec_ignore_governance:
            spec_state = self.ledger.get_spec_governance(fingerprint_hash)
            now = time.time()
            if spec_state:
                if int(spec_state.get("spec_disabled", 0) or 0):
                    spec_requested = False
                    spec_reason = "governance_disabled"
                elif float(spec_state.get("cooldown_until", 0.0) or 0.0) > now:
                    spec_requested = False
                    spec_reason = "governance_cooldown"
        if not (caps.kv_replay and caps.deterministic_seeding and policy.get("allow_amf_reuse", True)):
            lane = "SPEC_MISS" if (spec_requested and spec_capable) else "MISS"
            if lane == "SPEC_MISS":
                return _finalize_lane(lane, "spec_enabled")
            return _finalize_lane(lane, spec_reason)
        gov = self.ledger.get_replay_governance(fingerprint_hash)
        now = time.time()
        if gov:
            if int(gov.get("replay_disabled", 0)):
                lane = "SPEC_MISS" if (spec_requested and spec_capable) else "MISS"
                if lane == "SPEC_MISS":
                    return _finalize_lane(lane, "spec_enabled")
                return _finalize_lane(lane, "replay_governance_disabled")
            if float(gov.get("cooldown_until", 0.0) or 0.0) > now:
                lane = "SPEC_MISS" if (spec_requested and spec_capable) else "MISS"
                if lane == "SPEC_MISS":
                    return _finalize_lane(lane, "spec_enabled")
                return _finalize_lane(lane, "replay_governance_cooldown")
        prior = self.ledger.find_job_by_fingerprint(fingerprint_hash, org_id=org_id)
        if prior:
            if self._router_require_ready_snapshot_for_hit():
                if not self._has_ready_snapshot_for_hit(fingerprint_hash=fingerprint_hash, org_id=org_id, prior_job_id=prior):
                    lane = "SPEC_MISS" if (spec_requested and spec_capable) else "MISS"
                    if lane == "SPEC_MISS":
                        return _finalize_lane(lane, "replay_snapshot_unavailable")
                    return _finalize_lane(lane, "replay_snapshot_unavailable")
            lane = "SPEC_HIT" if (spec_requested and spec_capable) else "HIT"
            if lane == "SPEC_HIT":
                return _finalize_lane(lane, "spec_enabled")
            return _finalize_lane(lane, spec_reason)
        lane = "SPEC_MISS" if (spec_requested and spec_capable) else "MISS"
        if lane == "SPEC_MISS":
            return _finalize_lane(lane, "spec_enabled")
        return _finalize_lane(lane, spec_reason)

    def _select_worker_single_node(
        self,
        jobspec: Dict[str, Any],
        workers: Optional[list] = None,
        fingerprint_hash: str = "",
        org_id: str = "default",
        predicted_lane: str = "MISS",
        shape_key: str = "",
    ) -> Tuple[Dict[str, Any], str]:
        workers = workers or self.registry.list_workers()
        if not workers:
            raise RuntimeError("no workers registered")
        session_id = jobspec.get("session_id")
        if session_id and session_id in self._session_affinity:
            wid = self._session_affinity[session_id].get("worker_id", "")
            for w in workers:
                if w["worker_id"] == wid:
                    return w, "worker_affinity"
        if predicted_lane in ("HIT", "SPEC_HIT") and self._router_bool("KORITH_ROUTER_ENABLE_FINGERPRINT_AFFINITY", True):
            key = f"{org_id}:{fingerprint_hash}"
            affinity = self._fingerprint_affinity.get(key, {})
            wid = str(affinity.get("worker_id", "") or "")
            if wid:
                for w in workers:
                    if str(w.get("worker_id", "") or "") == wid:
                        return w, "fingerprint_affinity"
        if (
            predicted_lane in ("MISS", "SPEC_MISS")
            and shape_key
            and self._router_bool("KORITH_ROUTER_ENABLE_SHAPE_AFFINITY", True)
        ):
            affinity = self._shape_affinity.get(shape_key, {})
            wid = str(affinity.get("worker_id", "") or "")
            if wid:
                for w in workers:
                    if str(w.get("worker_id", "") or "") != wid:
                        continue
                    if self._shape_affinity_worker_is_overloaded(w, workers):
                        break
                    return w, "shape_affinity"
        workers_sorted = sorted(workers, key=lambda w: (w.get("inflight", 0), w.get("last_heartbeat", 0)))
        worker = workers_sorted[0] if workers_sorted else workers[self._rr % len(workers)]
        self._rr += 1
        if session_id:
            self._session_affinity[session_id] = {"node_id": "", "worker_id": worker["worker_id"], "ts": str(time.time())}
        return worker, "least_loaded"

    def _router_bool(self, name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in _TRUTHY

    def _router_transfer_allowed_tiers(self) -> set[str]:
        raw = str(os.environ.get("KORITH_ROUTER_TRANSFER_ALLOWED_TIERS", "vram,ram,nvme,unknown") or "")
        tiers = {part.strip().lower() for part in raw.split(",") if part.strip()}
        if not tiers:
            return {"vram", "ram", "nvme", "unknown"}
        if "all" in tiers or "*" in tiers:
            return {"*"}
        return tiers

    def _router_transfer_tier_factor(self, tier: str) -> float:
        tier_key = str(tier or "unknown").strip().lower()
        env_name = {
            "vram": "KORITH_ROUTER_TRANSFER_TIER_FACTOR_VRAM",
            "ram": "KORITH_ROUTER_TRANSFER_TIER_FACTOR_RAM",
            "nvme": "KORITH_ROUTER_TRANSFER_TIER_FACTOR_NVME",
            "unknown": "KORITH_ROUTER_TRANSFER_TIER_FACTOR_UNKNOWN",
        }.get(tier_key, "KORITH_ROUTER_TRANSFER_TIER_FACTOR_UNKNOWN")
        defaults = {
            "vram": 1.0,
            "ram": 1.15,
            "nvme": 1.35,
            "unknown": 1.25,
        }
        default = defaults.get(tier_key, defaults["unknown"])
        try:
            value = float(os.environ.get(env_name, str(default)) or default)
        except Exception:
            value = default
        return max(0.1, value)

    def _pick_transfer_candidate(
        self,
        *,
        locations: list,
        chosen_node: str,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        allowed_tiers = self._router_transfer_allowed_tiers()
        best_location: Optional[Dict[str, Any]] = None
        best_transfer_ms = 0.0
        for location in locations:
            node_id = str(location.get("node_id") or "")
            if not node_id or node_id == chosen_node:
                continue
            tier = str(location.get("storage_tier", "unknown") or "unknown").strip().lower()
            if "*" not in allowed_tiers and tier not in allowed_tiers:
                continue
            size_bytes = int(location.get("size_bytes") or 0)
            transfer_ms = estimate_transfer_ms(
                size_bytes,
                self.transfer_bandwidth_mbps,
                self.transfer_rtt_ms,
            ) * self._router_transfer_tier_factor(tier)
            if best_location is None or transfer_ms < best_transfer_ms:
                best_location = location
                best_transfer_ms = transfer_ms
        return best_location, float(best_transfer_ms)

    def _router_require_ready_snapshot_for_hit(self) -> bool:
        return self._router_bool("KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT", False)

    def _has_ready_snapshot_for_hit(self, *, fingerprint_hash: str, org_id: str, prior_job_id: str = "") -> bool:
        if self.snapshot_index is not None:
            try:
                locations = self.snapshot_index.get_locations(fingerprint_hash, org_id)
                for row in locations:
                    path = Path(str(row.get("snapshot_path", "") or ""))
                    if path.exists():
                        return True
            except Exception:
                pass

        try:
            local_path = self.restore_store.get(fingerprint_hash)
            if local_path and Path(local_path).exists():
                return True
        except Exception:
            pass

        if prior_job_id:
            try:
                snap = self.ledger.get_snapshot_for_job(prior_job_id)
            except Exception:
                snap = None
            if isinstance(snap, dict):
                snap_path = str(snap.get("snapshot_path", "") or "")
                if snap_path and Path(snap_path).exists():
                    return True

        return False

    def _router_affinity_ttl_s(self) -> float:
        raw = os.environ.get("KORITH_ROUTER_AFFINITY_TTL_S", "900")
        try:
            return max(0.0, float(raw))
        except Exception:
            return 900.0

    def _router_fingerprint_affinity_max_entries(self) -> int:
        raw = os.environ.get("KORITH_ROUTER_FINGERPRINT_AFFINITY_MAX_ENTRIES", "4096")
        try:
            return max(64, int(raw))
        except Exception:
            return 4096

    def _router_shape_affinity_max_entries(self) -> int:
        raw = os.environ.get("KORITH_ROUTER_SHAPE_AFFINITY_MAX_ENTRIES", "8192")
        try:
            return max(128, int(raw))
        except Exception:
            return 8192

    def _router_shape_affinity_max_inflight_delta(self) -> int:
        raw = os.environ.get("KORITH_ROUTER_SHAPE_AFFINITY_MAX_INFLIGHT_DELTA", "2")
        try:
            return max(0, int(raw))
        except Exception:
            return 2

    def _routing_shape_key(self, *, jobspec: Dict[str, Any], prompt_tokens: int = 0) -> str:
        model = jobspec.get("model", {}) if isinstance(jobspec.get("model", {}), dict) else {}
        det = jobspec.get("deterministic_cfg", {}) if isinstance(jobspec.get("deterministic_cfg", {}), dict) else {}
        model_id = str(model.get("model_id", "") or "")
        n_batch = max(0, int(det.get("n_batch", 0) or 0))
        n_ctx = _bucket_ceil(int(det.get("n_ctx", 0) or 0), 1024)
        max_tokens = _bucket_ceil(int(det.get("max_tokens", 0) or 0), 64)
        prompt_bucket = _bucket_ceil(int(prompt_tokens or jobspec.get("prompt_tokens", 0) or 0), 64)
        return "|".join(
            [
                model_id,
                str(n_batch),
                str(n_ctx),
                str(max_tokens),
                str(prompt_bucket),
            ]
        )

    def _least_loaded_node(self, nodes: list, predicted_lane: str) -> Dict[str, Any]:
        if predicted_lane in ("HIT", "SPEC_HIT"):
            nodes_sorted = sorted(
                nodes,
                key=lambda n: (
                    int(n.get("queue_depth_hit", 0)),
                    int(n.get("inflight", 0)),
                    int(n.get("queue_depth_miss", 0)),
                ),
            )
        else:
            nodes_sorted = sorted(
                nodes,
                key=lambda n: (
                    int(n.get("inflight", 0)),
                    int(n.get("queue_depth_miss", 0)),
                ),
            )
        return nodes_sorted[0]

    def _shape_affinity_is_overloaded(self, node_id: str, nodes: list) -> bool:
        if not node_id:
            return False
        candidate = None
        inflights = []
        for row in nodes:
            try:
                inflight = int(row.get("inflight", 0) or 0)
            except Exception:
                inflight = 0
            inflights.append(inflight)
            if str(row.get("node_id", "") or "") == node_id:
                candidate = inflight
        if candidate is None or not inflights:
            return False
        best = min(inflights)
        max_delta = self._router_shape_affinity_max_inflight_delta()
        return (candidate - best) > max_delta

    def _shape_affinity_worker_is_overloaded(self, worker: Dict[str, Any], workers: list) -> bool:
        try:
            candidate = int(worker.get("inflight", 0) or 0)
        except Exception:
            candidate = 0
        inflights = []
        for row in workers:
            try:
                inflights.append(int(row.get("inflight", 0) or 0))
            except Exception:
                continue
        if not inflights:
            return False
        best = min(inflights)
        max_delta = self._router_shape_affinity_max_inflight_delta()
        return (candidate - best) > max_delta

    def _resolve_worker_id(self, worker_id: str, worker_ids: set[str]) -> str:
        if not worker_id:
            return ""
        if worker_ids and worker_id not in worker_ids:
            return ""
        return worker_id

    def _pick_worker_for_node(self, node_id: str, workers: list) -> str:
        if not node_id:
            return ""
        node_workers = [w for w in workers if str(w.get("node_id", "") or "") == node_id]
        if not node_workers:
            return ""
        ranked = sorted(node_workers, key=lambda w: (int(w.get("inflight", 0)), -float(w.get("last_heartbeat", 0.0) or 0.0)))
        return str(ranked[0].get("worker_id", "") or "")

    def _prune_affinity_state(self, worker_ids: set[str], node_ids: set[str]) -> None:
        now = time.time()
        ttl_s = self._router_affinity_ttl_s()

        def _entry_valid(entry: Dict[str, Any]) -> bool:
            ts = float(entry.get("ts", 0.0) or 0.0)
            if ttl_s <= 0.0:
                return False
            if ts > 0.0 and (now - ts) > ttl_s:
                return False
            worker_id = str(entry.get("worker_id", "") or "")
            if worker_id and worker_ids and worker_id not in worker_ids:
                return False
            node_id = str(entry.get("node_id", "") or "")
            if node_id and node_ids and node_id not in node_ids:
                return False
            return True

        for key, entry in list(self._session_affinity.items()):
            if not isinstance(entry, dict) or not _entry_valid(entry):
                self._session_affinity.pop(key, None)
        for key, entry in list(self._fingerprint_affinity.items()):
            if not isinstance(entry, dict) or not _entry_valid(entry):
                self._fingerprint_affinity.pop(key, None)
        for key, entry in list(self._shape_affinity.items()):
            if not isinstance(entry, dict) or not _entry_valid(entry):
                self._shape_affinity.pop(key, None)

    def _remember_fingerprint_affinity(self, key: str, node_id: str, worker_id: str) -> None:
        self._fingerprint_affinity[key] = {
            "node_id": str(node_id or ""),
            "worker_id": str(worker_id or ""),
            "ts": str(time.time()),
        }
        max_entries = self._router_fingerprint_affinity_max_entries()
        if len(self._fingerprint_affinity) <= max_entries:
            return
        ordered = sorted(
            self._fingerprint_affinity.items(),
            key=lambda kv: float((kv[1] or {}).get("ts", 0.0) or 0.0),
        )
        drop = len(self._fingerprint_affinity) - max_entries
        for idx in range(drop):
            self._fingerprint_affinity.pop(ordered[idx][0], None)

    def _remember_shape_affinity(self, key: str, node_id: str, worker_id: str) -> None:
        if not key:
            return
        self._shape_affinity[key] = {
            "node_id": str(node_id or ""),
            "worker_id": str(worker_id or ""),
            "ts": str(time.time()),
        }
        max_entries = self._router_shape_affinity_max_entries()
        if len(self._shape_affinity) <= max_entries:
            return
        ordered = sorted(
            self._shape_affinity.items(),
            key=lambda kv: float((kv[1] or {}).get("ts", 0.0) or 0.0),
        )
        drop = len(self._shape_affinity) - max_entries
        for idx in range(drop):
            self._shape_affinity.pop(ordered[idx][0], None)

    def _append_event(self, events_path, event_type: str, payload: Dict[str, Any]) -> None:
        evt = {"type": event_type, "ts": utc_now(), "payload": payload}
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
