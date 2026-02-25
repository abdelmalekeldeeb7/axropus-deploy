from __future__ import annotations

import importlib.util
import json
import sysconfig
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..artifacts.store import ArtifactStore
from ..ledger import db as ledger


def _load_stdlib_queue_module():
    try:
        import queue as candidate  # type: ignore
        if hasattr(candidate, "Queue") and hasattr(candidate, "Empty"):
            return candidate
    except Exception:
        pass
    stdlib_path = Path(sysconfig.get_path("stdlib")) / "queue.py"
    spec = importlib.util.spec_from_file_location("_korith_stdlib_queue", stdlib_path)
    if not spec or not spec.loader:
        raise RuntimeError("failed to load stdlib queue module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = _load_stdlib_queue_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Worker:
    def __init__(
        self,
        worker_id: str,
        gpu_id: int,
        db_path: Path,
        artifacts: ArtifactStore,
        adapter_factory,
        restore_store,
    ) -> None:
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.db_path = db_path
        self.artifacts = artifacts
        self.adapter_factory = adapter_factory
        self.restore_store = restore_store

        self.q_hit: queue.Queue = queue.Queue()
        self.q_miss: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def enqueue(self, job: Dict[str, Any], lane: str, enqueued_at: float) -> None:
        payload = {"job": job, "lane": lane, "enqueued_at": enqueued_at}
        if lane == "HIT":
            self.q_hit.put(payload)
        else:
            self.q_miss.put(payload)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.q_hit.empty():
                    item = self.q_hit.get(timeout=0.2)
                else:
                    item = self.q_miss.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process(item["job"], item["lane"], item["enqueued_at"])

    def _process(self, job: Dict[str, Any], lane: str, enqueued_at: float) -> None:
        job_id = job["job_id"]
        artifacts = self.artifacts.init_job(job_id)
        events_path = artifacts["events"]

        queue_latency_ms = (time.time() - enqueued_at) * 1000.0

        self._append_event(events_path, "WORKER_START", {
            "job_id": job_id,
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "lane": lane,
            "queue_latency_ms": queue_latency_ms,
        })
        ledger.update_status(self.db_path, job_id, "RUNNING")

        adapter = self.adapter_factory(job)
        caps = adapter.get_capabilities()
        deterministic_cfg = job.get("deterministic_cfg", {})
        max_tokens = int(deterministic_cfg.get("max_tokens", 256) or 256)
        prompt = job.get("prompt_rendered", "")
        policy = job.get("policy", {})
        if not (caps.kv_replay and caps.deterministic_seeding):
            policy = dict(policy)
            policy["allow_amf_reuse"] = False

        fingerprint_hash = job.get("fingerprint_hash")
        mf_snapshot_in = self.restore_store.get(fingerprint_hash)

        started_at = utc_now()
        try:
            if policy.get("allow_amf_reuse", True) and caps.kv_replay and caps.deterministic_seeding:
                self._append_event(events_path, "AMF_LOOKUP", {"job_id": job_id})
            else:
                reason = "policy_blocked" if not policy.get("allow_amf_reuse", True) else "unavailable"
                self._append_event(events_path, "AMF_BLOCK", {"job_id": job_id, "reason": reason})

            result = adapter.run_baseline(
                prompt=prompt,
                max_tokens=max_tokens,
                deterministic_cfg=deterministic_cfg,
                policy=policy,
                artifacts={k: str(v) for k, v in artifacts.items()},
                mf_snapshot_in=mf_snapshot_in,
            )
        except Exception as exc:
            finished_at = utc_now()
            self._append_event(events_path, "ERROR", {"job_id": job_id, "error": str(exc)})
            ledger.update_status(self.db_path, job_id, "FAILED")
            return
        finished_at = utc_now()

        # Combine metrics from adapter/engine
        metrics = self._compose_metrics(job, result, lane, queue_latency_ms, started_at, finished_at)
        Path(artifacts["metrics"]).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        engine_metrics = result.get("engine_metrics") or {}
        # Append engine events if provided
        engine_events_path = result.get("engine_events_path")
        if engine_events_path and Path(engine_events_path).exists():
            with Path(engine_events_path).open("r", encoding="utf-8") as f, Path(events_path).open("a", encoding="utf-8") as out:
                for line in f:
                    out.write(line)
        else:
            # Baseline-only adapter: emit MF apply/snapshot for auditability.
            self._append_event(events_path, "MF_APPLY", {"job_id": job_id, "supported": False})
        cp = engine_metrics.get("cp") if isinstance(engine_metrics, dict) else None
        if cp:
            self._append_event(events_path, "CP_DECISION", {"job_id": job_id, **cp})
        else:
            self._append_event(events_path, "CP_DECISION", {"job_id": job_id, "reason": "unavailable"})
        amf_decision = metrics.get("amf", {}).get("decision")
        if amf_decision in ("hit", "miss", "blocked"):
            self._append_event(events_path, f"AMF_{amf_decision.upper()}", {"job_id": job_id})

        self._append_event(events_path, "WORKER_END", {
            "job_id": job_id,
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "exit_code": result.get("exit_code", 0),
        })

        run_id = str(uuid.uuid4())
        ledger.insert_run(
            self.db_path,
            run_id,
            job_id,
            started_at,
            finished_at,
            int(result.get("exit_code", 0)),
            str(artifacts["metrics"]),
            str(artifacts["events"]),
            str(artifacts["output"]),
            str(artifacts["log"]),
        )

        if Path(artifacts["mf_snapshot"]).exists():
            if not engine_events_path or not Path(engine_events_path).exists():
                self._append_event(events_path, "MF_SNAPSHOT", {"job_id": job_id, "path": str(artifacts["mf_snapshot"])})
            ledger.insert_snapshot(
                self.db_path,
                snapshot_id=job_id,
                job_id=job_id,
                fingerprint_hash=fingerprint_hash,
                snapshot_path=str(artifacts["mf_snapshot"]),
                created_at=finished_at,
            )

        status = "SUCCEEDED" if int(result.get("exit_code", 0)) == 0 else "FAILED"
        ledger.update_status(self.db_path, job_id, status)

    def _compose_metrics(self, job: Dict[str, Any], result: Dict[str, Any], lane: str, queue_latency_ms: float,
                         started_at: str, finished_at: str) -> Dict[str, Any]:
        engine_metrics = result.get("engine_metrics") or {}
        amf = engine_metrics.get("amf", {})
        mf = engine_metrics.get("mf", {})
        perf = engine_metrics.get("perf", {})

        ids = {
            "job_id": job["job_id"],
            "parent_job_id": job.get("parent_job_id"),
            "created_at": job.get("created_at"),
            "started_at": started_at,
            "finished_at": finished_at,
        }
        backend = {
            "backend_id": job["backend_id"],
            "backend_version": job.get("backend_version", "v1"),
        }
        model = {
            "model_id": job["model"]["model_id"],
            "model_hash": job["fingerprint"].get("model_hash", ""),
            "tokenizer_hash": job["fingerprint"].get("tokenizer_hash", ""),
        }
        input_block = {
            "prompt_hash": job["prompt_hash"],
            "sampling_hash": job["sampling_hash"],
            "prompt_tokens": job.get("prompt_tokens", 0),
        }
        scheduling = {
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "lane": lane,
            "queue_latency_ms": queue_latency_ms,
        }
        perf_block = {
            "tokens_out": perf.get("tokens_out", 0),
            "total_ms": perf.get("total_ms", result.get("total_ms", 0.0)),
            "prefill_ms": perf.get("prefill_ms", 0.0),
            "decode_ms": perf.get("decode_ms", 0.0),
            "avg_tps": perf.get("avg_tps", 0.0),
        }
        amf_supported = bool(amf.get("supported", False))
        if "supported" not in amf:
            amf_supported = False
        mf_supported = bool(mf.get("supported", False))
        if "supported" not in mf:
            mf_supported = False

        snapshot_id = mf.get("snapshot_id", "")
        if not snapshot_id and job.get("job_id"):
            snapshot_id = job["job_id"]

        errors = engine_metrics.get("errors", [])
        if int(result.get("exit_code", 0)) != 0:
            errors = list(errors) if isinstance(errors, list) else []
            errors.append(f"exit_code:{result.get('exit_code')}")

        return {
            "ids": ids,
            "backend": backend,
            "model": model,
            "input": input_block,
            "scheduling": scheduling,
            "amf": {
                "supported": amf_supported,
                "decision": amf.get("decision", "unavailable" if not amf_supported else "miss"),
                "prefix_len": amf.get("prefix_len", 0),
                "skipped_tokens": amf.get("skipped_tokens", 0),
                "skip_ratio": amf.get("skip_ratio", 0.0),
                "restore_ms": amf.get("restore_ms", 0.0),
                "baseline_prefix_ms": amf.get("baseline_prefix_ms", 0.0),
                "saved_ms": amf.get("saved_ms", 0.0),
                "roi": amf.get("roi", 0.0),
            },
            "mf": {
                "supported": mf_supported,
                "min_admit_roi": mf.get("min_admit_roi", 0.0),
                "eviction_pressure": mf.get("eviction_pressure", 1.0),
                "replay_disable_mask": mf.get("replay_disable_mask", 0),
                "cooldown_ms": mf.get("cooldown_ms", 0),
                "snapshot_id": snapshot_id,
            },
            "perf": perf_block,
            "health": engine_metrics.get("health", {}),
            "errors": errors,
        }

    def _append_event(self, events_path: Path, event_type: str, payload: Dict[str, Any]) -> None:
        evt = {
            "type": event_type,
            "ts": utc_now(),
            "payload": payload,
        }
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
