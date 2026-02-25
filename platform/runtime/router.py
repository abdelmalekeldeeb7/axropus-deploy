from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..artifacts.store import ArtifactStore
from ..ledger import db as ledger
from .prompt_canonicalization import canonicalize_prompt_text, canonicalize_template_data


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_str(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class Router:
    def __init__(
        self,
        db_path: Path,
        artifacts: ArtifactStore,
        workers: List,
        adapter_registry,
        restore_store,
    ) -> None:
        self.db_path = db_path
        self.artifacts = artifacts
        self.workers = workers
        self.adapter_registry = adapter_registry
        self.restore_store = restore_store
        self.session_affinity: Dict[str, int] = {}
        self._rr = 0

    def submit(self, jobspec: Dict[str, Any]) -> str:
        self._validate_jobspec(jobspec)
        idempotency_key = jobspec.get("idempotency_key")
        if idempotency_key:
            existing = ledger.find_job_by_idempotency(self.db_path, idempotency_key)
            if existing:
                return existing
        job_id = jobspec.get("job_id") or str(uuid.uuid4())
        created_at = utc_now()
        prompt = self._render_prompt(jobspec)
        prompt_hash = sha256_str(prompt)
        sampling_hash = sha256_str(json.dumps(jobspec.get("deterministic_cfg", {}), sort_keys=True))

        adapter = self.adapter_registry.get_adapter(jobspec)
        prompt_tokens = int(adapter.tokenize(prompt) or 0)
        self._validate_context_budget(jobspec, prompt, prompt_tokens)
        fingerprint = self._fingerprint(jobspec, adapter)
        fingerprint_hash = sha256_str(json.dumps(fingerprint, sort_keys=True))

        lane = self._select_lane(jobspec, adapter, fingerprint_hash)

        job = {
            "job_id": job_id,
            "created_at": created_at,
            "backend_id": jobspec["backend_id"],
            "backend_version": adapter.backend_version,
            "model": jobspec["model"],
            "prompt_rendered": prompt,
            "prompt_hash": prompt_hash,
            "sampling_hash": sampling_hash,
            "prompt_tokens": prompt_tokens,
            "fingerprint": fingerprint,
            "fingerprint_hash": fingerprint_hash,
            "deterministic_cfg": jobspec.get("deterministic_cfg", {}),
            "policy": jobspec.get("policy", {}),
            "parent_job_id": jobspec.get("parent_job_id"),
            "session_id": jobspec.get("session_id"),
        }

        ledger.insert_job(
            self.db_path,
            job_id=job_id,
            created_at=created_at,
            jobspec=jobspec,
            fingerprint=fingerprint,
            prompt_hash=prompt_hash,
            fingerprint_hash=fingerprint_hash,
            idempotency_key=jobspec.get("idempotency_key"),
            status="QUEUED",
        )

        artifacts = self.artifacts.init_job(job_id)
        self._append_event(artifacts["events"], "ROUTER_ACCEPT", {"job_id": job_id})

        worker = self._select_worker(job)

        self._append_event(
            artifacts["events"],
            "ROUTER_ASSIGN",
            {"job_id": job_id, "worker_id": worker.worker_id, "gpu_id": worker.gpu_id, "lane": lane},
        )
        worker.enqueue(job, lane, time.time())
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        rec = ledger.get_job(self.db_path, job_id)
        if not rec:
            return None
        latest = ledger.get_latest_run(self.db_path, job_id)
        if latest:
            rec.update(latest)
        return rec

    def list_jobs(self, limit: int = 20):
        return ledger.list_jobs(self.db_path, limit=limit)

    def restore(self, job_id: str) -> Dict[str, Any]:
        rec = ledger.get_job(self.db_path, job_id)
        if not rec:
            return {"restored": False, "reason": "job_not_found"}
        snap = ledger.get_snapshot_for_job(self.db_path, job_id)
        if not snap:
            return {"restored": False, "reason": "snapshot_not_found"}
        if snap["fingerprint_hash"] != rec.get("fingerprint_hash"):
            artifacts = self.artifacts.init_job(job_id)
            self._append_event(
                artifacts["events"],
                "MF_RESTORE_FAILED",
                {"job_id": job_id, "reason": "fingerprint_mismatch"},
            )
            return {"restored": False, "reason": "fingerprint_mismatch"}
        self.restore_store.set(snap["fingerprint_hash"], snap["snapshot_path"])
        artifacts = self.artifacts.init_job(job_id)
        self._append_event(
            artifacts["events"],
            "MF_RESTORE",
            {"job_id": job_id, "snapshot_path": snap["snapshot_path"]},
        )
        return {"restored": True, "snapshot_path": snap["snapshot_path"]}

    def _render_prompt(self, jobspec: Dict[str, Any]) -> str:
        if "prompt" in jobspec and jobspec["prompt"]:
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

    def _fingerprint(self, jobspec: Dict[str, Any], adapter) -> Dict[str, Any]:
        fp = adapter.get_fingerprint()
        cfg = jobspec.get("deterministic_cfg", {})
        fp.update({
            "backend_id": jobspec["backend_id"],
            "model_id": jobspec["model"]["model_id"],
            "model_path": jobspec["model"].get("model_path", ""),
            "endpoint": jobspec["model"].get("endpoint", ""),
            "n_ctx": cfg.get("n_ctx", 0),
            "n_batch": cfg.get("n_batch", 0),
            "sampling_hash": sha256_str(json.dumps(cfg, sort_keys=True)),
        })
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

    def _select_lane(self, jobspec: Dict[str, Any], adapter, fingerprint_hash: str) -> str:
        caps = adapter.get_capabilities()
        policy = jobspec.get("policy", {})
        if not (caps.kv_replay and caps.deterministic_seeding and policy.get("allow_amf_reuse", True)):
            return "MISS"
        prior = ledger.find_job_by_fingerprint(self.db_path, fingerprint_hash)
        return "HIT" if prior else "MISS"

    def _select_worker(self, job: Dict[str, Any]):
        session_id = job.get("session_id")
        if session_id and session_id in self.session_affinity:
            return self.workers[self.session_affinity[session_id]]
        # Round-robin
        idx = self._rr % len(self.workers)
        self._rr += 1
        if session_id:
            self.session_affinity[session_id] = idx
        return self.workers[idx]

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
        return max(prompt_tokens, char_estimate, mult_estimate)

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
        raise ValueError(
            "CONTEXT_OVERFLOW "
            f"required_ctx={required_ctx} available_ctx={n_ctx} prompt_tokens_est={prompt_tokens_est} "
            f"prompt_tokens={max(1, int(prompt_tokens))} max_tokens={max_tokens} reserve_tokens={reserve_tokens}"
        )

    def _append_event(self, events_path: Path, event_type: str, payload: Dict[str, Any]) -> None:
        evt = {"type": event_type, "ts": utc_now(), "payload": payload}
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
