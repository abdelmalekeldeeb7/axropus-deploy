#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import signal
import string
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


TRUTHY = {"1", "true", "yes", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    idx = max(0.0, min(1.0, p)) * (len(arr) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return arr[lo]
    frac = idx - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def http_json(
    method: str,
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    api_key: str = "",
    timeout_s: float = 20.0,
) -> Dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4())}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(url=url, method=method.upper(), data=data, headers=headers)
    with request.urlopen(req, timeout=float(timeout_s)) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def wait_http_ok(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + float(timeout_s)
    last_err = ""
    while time.time() < deadline:
        try:
            req = request.Request(url=url, method="GET")
            with request.urlopen(req, timeout=2.0) as resp:
                if 200 <= int(resp.status) < 300:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"timeout waiting for {url}: {last_err}")


def is_transient_http_error(exc: Exception) -> bool:
    if isinstance(exc, error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, error.HTTPError):
        try:
            code = int(getattr(exc, "code", 0) or 0)
        except Exception:
            code = 0
        return code in {408, 409, 425, 429, 500, 502, 503, 504}
    text = str(exc).lower()
    transient_markers = (
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "remote end closed connection",
        "bad gateway",
        "service unavailable",
    )
    return any(marker in text for marker in transient_markers)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


@dataclass
class ManagedProcess:
    name: str
    cmd: List[str]
    env: Dict[str, str]
    log_path: Path
    proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        ensure_dir(self.log_path.parent)
        with self.log_path.open("ab") as logf:
            self.proc = subprocess.Popen(  # noqa: S603
                self.cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=self.env,
                stdout=logf,
                stderr=logf,
            )

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, *, force: bool = False, wait_s: float = 8.0) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            return
        if force:
            self.proc.kill()
            self.proc.wait(timeout=max(1.0, wait_s))
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=max(1.0, wait_s))
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=max(1.0, wait_s))


class StackManager:
    def __init__(
        self,
        *,
        env: Dict[str, str],
        out_dir: Path,
        router_host: str,
        router_port: int,
        worker_host: str,
        worker_base_port: int,
        initial_workers: int,
        max_workers: int,
    ) -> None:
        self.env = dict(env)
        self.out_dir = out_dir
        self.router_host = router_host
        self.router_port = int(router_port)
        self.worker_host = worker_host
        self.worker_base_port = int(worker_base_port)
        self.initial_workers = max(1, int(initial_workers))
        self.max_workers = max(self.initial_workers, int(max_workers))
        self._lock = threading.Lock()
        self.router: Optional[ManagedProcess] = None
        self.workers: Dict[str, ManagedProcess] = {}
        self._next_worker_idx = self.initial_workers
        self._next_node_seq = 0

    @property
    def router_url(self) -> str:
        return f"http://{self.router_host}:{self.router_port}"

    def _next_node_id(self, prefix: str = "node") -> str:
        self._next_node_seq += 1
        return f"{prefix}-{self._next_node_seq}"

    def start_router(self) -> None:
        log_path = self.out_dir / "logs" / "router.log"
        cmd = [
            "python3",
            "platform/korith_platform.py",
            "router",
            "--host",
            self.router_host,
            "--port",
            str(self.router_port),
        ]
        proc = ManagedProcess(
            name="router",
            cmd=cmd,
            env=self.env,
            log_path=log_path,
        )
        proc.start()
        self.router = proc
        wait_http_ok(f"{self.router_url}/health", timeout_s=120.0)

    def start_worker(self, *, worker_name: str, port: int, node_id: str) -> None:
        log_path = self.out_dir / "logs" / f"{worker_name}.log"
        cmd = [
            "python3",
            "platform/korith_platform.py",
            "worker",
            "--worker-id",
            worker_name,
            "--gpu-id",
            "0",
            "--host",
            self.worker_host,
            "--port",
            str(port),
            "--node-id",
            node_id,
        ]
        proc = ManagedProcess(name=worker_name, cmd=cmd, env=self.env, log_path=log_path)
        proc.start()
        self.workers[worker_name] = proc
        wait_http_ok(f"http://{self.worker_host}:{port}/health", timeout_s=120.0)

    def start(self) -> None:
        with self._lock:
            self.start_router()
            for idx in range(self.initial_workers):
                worker_name = f"worker-{idx}"
                port = self.worker_base_port + idx
                node_id = self._next_node_id("node")
                self.start_worker(worker_name=worker_name, port=port, node_id=node_id)
            wait_http_ok(f"{self.router_url}/ready", timeout_s=120.0)

    def _stop_worker(self, worker_name: str, *, force: bool = False) -> None:
        proc = self.workers.get(worker_name)
        if proc is None:
            return
        proc.stop(force=force)

    def restart_router(self, *, force: bool = False) -> None:
        with self._lock:
            if self.router is not None:
                self.router.stop(force=force)
            self.start_router()
            wait_http_ok(f"{self.router_url}/ready", timeout_s=120.0)

    def restart_worker(self, worker_name: str, *, force: bool = False, node_churn: bool = True) -> None:
        with self._lock:
            proc = self.workers.get(worker_name)
            if proc is None:
                return
            port = self._port_for_worker(worker_name)
            proc.stop(force=force)
            node_id = self._next_node_id("node") if node_churn else f"stable-{worker_name}"
            self.start_worker(worker_name=worker_name, port=port, node_id=node_id)

    def _port_for_worker(self, worker_name: str) -> int:
        if worker_name.startswith("worker-"):
            tail = worker_name.split("worker-", 1)[1]
            if tail.isdigit():
                return self.worker_base_port + int(tail)
        if worker_name.startswith("worker-extra-"):
            tail = worker_name.split("worker-extra-", 1)[1]
            if tail.isdigit():
                return self.worker_base_port + 100 + int(tail)
        return self.worker_base_port

    def autoscale_up(self) -> Optional[str]:
        with self._lock:
            if len(self.workers) >= self.max_workers:
                return None
            idx = self._next_worker_idx
            self._next_worker_idx += 1
            worker_name = f"worker-extra-{idx}"
            port = self.worker_base_port + 100 + idx
            node_id = self._next_node_id("scale")
            self.start_worker(worker_name=worker_name, port=port, node_id=node_id)
            return worker_name

    def autoscale_down(self) -> Optional[str]:
        with self._lock:
            extras = sorted(name for name in self.workers if name.startswith("worker-extra-"))
            if not extras:
                return None
            victim = extras[-1]
            self._stop_worker(victim, force=False)
            self.workers.pop(victim, None)
            return victim

    def pick_worker_for_restart(self) -> Optional[str]:
        with self._lock:
            names = sorted(self.workers.keys())
        if not names:
            return None
        return random.choice(names)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            router_alive = bool(self.router and self.router.is_alive())
            workers = {name: proc.is_alive() for name, proc in self.workers.items()}
        return {
            "router_alive": router_alive,
            "workers_alive": workers,
            "workers_total": len(workers),
            "workers_up": sum(1 for _, alive in workers.items() if alive),
        }

    def stop_all(self) -> None:
        with self._lock:
            for name in list(self.workers.keys()):
                self._stop_worker(name, force=False)
            for name in list(self.workers.keys()):
                self.workers.pop(name, None)
            if self.router is not None:
                self.router.stop(force=False)


class WorkloadBuilder:
    def __init__(
        self,
        *,
        backend_id: str,
        model_id: str,
        model_path: str,
        model_endpoint: str,
        n_ctx: int,
        n_batch: int,
        max_tokens: int,
        allow_spec: bool,
        orgs: List[str],
        users_per_org: int,
        documents: int,
        prompt_tokens_max: int,
        short_context_rate: float,
        long_context_rate: float,
        mutation_rate: float,
        partial_overlap_rate: float,
        system_prompt: str,
        seed: int,
    ) -> None:
        self.backend_id = backend_id
        self.model_id = model_id
        self.model_path = model_path
        self.model_endpoint = model_endpoint
        self.n_ctx = int(n_ctx)
        self.n_batch = int(n_batch)
        self.max_tokens = int(max_tokens)
        self.allow_spec = bool(allow_spec)
        self.orgs = list(orgs)
        self.users_per_org = max(1, int(users_per_org))
        self.documents = max(1, int(documents))
        self.prompt_tokens_max = max(256, int(prompt_tokens_max))
        self.short_context_rate = float(short_context_rate)
        self.long_context_rate = float(long_context_rate)
        self.mutation_rate = float(mutation_rate)
        self.partial_overlap_rate = float(partial_overlap_rate)
        self.system_prompt = system_prompt
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._seq = 0

        max_prompt_by_ctx = max(256, self.n_ctx - self.max_tokens - 128)
        self.prompt_tokens_cap = min(self.prompt_tokens_max, max_prompt_by_ctx)

    def _pick_prompt_tokens(self) -> int:
        r = self._rng.random()
        if r < self.short_context_rate:
            return max(256, int(self.prompt_tokens_cap * 0.2))
        if r > 1.0 - self.long_context_rate:
            return max(512, int(self.prompt_tokens_cap * 0.9))
        return max(384, int(self.prompt_tokens_cap * 0.55))

    def _doc_text(self, doc_id: int, token_count: int) -> str:
        # Deterministic synthetic corpus for controlled overlap behavior.
        return " ".join(f"d{doc_id}_t{i % 257}" for i in range(max(1, token_count)))

    def _mutate_text(self, text: str) -> str:
        if self._rng.random() >= self.mutation_rate:
            return text
        variants = [
            text + "  ",
            text.replace(" ", "  ", 1),
            text + "\n",
            text.replace("\n", "\r\n"),
            " " + text,
        ]
        return self._rng.choice(variants)

    def next_jobspec(self) -> Dict[str, Any]:
        self._seq += 1
        org = self._rng.choice(self.orgs)
        user = f"user-{self._rng.randint(1, self.users_per_org)}"
        doc_id = self._rng.randint(0, self.documents - 1)
        session_id = f"{org}:doc-{doc_id}"

        target_tokens = self._pick_prompt_tokens()
        overlap_ratio = 1.0
        if self._rng.random() < self.partial_overlap_rate:
            overlap_ratio = self._rng.choice([0.35, 0.5, 0.65, 0.8, 0.9])
        prefix_tokens = max(32, int(target_tokens * overlap_ratio))

        doc_text = self._doc_text(doc_id, prefix_tokens)
        system_prompt = self._mutate_text(self.system_prompt)
        question = self._mutate_text(f"Q{self._seq % 97}: summarize key risks and obligations.")
        context = self._mutate_text(doc_text)

        model: Dict[str, Any] = {"model_id": self.model_id}
        if self.backend_id in ("korith_local", "korith_cuda", "hf_transformers"):
            model["model_path"] = self.model_path
        if self.backend_id in ("vllm", "vllm_openai", "openai_compatible"):
            model["endpoint"] = self.model_endpoint

        jobspec: Dict[str, Any] = {
            "schema_version": "korith.jobspec.v1",
            "backend_id": self.backend_id,
            "model": model,
            "prompt_template": "SYSTEM:\n{system}\n\nDOCUMENT:\n{context}\n\nQUESTION:\n{question}\n",
            "input": {
                "system": system_prompt,
                "context": context,
                "question": question,
            },
            "deterministic_cfg": {
                "seed": 42,
                "n_ctx": self.n_ctx,
                "n_batch": self.n_batch,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
            },
            "policy": {
                "allow_amf_reuse": True,
                "allow_spec": self.allow_spec,
            },
            "owner": {
                "org_id": org,
                "user_id": user,
            },
            "session_id": session_id,
        }
        return jobspec


class SoakRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(__file__).resolve().parents[1]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.out_dir = ensure_dir((self.root / args.output_root / stamp).resolve())
        ensure_dir(self.out_dir / "logs")
        ensure_dir(self.out_dir / "data")
        ensure_dir(self.out_dir / "report")
        ensure_dir(self.out_dir / "runtime")
        self.results_path = self.out_dir / "data" / "results.jsonl"
        self.system_path = self.out_dir / "data" / "system_metrics.csv"
        self.chaos_path = self.out_dir / "data" / "chaos_events.jsonl"
        self.meta_path = self.out_dir / "run_config.json"

        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._submit_errors = Counter()
        self._active_threads: List[threading.Thread] = []
        self._disk_pressure_file: Optional[Path] = None
        self._results_written = 0

        self.env = self._build_runtime_env()
        self.stack = StackManager(
            env=self.env,
            out_dir=self.out_dir,
            router_host=args.router_host,
            router_port=args.router_port,
            worker_host=args.worker_host,
            worker_base_port=args.worker_base_port,
            initial_workers=args.initial_workers,
            max_workers=args.max_workers,
        )
        self.api_key = args.api_key.strip()
        self.base_url = f"http://{args.router_host}:{args.router_port}"
        self.end_time = 0.0

        self.workload = WorkloadBuilder(
            backend_id=args.backend_id,
            model_id=args.model_id,
            model_path=args.model_path,
            model_endpoint=args.model_endpoint,
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            max_tokens=args.max_tokens,
            allow_spec=args.allow_spec,
            orgs=[x.strip() for x in args.orgs.split(",") if x.strip()],
            users_per_org=args.users_per_org,
            documents=args.documents,
            prompt_tokens_max=args.prompt_tokens_max,
            short_context_rate=args.short_context_rate,
            long_context_rate=args.long_context_rate,
            mutation_rate=args.mutation_rate,
            partial_overlap_rate=args.partial_overlap_rate,
            system_prompt=args.system_prompt,
            seed=args.seed,
        )

    def _build_runtime_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        runtime = ensure_dir(self.out_dir / "runtime")
        env.setdefault("KORITH_QUEUE_BACKEND", "sqlite")
        env["KORITH_PLATFORM_DB"] = str(runtime / "ledger.sqlite")
        env["KORITH_PLATFORM_ARTIFACTS"] = str(runtime / "artifacts")
        env["KORITH_QUEUE_DB"] = str(runtime / "queue.sqlite")
        env["KORITH_REGISTRY_DB"] = str(runtime / "registry.sqlite")
        env["KORITH_RESTORE_DB"] = str(runtime / "restore.sqlite")
        env["KORITH_NODE_REGISTRY_DB"] = str(runtime / "node_registry.sqlite")
        if not env.get("KORITH_API_KEY_SALT", "").strip():
            suffix = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(24))
            env["KORITH_API_KEY_SALT"] = f"soak_salt_{suffix}"
        return env

    def _validate_args(self) -> None:
        if self.args.backend_id in ("korith_local", "korith_cuda", "hf_transformers"):
            if not self.args.model_path:
                raise ValueError("--model-path is required for local backends")
        if self.args.backend_id in ("vllm", "vllm_openai", "openai_compatible"):
            if not self.args.model_endpoint:
                raise ValueError("--model-endpoint is required for endpoint backends")
        if not [x.strip() for x in self.args.orgs.split(",") if x.strip()]:
            raise ValueError("--orgs must include at least one tenant")

    def _write_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        reached_target = False
        with self._write_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if path == self.results_path:
                self._results_written += 1
                target = int(getattr(self.args, "target_requests", 0) or 0)
                if target > 0 and self._results_written >= target:
                    reached_target = True
        if reached_target:
            self._stop.set()

    def _create_api_key(self) -> str:
        key_id = f"soak-{uuid.uuid4().hex[:10]}"
        cmd = [
            "python3",
            "platform/korith_platform.py",
            "keys",
            "create",
            "--org",
            "soak",
            "--key-id",
            key_id,
            "--rate-limit-tpm",
            "5000000",
            "--rate-limit-rpm",
            "500000",
        ]
        proc = subprocess.run(cmd, cwd=str(self.root), env=self.env, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            raise RuntimeError(f"failed to create api key: {proc.stderr.strip() or proc.stdout.strip()}")
        payload = json.loads(proc.stdout)
        key = str(payload.get("api_key", "")).strip()
        if not key:
            raise RuntimeError("failed to parse api key from key-create output")
        return key

    def _start_stack_if_needed(self) -> None:
        if not self.args.start_stack:
            return
        try:
            self.stack.start()
        except Exception:
            self.stack.stop_all()
            raise
        if not self.api_key:
            self.api_key = self._create_api_key()
        wait_http_ok(f"{self.base_url}/health", timeout_s=120.0)
        wait_http_ok(f"{self.base_url}/ready", timeout_s=120.0)

    def _submit_one(self, thread_id: int) -> None:
        per_thread_rps = max(1e-6, float(self.args.rps) / float(self.args.concurrency))
        interval_s = 1.0 / per_thread_rps
        submit_retries = max(0, int(self.args.submit_retry_max))
        status_retries = max(0, int(self.args.status_retry_max))
        retry_backoff_s = max(0.01, float(self.args.retry_backoff_s))
        retry_jitter_s = max(0.0, float(self.args.retry_jitter_s))
        while (not self._stop.is_set()) and time.time() < self.end_time:
            started = time.time()
            jobspec = self.workload.next_jobspec()
            owner = jobspec.get("owner", {}) if isinstance(jobspec.get("owner", {}), dict) else {}
            org_id = str(owner.get("org_id", "default") or "default")
            session_id = str(jobspec.get("session_id", "") or "")
            row: Dict[str, Any] = {
                "ts_submit": utc_now(),
                "thread_id": thread_id,
                "org_id": org_id,
                "session_id": session_id,
            }
            job_id = ""
            status_payload: Dict[str, Any] = {}
            metrics_payload: Dict[str, Any] = {}
            try:
                submit_res: Optional[Dict[str, Any]] = None
                submit_error: Optional[Exception] = None
                for attempt in range(submit_retries + 1):
                    try:
                        submit_res = http_json(
                            "POST",
                            f"{self.base_url}/v1/jobs",
                            payload=jobspec,
                            api_key=self.api_key,
                            timeout_s=self.args.http_timeout_s,
                        )
                        submit_error = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        submit_error = exc
                        if (attempt >= submit_retries) or (not is_transient_http_error(exc)):
                            break
                        try:
                            wait_http_ok(f"{self.base_url}/ready", timeout_s=min(10.0, float(self.args.http_timeout_s)))
                        except Exception:
                            pass
                        delay = retry_backoff_s * (2**attempt) + random.uniform(0.0, retry_jitter_s)
                        time.sleep(delay)
                if submit_res is None:
                    raise submit_error or RuntimeError("submit failed with unknown error")
                job_id = str(submit_res.get("job_id", "") or "")
                row["job_id"] = job_id
                if not job_id:
                    raise RuntimeError("missing job_id in submit response")

                deadline = time.time() + float(self.args.job_timeout_s)
                while time.time() < deadline:
                    status_err: Optional[Exception] = None
                    for attempt in range(status_retries + 1):
                        try:
                            status_payload = http_json(
                                "GET",
                                f"{self.base_url}/v1/jobs/{job_id}",
                                api_key=self.api_key,
                                timeout_s=self.args.http_timeout_s,
                            )
                            status_err = None
                            break
                        except Exception as exc:  # noqa: BLE001
                            status_err = exc
                            if (attempt >= status_retries) or (not is_transient_http_error(exc)):
                                break
                            delay = retry_backoff_s * (2**attempt) + random.uniform(0.0, retry_jitter_s)
                            time.sleep(delay)
                    if status_err is not None:
                        if is_transient_http_error(status_err):
                            time.sleep(0.25)
                            continue
                        raise status_err
                    st = str(status_payload.get("status", "UNKNOWN") or "UNKNOWN")
                    if st in ("SUCCEEDED", "FAILED"):
                        break
                    time.sleep(0.25)

                st = str(status_payload.get("status", "UNKNOWN") or "UNKNOWN")
                row["status"] = st
                if st not in ("SUCCEEDED", "FAILED"):
                    row["status"] = "TIMED_OUT"
                try:
                    metrics_err: Optional[Exception] = None
                    for attempt in range(status_retries + 1):
                        try:
                            metrics_payload = http_json(
                                "GET",
                                f"{self.base_url}/v1/jobs/{job_id}/metrics",
                                api_key=self.api_key,
                                timeout_s=self.args.http_timeout_s,
                            )
                            metrics_err = None
                            break
                        except Exception as exc:  # noqa: BLE001
                            metrics_err = exc
                            if (attempt >= status_retries) or (not is_transient_http_error(exc)):
                                break
                            delay = retry_backoff_s * (2**attempt) + random.uniform(0.0, retry_jitter_s)
                            time.sleep(delay)
                    if metrics_err is not None:
                        raise metrics_err
                except Exception as exc:  # noqa: BLE001
                    row["metrics_error"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                key = exc.__class__.__name__
                self._submit_errors[key] += 1
                row["status"] = "SUBMIT_ERROR"
                row["error"] = str(exc)

            if metrics_payload:
                perf = metrics_payload.get("perf", {}) if isinstance(metrics_payload.get("perf", {}), dict) else {}
                amf = metrics_payload.get("amf", {}) if isinstance(metrics_payload.get("amf", {}), dict) else {}
                spec = metrics_payload.get("spec", {}) if isinstance(metrics_payload.get("spec", {}), dict) else {}
                sched = metrics_payload.get("scheduling", {}) if isinstance(metrics_payload.get("scheduling", {}), dict) else {}
                errs = metrics_payload.get("errors", [])
                row.update(
                    {
                        "metrics": {
                            "perf": {
                                "total_ms": float(perf.get("total_ms", 0.0) or 0.0),
                                "prefill_ms": float(perf.get("prefill_ms", 0.0) or 0.0),
                                "decode_ms": float(perf.get("decode_ms", 0.0) or 0.0),
                                "avg_tps": float(perf.get("avg_tps", 0.0) or 0.0),
                            },
                            "amf": {
                                "decision": str(amf.get("decision", "")),
                                "skip_ratio": float(amf.get("skip_ratio", 0.0) or 0.0),
                                "saved_ms": float(amf.get("saved_ms", 0.0) or 0.0),
                                "roi": float(amf.get("roi", 0.0) or 0.0),
                            },
                            "spec": {
                                "enabled": bool(spec.get("enabled", False)),
                                "acceptance_rate": float(spec.get("acceptance_rate", 0.0) or 0.0),
                                "roi": float(spec.get("roi", 0.0) or 0.0),
                            },
                            "scheduling": {
                                "lane": str(sched.get("lane", "")),
                                "queue_latency_ms": float(sched.get("queue_latency_ms", 0.0) or 0.0),
                            },
                            "errors_count": len(errs) if isinstance(errs, list) else 0,
                            "errors_excerpt": [str(x)[:240] for x in errs[:5]] if isinstance(errs, list) else [],
                        }
                    }
                )
                if str(row.get("status", "")) == "FAILED" and isinstance(errs, list) and errs and not row.get("error"):
                    row["error"] = "; ".join(str(x)[:160] for x in errs[:2])

            if status_payload:
                row["routing"] = status_payload.get("routing_decision", {})
            row["ts_done"] = utc_now()
            row["elapsed_s"] = float(max(0.0, time.time() - started))
            self._write_jsonl(self.results_path, row)

            sleep_s = interval_s - (time.time() - started)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _system_sampler(self) -> None:
        headers = [
            "ts_epoch",
            "ts_utc",
            "mem_total_bytes",
            "mem_avail_bytes",
            "mem_used_pct",
            "disk_total_bytes",
            "disk_used_bytes",
            "disk_free_bytes",
            "gpu_util_pct",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "router_alive",
            "workers_up",
            "workers_total",
        ]
        with self.system_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            while (not self._stop.is_set()) and time.time() < self.end_time:
                sample = self._collect_system_sample()
                writer.writerow(sample)
                fh.flush()
                time.sleep(max(1.0, float(self.args.system_sample_s)))

    def _collect_system_sample(self) -> Dict[str, Any]:
        ts_epoch = time.time()
        mem_total = 0
        mem_avail = 0
        try:
            with Path("/proc/meminfo").open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        mem_avail = int(line.split()[1]) * 1024
        except Exception:
            pass
        mem_used_pct = 0.0
        if mem_total > 0:
            mem_used_pct = ((mem_total - mem_avail) / mem_total) * 100.0

        du = shutil.disk_usage(str(self.out_dir))
        gpu_util = ""
        gpu_mem_used = ""
        gpu_mem_total = ""
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                first = proc.stdout.splitlines()[0].strip()
                parts = [x.strip() for x in first.split(",")]
                if len(parts) >= 3:
                    gpu_util, gpu_mem_used, gpu_mem_total = parts[0], parts[1], parts[2]
        except Exception:
            pass

        st = self.stack.status()
        return {
            "ts_epoch": f"{ts_epoch:.3f}",
            "ts_utc": utc_now(),
            "mem_total_bytes": mem_total,
            "mem_avail_bytes": mem_avail,
            "mem_used_pct": f"{mem_used_pct:.4f}",
            "disk_total_bytes": du.total,
            "disk_used_bytes": du.used,
            "disk_free_bytes": du.free,
            "gpu_util_pct": gpu_util,
            "gpu_mem_used_mb": gpu_mem_used,
            "gpu_mem_total_mb": gpu_mem_total,
            "router_alive": int(bool(st.get("router_alive", False))),
            "workers_up": int(st.get("workers_up", 0) or 0),
            "workers_total": int(st.get("workers_total", 0) or 0),
        }

    def _chaos_loop(self) -> None:
        min_s = max(5.0, float(self.args.chaos_min_interval_s))
        max_s = max(min_s, float(self.args.chaos_max_interval_s))
        while (not self._stop.is_set()) and time.time() < self.end_time:
            sleep_s = random.uniform(min_s, max_s)
            wake = min(self.end_time, time.time() + sleep_s)
            while (not self._stop.is_set()) and time.time() < wake:
                time.sleep(0.5)
            if self._stop.is_set() or time.time() >= self.end_time:
                break
            self._run_chaos_event()

    def _run_chaos_event(self) -> None:
        choices = [
            "restart_worker",
            "kill_worker",
            "restart_router",
            "node_churn_worker",
        ]
        if self.args.autoscale_chaos:
            choices.append("autoscale")
        if self.args.disk_pressure_bytes > 0:
            choices.append("disk_pressure")
        action = random.choice(choices)
        started = time.time()
        detail: Dict[str, Any] = {"action": action, "ts": utc_now()}
        try:
            if action in ("restart_worker", "kill_worker", "node_churn_worker"):
                target = self.stack.pick_worker_for_restart()
                detail["worker"] = target or ""
                if target:
                    self.stack.restart_worker(
                        target,
                        force=(action == "kill_worker"),
                        node_churn=(action == "node_churn_worker"),
                    )
            elif action == "restart_router":
                self.stack.restart_router(force=False)
            elif action == "autoscale":
                if random.random() < 0.5:
                    up = self.stack.autoscale_up()
                    detail["autoscale_up"] = up or ""
                else:
                    down = self.stack.autoscale_down()
                    detail["autoscale_down"] = down or ""
            elif action == "disk_pressure":
                detail.update(self._toggle_disk_pressure())
            detail["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            detail["status"] = "error"
            detail["error"] = str(exc)
        detail["duration_s"] = float(max(0.0, time.time() - started))
        self._write_jsonl(self.chaos_path, detail)

    def _toggle_disk_pressure(self) -> Dict[str, Any]:
        target_dir = ensure_dir((self.out_dir / "runtime" / "disk_pressure").resolve())
        chunk = b"\0" * (1024 * 1024)
        if self._disk_pressure_file is None or (not self._disk_pressure_file.exists()):
            path = target_dir / f"pressure_{int(time.time())}.bin"
            remaining = int(self.args.disk_pressure_bytes)
            with path.open("wb") as fh:
                while remaining > 0:
                    to_write = min(len(chunk), remaining)
                    fh.write(chunk[:to_write])
                    remaining -= to_write
            self._disk_pressure_file = path
            return {"disk_pressure": "fill", "bytes": int(self.args.disk_pressure_bytes), "path": str(path)}
        self._disk_pressure_file.unlink(missing_ok=True)
        old = str(self._disk_pressure_file)
        self._disk_pressure_file = None
        return {"disk_pressure": "release", "path": old}

    def _build_report(self) -> Dict[str, Any]:
        rows = read_jsonl(self.results_path)
        chaos_rows = read_jsonl(self.chaos_path)

        status_counts = Counter(str(r.get("status", "UNKNOWN")) for r in rows)
        succeeded = [r for r in rows if str(r.get("status", "")) == "SUCCEEDED"]
        total_ms = [
            float(((r.get("metrics", {}) or {}).get("perf", {}) or {}).get("total_ms", 0.0) or 0.0)
            for r in succeeded
        ]
        prefill_ms = [
            float(((r.get("metrics", {}) or {}).get("perf", {}) or {}).get("prefill_ms", 0.0) or 0.0)
            for r in succeeded
        ]
        decode_ms = [
            float(((r.get("metrics", {}) or {}).get("perf", {}) or {}).get("decode_ms", 0.0) or 0.0)
            for r in succeeded
        ]
        skip_ratio = [
            float(((r.get("metrics", {}) or {}).get("amf", {}) or {}).get("skip_ratio", 0.0) or 0.0)
            for r in succeeded
        ]
        hit_flags = [
            1
            if str(((r.get("metrics", {}) or {}).get("amf", {}) or {}).get("decision", "")) == "hit"
            else 0
            for r in succeeded
        ]
        errors_count = sum(int(((r.get("metrics", {}) or {}).get("errors_count", 0) or 0)) for r in succeeded)

        start_epoch = None
        if rows:
            try:
                first_ts = rows[0].get("ts_submit", "")
                start_epoch = datetime.fromisoformat(str(first_ts).replace("Z", "+00:00")).timestamp()
            except Exception:
                start_epoch = None
        bucket_s = max(60, int(self.args.report_bucket_s))
        buckets: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"lat": [], "hit": 0, "count": 0, "prefill": [], "decode": []})
        if start_epoch is not None:
            for r in succeeded:
                try:
                    ts_done = str(r.get("ts_done", "") or "")
                    done_epoch = datetime.fromisoformat(ts_done.replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
                idx = int(max(0.0, done_epoch - start_epoch) // bucket_s)
                perf = (r.get("metrics", {}) or {}).get("perf", {}) or {}
                amf = (r.get("metrics", {}) or {}).get("amf", {}) or {}
                buckets[idx]["lat"].append(float(perf.get("total_ms", 0.0) or 0.0))
                buckets[idx]["prefill"].append(float(perf.get("prefill_ms", 0.0) or 0.0))
                buckets[idx]["decode"].append(float(perf.get("decode_ms", 0.0) or 0.0))
                buckets[idx]["count"] += 1
                if str(amf.get("decision", "")) == "hit":
                    buckets[idx]["hit"] += 1

        series: List[Dict[str, Any]] = []
        for idx in sorted(buckets.keys()):
            b = buckets[idx]
            series.append(
                {
                    "bucket_idx": idx,
                    "bucket_start_s": idx * bucket_s,
                    "count": int(b["count"]),
                    "hit_rate": (float(b["hit"]) / float(b["count"])) if b["count"] else 0.0,
                    "p50_ms": percentile(b["lat"], 0.50),
                    "p95_ms": percentile(b["lat"], 0.95),
                    "p99_ms": percentile(b["lat"], 0.99),
                    "prefill_p95_ms": percentile(b["prefill"], 0.95),
                    "decode_p95_ms": percentile(b["decode"], 0.95),
                }
            )

        drift = {
            "hit_rate_delta": 0.0,
            "p95_ms_delta": 0.0,
        }
        if len(series) >= 2:
            drift["hit_rate_delta"] = float(series[-1]["hit_rate"]) - float(series[0]["hit_rate"])
            drift["p95_ms_delta"] = float(series[-1]["p95_ms"]) - float(series[0]["p95_ms"])

        system_rows: List[Dict[str, Any]] = []
        if self.system_path.exists():
            with self.system_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                system_rows = list(reader)
        mem_peak = 0.0
        disk_start = 0
        disk_end = 0
        gpu_util_peak = 0.0
        for i, r in enumerate(system_rows):
            try:
                mem_peak = max(mem_peak, float(r.get("mem_used_pct", "0") or 0.0))
            except Exception:
                pass
            if i == 0:
                try:
                    disk_start = int(r.get("disk_used_bytes", "0") or 0)
                except Exception:
                    disk_start = 0
            try:
                disk_end = int(r.get("disk_used_bytes", disk_end) or disk_end)
            except Exception:
                pass
            try:
                gpu_util_peak = max(gpu_util_peak, float(r.get("gpu_util_pct", "0") or 0.0))
            except Exception:
                pass

        chaos_counts = Counter(str(r.get("action", "unknown")) for r in chaos_rows)
        chaos_errors = sum(1 for r in chaos_rows if str(r.get("status", "")) != "ok")

        per_org = defaultdict(lambda: {"count": 0, "hit": 0, "lat": []})
        for r in succeeded:
            org = str(r.get("org_id", "unknown") or "unknown")
            amf = (r.get("metrics", {}) or {}).get("amf", {}) or {}
            perf = (r.get("metrics", {}) or {}).get("perf", {}) or {}
            per_org[org]["count"] += 1
            if str(amf.get("decision", "")) == "hit":
                per_org[org]["hit"] += 1
            per_org[org]["lat"].append(float(perf.get("total_ms", 0.0) or 0.0))

        per_org_summary = {
            org: {
                "count": int(v["count"]),
                "hit_rate": (float(v["hit"]) / float(v["count"])) if v["count"] else 0.0,
                "p95_ms": percentile(v["lat"], 0.95),
            }
            for org, v in per_org.items()
        }

        report = {
            "generated_at": utc_now(),
            "run_dir": str(self.out_dir),
            "duration_s": float(self.args.duration_s),
            "summary": {
                "requests_total": len(rows),
                "requests_succeeded": len(succeeded),
                "requests_target": int(getattr(self.args, "target_requests", 0) or 0),
                "target_reached": bool(
                    int(getattr(self.args, "target_requests", 0) or 0) > 0
                    and len(rows) >= int(getattr(self.args, "target_requests", 0) or 0)
                ),
                "status_counts": dict(status_counts),
                "submit_error_counts": dict(self._submit_errors),
                "latency_ms": {
                    "p50": percentile(total_ms, 0.50),
                    "p95": percentile(total_ms, 0.95),
                    "p99": percentile(total_ms, 0.99),
                },
                "prefill_ms_p95": percentile(prefill_ms, 0.95),
                "decode_ms_p95": percentile(decode_ms, 0.95),
                "amf": {
                    "hit_rate": (sum(hit_flags) / len(hit_flags)) if hit_flags else 0.0,
                    "skip_ratio_avg": (sum(skip_ratio) / len(skip_ratio)) if skip_ratio else 0.0,
                },
                "error_events_in_metrics": int(errors_count),
                "drift": drift,
            },
            "system": {
                "mem_used_pct_peak": mem_peak,
                "gpu_util_pct_peak": gpu_util_peak,
                "disk_growth_bytes": int(max(0, disk_end - disk_start)),
            },
            "chaos": {
                "events_total": len(chaos_rows),
                "events_error": chaos_errors,
                "event_counts": dict(chaos_counts),
            },
            "per_org": per_org_summary,
            "series": series,
        }
        return report

    def _write_report_files(self, report: Dict[str, Any]) -> None:
        report_json = self.out_dir / "report" / "summary.json"
        report_md = self.out_dir / "report" / "summary.md"
        curves_csv = self.out_dir / "report" / "curves.csv"

        report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        series = report.get("series", [])
        with curves_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "bucket_idx",
                    "bucket_start_s",
                    "count",
                    "hit_rate",
                    "p50_ms",
                    "p95_ms",
                    "p99_ms",
                    "prefill_p95_ms",
                    "decode_p95_ms",
                ],
            )
            writer.writeheader()
            for row in series:
                writer.writerow(row)

        s = report.get("summary", {})
        amf = s.get("amf", {})
        lat = s.get("latency_ms", {})
        chaos = report.get("chaos", {})
        system = report.get("system", {})
        lines = [
            "# 24h Soak + Chaos Report",
            "",
            f"Generated: {report.get('generated_at', '')}",
            f"Run dir: `{report.get('run_dir', '')}`",
            "",
            "## High-level",
            f"- Requests total: {s.get('requests_total', 0)}",
            f"- Requests succeeded: {s.get('requests_succeeded', 0)}",
            f"- Requests target: {s.get('requests_target', 0)}",
            f"- Target reached: {bool(s.get('target_reached', False))}",
            f"- AMF hit rate: {float(amf.get('hit_rate', 0.0)):.4f}",
            f"- AMF skip ratio avg: {float(amf.get('skip_ratio_avg', 0.0)):.4f}",
            f"- Latency p50/p95/p99 (ms): {float(lat.get('p50', 0.0)):.2f} / {float(lat.get('p95', 0.0)):.2f} / {float(lat.get('p99', 0.0)):.2f}",
            f"- Drift hit_rate delta: {float((s.get('drift', {}) or {}).get('hit_rate_delta', 0.0)):.4f}",
            f"- Drift p95 delta (ms): {float((s.get('drift', {}) or {}).get('p95_ms_delta', 0.0)):.2f}",
            "",
            "## Chaos",
            f"- Events total: {chaos.get('events_total', 0)}",
            f"- Events error: {chaos.get('events_error', 0)}",
            f"- Event counts: `{json.dumps(chaos.get('event_counts', {}), ensure_ascii=False)}`",
            "",
            "## System",
            f"- Memory peak used %: {float(system.get('mem_used_pct_peak', 0.0)):.2f}",
            f"- GPU util peak %: {float(system.get('gpu_util_pct_peak', 0.0)):.2f}",
            f"- Disk growth bytes: {int(system.get('disk_growth_bytes', 0) or 0)}",
            "",
            "## Artifacts",
            f"- Raw results: `{self.results_path}`",
            f"- Chaos events: `{self.chaos_path}`",
            f"- System metrics: `{self.system_path}`",
            f"- Curves CSV: `{curves_csv}`",
            f"- Summary JSON: `{report_json}`",
        ]
        report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> Dict[str, Any]:
        self._validate_args()
        config_payload = {
            "generated_at": utc_now(),
            "args": vars(self.args),
            "env_overrides": {
                "KORITH_PLATFORM_DB": self.env.get("KORITH_PLATFORM_DB", ""),
                "KORITH_PLATFORM_ARTIFACTS": self.env.get("KORITH_PLATFORM_ARTIFACTS", ""),
                "KORITH_QUEUE_DB": self.env.get("KORITH_QUEUE_DB", ""),
                "KORITH_REGISTRY_DB": self.env.get("KORITH_REGISTRY_DB", ""),
                "KORITH_RESTORE_DB": self.env.get("KORITH_RESTORE_DB", ""),
                "KORITH_NODE_REGISTRY_DB": self.env.get("KORITH_NODE_REGISTRY_DB", ""),
                "KORITH_API_KEY_SALT_SET": bool(self.env.get("KORITH_API_KEY_SALT", "")),
            },
        }
        self.meta_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        if self.args.dry_run:
            return {
                "generated_at": utc_now(),
                "run_dir": str(self.out_dir),
                "dry_run": True,
            }

        self._start_stack_if_needed()
        if not self.api_key:
            raise ValueError("--api-key is required when --start-stack=0")

        self.end_time = time.time() + float(self.args.duration_s)
        try:
            # Monitoring thread.
            sampler = threading.Thread(target=self._system_sampler, name="system-sampler", daemon=True)
            sampler.start()
            self._active_threads.append(sampler)

            # Optional chaos.
            if self.args.chaos_enabled:
                chaos = threading.Thread(target=self._chaos_loop, name="chaos-loop", daemon=True)
                chaos.start()
                self._active_threads.append(chaos)

            # Load threads.
            for idx in range(max(1, int(self.args.concurrency))):
                t = threading.Thread(target=self._submit_one, args=(idx,), name=f"submit-{idx}", daemon=True)
                t.start()
                self._active_threads.append(t)

            while (not self._stop.is_set()) and time.time() < self.end_time:
                time.sleep(1.0)

        finally:
            self._stop.set()
            for t in self._active_threads:
                t.join(timeout=5.0)
            if self.args.start_stack:
                self.stack.stop_all()

        report = self._build_report()
        self._write_report_files(report)
        return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="24h soak + chaos harness for Axropus runtime consistency.")
    p.add_argument("--duration-s", type=float, default=24 * 3600)
    p.add_argument("--output-root", default="platform_data/soak")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--start-stack", type=int, default=1, help="1=start router/worker stack, 0=use existing stack")
    p.add_argument("--router-host", default="127.0.0.1")
    p.add_argument("--router-port", type=int, default=18000)
    p.add_argument("--worker-host", default="127.0.0.1")
    p.add_argument("--worker-base-port", type=int, default=19000)
    p.add_argument("--initial-workers", type=int, default=1)
    p.add_argument("--max-workers", type=int, default=2)

    p.add_argument("--api-key", default=os.environ.get("KORITH_API_KEY", ""))
    p.add_argument("--http-timeout-s", type=float, default=20.0)
    p.add_argument("--job-timeout-s", type=float, default=600.0)
    p.add_argument("--target-requests", type=int, default=0, help="stop early when this many result rows are recorded")
    p.add_argument("--submit-retry-max", type=int, default=4)
    p.add_argument("--status-retry-max", type=int, default=2)
    p.add_argument("--retry-backoff-s", type=float, default=0.25)
    p.add_argument("--retry-jitter-s", type=float, default=0.15)
    p.add_argument("--rps", type=float, default=0.25, help="target total requests/sec across all submit threads")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--system-sample-s", type=float, default=10.0)

    p.add_argument("--backend-id", default=os.environ.get("KORITH_SOAK_BACKEND_ID", "korith_local"))
    p.add_argument("--model-id", default=os.environ.get("KORITH_SOAK_MODEL_ID", "soak-model"))
    p.add_argument("--model-path", default=os.environ.get("KORITH_SOAK_MODEL_PATH", ""))
    p.add_argument("--model-endpoint", default=os.environ.get("KORITH_SOAK_MODEL_ENDPOINT", ""))
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--n-batch", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--allow-spec", type=int, default=0)

    p.add_argument("--orgs", default="tenant_a,tenant_b,tenant_c")
    p.add_argument("--users-per-org", type=int, default=3)
    p.add_argument("--documents", type=int, default=8)
    p.add_argument("--prompt-tokens-max", type=int, default=4096)
    p.add_argument("--short-context-rate", type=float, default=0.30)
    p.add_argument("--long-context-rate", type=float, default=0.20)
    p.add_argument("--mutation-rate", type=float, default=0.25)
    p.add_argument("--partial-overlap-rate", type=float, default=0.55)
    p.add_argument("--system-prompt", default="You are an enterprise assistant. Keep outputs deterministic.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--chaos-enabled", type=int, default=1)
    p.add_argument("--chaos-min-interval-s", type=float, default=2 * 3600)
    p.add_argument("--chaos-max-interval-s", type=float, default=4 * 3600)
    p.add_argument("--autoscale-chaos", type=int, default=1)
    p.add_argument("--disk-pressure-bytes", type=int, default=0)

    p.add_argument("--report-bucket-s", type=int, default=900)
    args = p.parse_args()
    args.start_stack = bool(int(args.start_stack))
    args.allow_spec = bool(int(args.allow_spec))
    args.chaos_enabled = bool(int(args.chaos_enabled))
    args.autoscale_chaos = bool(int(args.autoscale_chaos))
    return args


def main() -> None:
    args = parse_args()
    runner = SoakRunner(args)
    report = runner.run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
