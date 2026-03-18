from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from ..gateway.auth import AuthContext, AuthError, authenticate
from ..gateway.auth import require_api_key_salt_configured
from ..gateway.rate_limit import build_rate_limiter
from ..observability.metrics import GLOBAL_METRICS
from ..observability.metrics_exporter import render_router_amf_metrics
from ..observability.platform_logging import emit_log


RATE_LIMITER = build_rate_limiter(
    backend=os.environ.get("KORITH_RATE_LIMIT_BACKEND", "sqlite"),
    sqlite_path=Path(os.environ.get("KORITH_RATE_LIMIT_DB", "./platform_data/rate_limit.sqlite")),
    redis_url=os.environ.get("KORITH_REDIS_URL"),
    burst_factor=float(os.environ.get("KORITH_RATE_LIMIT_BURST", "2.0")),
)


class KorithHandler(BaseHTTPRequestHandler):
    router = None
    _soft_limit_lock = threading.Lock()
    _soft_limit_seen: Dict[str, float] = {}
    _tenant_rps_lock = threading.Lock()
    _tenant_rps_state: Dict[str, tuple[int, int]] = {}

    def _request_id(self) -> str:
        rid = self.headers.get("X-Request-Id") or self.headers.get("x-request-id")
        return rid.strip() if rid else str(uuid.uuid4())

    def _send(
        self,
        code: int,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if request_id:
            self.send_header("X-Request-Id", request_id)
        if headers:
            for name, value in headers.items():
                self.send_header(str(name), str(value))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self, request_id: str) -> AuthContext:
        GLOBAL_METRICS.inc("auth_requests_total", 1.0)
        try:
            ctx = authenticate(dict(self.headers.items()), KorithHandler.router.ledger)
            return ctx
        except AuthError as exc:
            GLOBAL_METRICS.inc("auth_failures_total", 1.0)
            emit_log(
                "gateway",
                "AUTH_FAILURE",
                {
                    "job_id": "",
                    "run_id": "",
                    "worker_id": "",
                    "session_id": "",
                    "org_id": "",
                    "request_id": request_id,
                    "latency_ms": 0.0,
                    "error_type": "AuthError",
                    "error": exc.message,
                    "replay_related": False,
                },
            )
            raise

    def _tenant_id(self, auth_ctx: AuthContext) -> str:
        raw = self.headers.get("X-Axropus-Tenant-Id") or self.headers.get("x-axropus-tenant-id") or ""
        tenant = str(raw or auth_ctx.org_id or "default").strip() or "default"
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in tenant)
        if not safe:
            safe = str(auth_ctx.org_id or "default")
        return safe[:128]

    def _check_tenant_rps(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        raw = str(os.environ.get("KORITH_AMF_TENANT_RATE_LIMIT_RPS", "") or "").strip()
        if not raw:
            return None
        try:
            limit = int(raw)
        except Exception:
            return None
        if limit <= 0:
            return None
        now_sec = int(time.time())
        retry_after = 1
        with KorithHandler._tenant_rps_lock:
            sec, count = KorithHandler._tenant_rps_state.get(tenant_id, (now_sec, 0))
            if sec != now_sec:
                sec, count = now_sec, 0
            if count >= limit:
                KorithHandler._tenant_rps_state[tenant_id] = (sec, count)
                # keep the map bounded by dropping stale windows.
                stale = [k for k, (ts, _) in KorithHandler._tenant_rps_state.items() if ts < (now_sec - 2)]
                for key in stale:
                    KorithHandler._tenant_rps_state.pop(key, None)
                return {
                    "error": "tenant rate limit exceeded",
                    "tenant_id": tenant_id,
                    "limit_rps": limit,
                    "retry_after": retry_after,
                }
            KorithHandler._tenant_rps_state[tenant_id] = (sec, count + 1)
            stale = [k for k, (ts, _) in KorithHandler._tenant_rps_state.items() if ts < (now_sec - 2)]
            for key in stale:
                KorithHandler._tenant_rps_state.pop(key, None)
        return None

    def _check_limits(self, org_id: str, token_estimate: int, ctx: AuthContext, request_id: str) -> Optional[Dict[str, Any]]:
        decision = RATE_LIMITER.check_and_consume(
            org_id=org_id,
            request_inc=1,
            token_inc=max(1, token_estimate),
            rpm_limit=ctx.rate_limit_rpm,
            tpm_limit=ctx.rate_limit_tpm,
        )
        GLOBAL_METRICS.set_gauge("org_request_rate", decision.org_request_rate, labels={"org_id": org_id})
        GLOBAL_METRICS.set_gauge("org_token_rate", decision.org_token_rate, labels={"org_id": org_id})
        if decision.allowed:
            req_ratio = (decision.org_request_rate / float(decision.rpm_cap)) if decision.rpm_cap > 0 else 0.0
            tok_ratio = (decision.org_token_rate / float(decision.tpm_cap)) if decision.tpm_cap > 0 else 0.0
            if max(req_ratio, tok_ratio) >= 0.8:
                cache_key = f"{org_id}:{decision.window_epoch}"
                now = time.time()
                should_emit = False
                with KorithHandler._soft_limit_lock:
                    if cache_key not in KorithHandler._soft_limit_seen:
                        should_emit = True
                        KorithHandler._soft_limit_seen[cache_key] = now
                    # Keep only recent windows.
                    stale_before = now - 600.0
                    stale_keys = [k for k, ts in KorithHandler._soft_limit_seen.items() if ts < stale_before]
                    for stale_key in stale_keys:
                        KorithHandler._soft_limit_seen.pop(stale_key, None)
                if should_emit:
                    GLOBAL_METRICS.inc("rate_limit_soft_warnings_total", 1.0, labels={"org_id": org_id})
                    emit_log(
                        "gateway",
                        "RATE_LIMIT_SOFT_WARN",
                        {
                            "job_id": "",
                            "run_id": "",
                            "worker_id": "",
                            "session_id": "",
                            "org_id": org_id,
                            "request_id": request_id,
                            "latency_ms": 0.0,
                            "org_request_rate": decision.org_request_rate,
                            "org_token_rate": decision.org_token_rate,
                            "rpm_cap": decision.rpm_cap,
                            "tpm_cap": decision.tpm_cap,
                        },
                    )
            return None
        GLOBAL_METRICS.inc("rate_limit_blocks_total", 1.0, labels={"org_id": org_id})
        emit_log(
            "gateway",
            "RATE_LIMIT_BLOCK",
            {
                "job_id": "",
                "run_id": "",
                "worker_id": "",
                "session_id": "",
                "org_id": org_id,
                "request_id": request_id,
                "latency_ms": 0.0,
                "reason": decision.reason,
                "token_estimate": token_estimate,
            },
        )
        return {
            "error": "rate limit exceeded",
            "reason": decision.reason,
            "org_request_rate": decision.org_request_rate,
            "org_token_rate": decision.org_token_rate,
        }

    def _estimate_tokens(self, job: Dict[str, Any]) -> int:
        prompt = job.get("prompt")
        if not prompt:
            template = job.get("prompt_template", "")
            data = job.get("input", {}) or {}
            try:
                prompt = template.format(**data)
            except Exception:
                prompt = template
        prompt = prompt or ""
        prompt_tokens = max(1, len(prompt.split()))
        backend_id = job.get("backend_id", "")
        if backend_id and backend_id != "hf_transformers":
            try:
                adapter = KorithHandler.router.adapter_registry.get_adapter(job)
                prompt_tokens = max(1, int(adapter.tokenize(prompt)))
            except Exception:
                pass
        max_tokens = int(job.get("deterministic_cfg", {}).get("max_tokens", 256) or 256)
        return prompt_tokens + max_tokens

    def _active_alerts(self) -> list[Dict[str, Any]]:
        snap = GLOBAL_METRICS.snapshot()
        counters = snap["counters"]
        gauges = snap["gauges"]

        alerts: list[Dict[str, Any]] = []
        jobs_total = counters.get("jobs_processed_total", 0.0)
        replay_failures = counters.get("replay_failures_total", 0.0)
        if jobs_total > 0 and (replay_failures / jobs_total) > float(os.environ.get("KORITH_ALERT_REPLAY_FAILURE_RATE", "0.05")):
            alerts.append({"name": "replay_failure_rate", "severity": "warning", "value": replay_failures / jobs_total})

        if counters.get("mf_restore_failures", 0.0) > float(os.environ.get("KORITH_ALERT_MF_RESTORE_FAILURES", "10")):
            alerts.append({"name": "mf_restore_failures", "severity": "warning", "value": counters.get("mf_restore_failures", 0.0)})

        queue_latency = gauges.get("queue_latency_ms", 0.0)
        if queue_latency > float(os.environ.get("KORITH_ALERT_QUEUE_LATENCY_MS", "500")):
            alerts.append({"name": "queue_latency_sla", "severity": "critical", "value": queue_latency})

        if counters.get("negative_roi_streaks", 0.0) > float(os.environ.get("KORITH_ALERT_NEGATIVE_ROI_STREAKS", "25")):
            alerts.append({"name": "negative_roi_streaks", "severity": "warning", "value": counters.get("negative_roi_streaks", 0.0)})

        worker_health_values = [value for key, value in gauges.items() if key.startswith("worker_health")]
        if worker_health_values and min(worker_health_values) < 1.0:
            alerts.append({"name": "worker_health_drop", "severity": "critical", "value": min(worker_health_values)})

        return alerts

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        request_id = self._request_id()

        if path == "/health":
            self._send(200, {"status": "ok"}, request_id=request_id)
            return
        if path == "/ready":
            try:
                workers = KorithHandler.router.get_workers() if hasattr(KorithHandler.router, "get_workers") else []
                ready = len(workers) > 0
            except Exception:
                ready = False
            self._send(200 if ready else 503, {"ready": ready}, request_id=request_id)
            return

        if path == "/metrics":
            text = GLOBAL_METRICS.render_prometheus()
            if str(os.environ.get("KORITH_METRICS_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on"):
                try:
                    workers = KorithHandler.router.get_workers() if hasattr(KorithHandler.router, "get_workers") else []
                    extra = render_router_amf_metrics(workers)
                    if extra:
                        text += "\n" + extra
                except Exception:
                    pass
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        auth_ctx = None
        if path.startswith("/v1/"):
            try:
                auth_ctx = self._auth(request_id)
            except AuthError as exc:
                self._send(exc.status_code, {"error": exc.message}, request_id=request_id)
                return

        if path == "/v1/jobs":
            limit = 20
            org_filter = auth_ctx.org_id
            if "limit" in query:
                try:
                    limit = int(query["limit"][0])
                except ValueError:
                    limit = 20
            jobs = KorithHandler.router.list_jobs(limit, org_id=org_filter)
            self._send(200, {"jobs": jobs}, request_id=request_id)
            return
        if path == "/v1/health":
            self._send(200, {"status": "ok"}, request_id=request_id)
            return
        if path == "/v1/alerts":
            self._send(200, {"alerts": self._active_alerts()}, request_id=request_id)
            return
        if path == "/v1/workers":
            if hasattr(KorithHandler.router, "get_workers"):
                self._send(200, {"workers": KorithHandler.router.get_workers()}, request_id=request_id)
                return
        if path == "/v1/queue_status":
            if hasattr(KorithHandler.router, "queue_status"):
                self._send(200, {"queue": KorithHandler.router.queue_status()}, request_id=request_id)
                return
        if path == "/v1/capabilities":
            if hasattr(KorithHandler.router, "capabilities"):
                self._send(200, {"capabilities": KorithHandler.router.capabilities()}, request_id=request_id)
                return
        if path == "/v1/cluster/nodes":
            if hasattr(KorithHandler.router, "cluster_nodes"):
                self._send(200, {"nodes": KorithHandler.router.cluster_nodes()}, request_id=request_id)
                return
        if path == "/v1/cluster/routing":
            fingerprint = query.get("fingerprint", [""])[0]
            if not fingerprint:
                self._send(400, {"error": "fingerprint query param required"}, request_id=request_id)
                return
            if hasattr(KorithHandler.router, "routing_debug"):
                self._send(
                    200,
                    KorithHandler.router.routing_debug(fingerprint_hash=fingerprint, org_id=auth_ctx.org_id),
                    request_id=request_id,
                )
                return
        if path == "/v1/snapshots/locations":
            fingerprint = query.get("fingerprint", [""])[0]
            if not fingerprint:
                self._send(400, {"error": "fingerprint query param required"}, request_id=request_id)
                return
            if hasattr(KorithHandler.router, "snapshot_locations"):
                self._send(
                    200,
                    KorithHandler.router.snapshot_locations(fingerprint_hash=fingerprint, org_id=auth_ctx.org_id),
                    request_id=request_id,
                )
                return
        if path == "/v1/replay/status":
            fingerprint = query.get("fingerprint", [""])[0]
            if not fingerprint:
                self._send(400, {"error": "fingerprint query param required"}, request_id=request_id)
                return
            self._send(200, KorithHandler.router.replay_status(fingerprint), request_id=request_id)
            return
        if path == "/v1/spec/status":
            fingerprint = query.get("fingerprint", [""])[0]
            if not fingerprint:
                self._send(400, {"error": "fingerprint query param required"}, request_id=request_id)
                return
            if hasattr(KorithHandler.router, "spec_status"):
                self._send(200, KorithHandler.router.spec_status(fingerprint, org_id=auth_ctx.org_id), request_id=request_id)
                return
        if path == "/v1/kernels/status":
            if hasattr(KorithHandler.router, "kernels_status"):
                self._send(200, KorithHandler.router.kernels_status(), request_id=request_id)
                return
        if path == "/v1/kpi/summary":
            if hasattr(KorithHandler.router, "kpi_summary"):
                limit = 500
                gpu_hourly_cost = 2.5
                if "limit" in query:
                    try:
                        limit = max(1, int(query["limit"][0]))
                    except Exception:
                        limit = 500
                if "gpu_hourly_cost" in query:
                    try:
                        gpu_hourly_cost = float(query["gpu_hourly_cost"][0])
                    except Exception:
                        gpu_hourly_cost = 2.5
                self._send(
                    200,
                    KorithHandler.router.kpi_summary(
                        org_id=auth_ctx.org_id,
                        limit=limit,
                        gpu_hourly_cost=gpu_hourly_cost,
                    ),
                    request_id=request_id,
                )
                return
        if path.startswith("/v1/metrics/tenant/"):
            parts = path.split("/")
            tenant_id = parts[4] if len(parts) > 4 else ""
            if not tenant_id:
                self._send(400, {"error": "tenant_id required"}, request_id=request_id)
                return
            if hasattr(KorithHandler.router, "tenant_metrics"):
                self._send(
                    200,
                    KorithHandler.router.tenant_metrics(
                        tenant_id,
                        org_id=auth_ctx.org_id,
                    ),
                    request_id=request_id,
                )
                return
        if path.startswith("/v1/jobs/"):
            parts = path.split("/")
            job_id = parts[3] if len(parts) > 3 else ""
            rec = KorithHandler.router.get_job(job_id, org_id=auth_ctx.org_id)
            if not rec:
                self._send(404, {"error": "job not found"}, request_id=request_id)
                return
            if path.endswith("/metrics"):
                metrics_path = rec.get("metrics_path")
                if metrics_path and Path(metrics_path).exists():
                    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
                    self._send(200, payload, request_id=request_id)
                else:
                    self._send(200, {"metrics_path": metrics_path}, request_id=request_id)
                return
            if path.endswith("/events"):
                events_path = rec.get("events_path")
                if events_path and Path(events_path).exists():
                    events = []
                    with Path(events_path).open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                events.append(json.loads(line))
                            except Exception:
                                events.append({"raw": line})
                    self._send(200, {"events": events}, request_id=request_id)
                else:
                    self._send(200, {"events_path": events_path}, request_id=request_id)
                return
            if path.endswith("/logs"):
                log_path = rec.get("log_path")
                platform_log_path = ""
                if rec.get("org_id"):
                    platform_log_path = str(Path(os.environ.get("KORITH_PLATFORM_ARTIFACTS", "./platform_data/artifacts")) / rec["org_id"] / job_id / "platform.log")
                data = ""
                if log_path and Path(log_path).exists():
                    data = Path(log_path).read_text(encoding="utf-8", errors="replace")
                platform_data = ""
                if platform_log_path and Path(platform_log_path).exists():
                    platform_data = Path(platform_log_path).read_text(encoding="utf-8", errors="replace")
                self._send(200, {"log": data, "platform_log": platform_data}, request_id=request_id)
                return
            self._send(200, rec, request_id=request_id)
            return

        self._send(404, {"error": "not found"}, request_id=request_id)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        request_id = self._request_id()

        auth_ctx = None
        if parsed.path.startswith("/v1/"):
            try:
                auth_ctx = self._auth(request_id)
            except AuthError as exc:
                self._send(exc.status_code, {"error": exc.message}, request_id=request_id)
                return

        if parsed.path == "/v1/jobs":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                job = json.loads(body)
            except json.JSONDecodeError as exc:
                self._send(
                    400,
                    {"error": f"invalid json body: {exc}", "code": "INVALID_JSON"},
                    request_id=request_id,
                )
                return
            if "owner" not in job or not isinstance(job["owner"], dict):
                job["owner"] = {}
            job["owner"]["org_id"] = auth_ctx.org_id
            tenant_id = self._tenant_id(auth_ctx)
            job["owner"]["tenant_id"] = tenant_id
            job["tenant_id"] = tenant_id
            tenant_blocked = self._check_tenant_rps(tenant_id)
            if tenant_blocked:
                self._send(
                    429,
                    tenant_blocked,
                    request_id=request_id,
                    headers={"Retry-After": str(int(tenant_blocked.get("retry_after", 1) or 1))},
                )
                return
            token_estimate = self._estimate_tokens(job)
            blocked = self._check_limits(auth_ctx.org_id, token_estimate, auth_ctx, request_id)
            if blocked:
                self._send(429, blocked, request_id=request_id)
                return
            try:
                job_id = KorithHandler.router.submit(
                    job,
                    org_id=auth_ctx.org_id,
                    request_id=request_id,
                    tenant_id=tenant_id,
                )
            except ValueError as exc:
                status_code = int(getattr(exc, "status_code", 400) or 400)
                if hasattr(exc, "to_payload") and callable(getattr(exc, "to_payload")):
                    payload = exc.to_payload()
                else:
                    payload = {"error": str(exc), "code": "BAD_REQUEST"}
                self._send(status_code, payload, request_id=request_id)
                return
            self._send(200, {"job_id": job_id, "request_id": request_id}, request_id=request_id)
            return
        if parsed.path == "/v1/billing/report":
            if not hasattr(KorithHandler.router, "billing_report"):
                self._send(404, {"error": "not found"}, request_id=request_id)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
            payload: Dict[str, Any]
            try:
                payload = json.loads(body) if body.strip() else {}
            except Exception:
                payload = {}
            tenant_id = str(payload.get("tenant_id", "") or "")
            try:
                gpu_hourly_cost = float(payload.get("gpu_hourly_cost", 2.5) or 2.5)
            except Exception:
                gpu_hourly_cost = 2.5
            try:
                limit = int(payload.get("limit", 5000) or 5000)
            except Exception:
                limit = 5000
            report = KorithHandler.router.billing_report(
                org_id=auth_ctx.org_id,
                tenant_id=tenant_id,
                limit=limit,
                gpu_hourly_cost=gpu_hourly_cost,
            )
            self._send(200, report, request_id=request_id)
            return
        if parsed.path.startswith("/v1/jobs/") and parsed.path.endswith("/restore"):
            job_id = parsed.path.split("/")[3]
            res = KorithHandler.router.restore(job_id, org_id=auth_ctx.org_id, request_id=request_id)
            code = 200 if res.get("restored") else 400
            self._send(code, res, request_id=request_id)
            return

        self._send(404, {"error": "not found"}, request_id=request_id)


def serve(router, host: str = "0.0.0.0", port: int = 8000):
    require_api_key_salt_configured()
    KorithHandler.router = router
    httpd = ThreadingHTTPServer((host, port), KorithHandler)
    httpd.serve_forever()
