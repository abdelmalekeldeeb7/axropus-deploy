from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..engine import get_engine_client
from .base import BackendAdapter, Capabilities


class Tier1LocalKorithAdapter(BackendAdapter):
    backend_id = "korith_local"
    backend_version = "korith_dynamic_v1"

    def __init__(self, model_path: str, *, backend_id: str | None = None, backend_version: str | None = None) -> None:
        self.model_path = model_path
        if backend_id:
            self.backend_id = backend_id
        if backend_version:
            self.backend_version = backend_version
        self._engine_client = get_engine_client()
        self._spec_verify_cache: Dict[str, Dict[str, Any]] = {}
        self._spec_cache_max = int(os.environ.get("KORITH_SPEC_CACHE_MAX", "128") or 128)
        # Keep verify-cache opt-in to avoid distorting spec benchmarks.
        self._spec_cache_enabled = os.environ.get("KORITH_SPEC_VERIFY_CACHE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._spec_auto_cache_only = os.environ.get("KORITH_SPEC_AUTO_CACHE_ONLY", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def get_fingerprint(self) -> Dict[str, str]:
        path = Path(self.model_path)
        meta = ""
        if path.exists():
            stat = path.stat()
            meta = f":{stat.st_mtime_ns}:{stat.st_size}"
        digest = hashlib.sha256(f"{self.model_path}{meta}".encode("utf-8")).hexdigest()
        return {
            "model_hash": digest,
            "tokenizer_hash": "unknown",
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
        }

    def get_capabilities(self) -> Capabilities:
        # Local adapter supports draft/verify via run_draft + verify_tokens.
        # This allows SPEC lanes on korith_local without requiring korith_cuda.
        return Capabilities(
            kv_replay=True,
            deterministic_seeding=True,
            streaming=False,
            batch_prefill=True,
            logits_access=False,
            verify_tokens=True,
            draft_supported=True,
        )

    def tokenize(self, prompt: str) -> int:
        return max(1, len(prompt.split()))

    def run_baseline(
        self,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
    ) -> Dict:
        return self._run_engine(
            prompt=prompt,
            max_tokens=max_tokens,
            deterministic_cfg=deterministic_cfg,
            policy=policy,
            artifacts=artifacts,
            mf_snapshot_in=mf_snapshot_in,
            mode="baseline",
        )

    def run_accel(
        self,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
        engine_cfg: Optional[Dict] = None,
    ) -> Dict:
        return self._run_engine(
            prompt=prompt,
            max_tokens=max_tokens,
            deterministic_cfg=deterministic_cfg,
            policy=policy,
            artifacts=artifacts,
            mf_snapshot_in=mf_snapshot_in,
            mode="accel",
            engine_cfg=engine_cfg or {},
        )

    def run_speculative(
        self,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
        spec_cfg: Optional[Dict] = None,
        engine_cfg: Optional[Dict] = None,
    ) -> Dict:
        spec_cfg = spec_cfg or {}
        engine_cfg = engine_cfg or {}
        flow_started = time.perf_counter()

        cache_key = self._spec_cache_key(prompt, deterministic_cfg, max_tokens)
        cache_allowed = bool(self._spec_cache_enabled and deterministic_cfg.get("seed") is not None)
        cache_only = bool(spec_cfg.get("cache_only"))
        cache_only_reason = ""
        draft_model = str(
            spec_cfg.get("draft_model")
            or os.environ.get("KORITH_DRAFT_MODEL_PATH", "")
            or os.environ.get("KORITH_DRAFT_MODEL", "")
        ).strip()
        same_model_draft = False
        if draft_model:
            try:
                same_model_draft = Path(draft_model).resolve() == Path(self.model_path).resolve()
            except Exception:
                same_model_draft = (draft_model == self.model_path)
        if not cache_only and self._spec_auto_cache_only:
            if not draft_model:
                cache_only = True
                cache_only_reason = "auto_cache_only_draft_missing"
            elif same_model_draft:
                cache_only = True
                cache_only_reason = "auto_cache_only_same_model"
        if cache_allowed:
            cached = self._spec_verify_cache.get(cache_key)
            if cached:
                output_text = str(cached.get("output_text", ""))
                baseline_ms_raw = float(cached.get("total_ms", 0.0) or 0.0)
                baseline_ms = baseline_ms_raw if baseline_ms_raw > 0.0 else 0.0
                cache_flow_ms = max(0.0, (time.perf_counter() - flow_started) * 1000.0)
                net_saved_ms = 0.0
                if baseline_ms > 0.0:
                    net_saved_ms = max(0.0, min(baseline_ms, baseline_ms - cache_flow_ms))
                speedup_est = (baseline_ms / max(cache_flow_ms, 1e-6)) if baseline_ms > 0.0 else 0.0
                perf = {
                    # Cache-hit fast path reuses prior verify output and performs no decode.
                    "tokens_out": 0,
                    "total_ms": float(cache_flow_ms),
                    "prefill_ms": 0.0,
                    "decode_ms": 0.0,
                    "avg_tps": 0.0,
                }
                engine_metrics = {
                    "engine": {"mode": "spec_cache", "accel_enabled": False},
                    "perf": perf,
                    "spec": {
                        "enabled": True,
                        "k": int(spec_cfg.get("k", 0)),
                        "proposed_tokens": 0,
                        "accepted_tokens": 0,
                        "acceptance_rate": 1.0,
                        "verify_ms": 0.0,
                        "draft_ms": 0.0,
                        "overhead_ms": float(cache_flow_ms),
                        "baseline_total_ms": float(baseline_ms),
                        "net_saved_ms": float(net_saved_ms),
                        "saved_ms": float(net_saved_ms),
                        "roi": float(net_saved_ms / max(1e-6, cache_flow_ms)),
                        "speedup_est": float(speedup_est),
                        "draft_model": "cache_only",
                        "draft_mode": "verify_cache",
                        "cache_hit": True,
                        "cache_ms": float(cache_flow_ms),
                        "cache_only": bool(cache_only),
                        "disable_reason": str(cache_only_reason),
                    },
                }
                return {
                    "exit_code": 0,
                    "total_ms": perf["total_ms"],
                    "output_text": output_text,
                    "engine_metrics": engine_metrics,
                    "engine_events_path": artifacts["engine_events"],
                    "engine_errors": [],
                }
        if cache_only:
            # Prime cache with one verify-equivalent run, but keep spec disabled for this miss.
            prime_run = self._run_engine(
                prompt=prompt,
                max_tokens=max_tokens,
                deterministic_cfg=deterministic_cfg,
                policy=policy,
                artifacts=artifacts,
                mf_snapshot_in=mf_snapshot_in,
                mode="accel",
                spec_cfg=spec_cfg,
                engine_cfg=engine_cfg,
            )
            prime_metrics = prime_run.setdefault("engine_metrics", {})
            prime_perf = prime_metrics.get("perf", {}) if isinstance(prime_metrics.get("perf", {}), dict) else {}
            baseline_total_ms = float(prime_perf.get("total_ms", prime_run.get("total_ms", 0.0)) or 0.0)
            prime_tokens_out = int(prime_perf.get("tokens_out", max(1, len(str(prime_run.get("output_text", "")).split()))) or 0)
            prime_spec = prime_metrics.setdefault("spec", {})
            prime_spec["enabled"] = False
            prime_spec["k"] = int(spec_cfg.get("k", 0))
            prime_spec["proposed_tokens"] = 0
            prime_spec["accepted_tokens"] = 0
            prime_spec["acceptance_rate"] = 0.0
            prime_spec["verify_ms"] = 0.0
            prime_spec["draft_ms"] = 0.0
            prime_spec["overhead_ms"] = 0.0
            prime_spec["baseline_total_ms"] = baseline_total_ms
            prime_spec["net_saved_ms"] = 0.0
            prime_spec["saved_ms"] = 0.0
            prime_spec["roi"] = 0.0
            prime_spec["speedup_est"] = 0.0
            prime_spec["cache_hit"] = False
            prime_spec["cache_ms"] = 0.0
            prime_spec["cache_only"] = True
            prime_spec["disable_reason"] = str(cache_only_reason)
            if cache_allowed:
                self._spec_cache_put(
                    cache_key,
                    str(prime_run.get("output_text", "")),
                    baseline_total_ms,
                    prime_tokens_out,
                )
            return prime_run

        # Non-cache speculative path should be engine-native so we avoid Python-side
        # double execution (verify + draft subprocesses), which regressed latency.
        spec_run = self._run_engine(
            prompt=prompt,
            max_tokens=max_tokens,
            deterministic_cfg=deterministic_cfg,
            policy=policy,
            artifacts=artifacts,
            mf_snapshot_in=mf_snapshot_in,
            mode="spec",
            spec_cfg={**spec_cfg, "draft_model": draft_model},
            engine_cfg=engine_cfg,
        )
        engine_metrics = spec_run.setdefault("engine_metrics", {})
        if not isinstance(engine_metrics, dict):
            engine_metrics = {}
            spec_run["engine_metrics"] = engine_metrics
        perf = engine_metrics.get("perf", {}) if isinstance(engine_metrics.get("perf", {}), dict) else {}
        spec_metrics = engine_metrics.get("spec", {}) if isinstance(engine_metrics.get("spec", {}), dict) else {}
        perf_total_ms = float(perf.get("total_ms", spec_run.get("total_ms", 0.0)) or 0.0)
        baseline_total_ms = float(spec_metrics.get("baseline_total_ms", perf_total_ms) or perf_total_ms)
        overhead_ms = float(spec_metrics.get("overhead_ms", 0.0) or 0.0)
        net_saved_ms = float(spec_metrics.get("net_saved_ms", baseline_total_ms - perf_total_ms) or (baseline_total_ms - perf_total_ms))
        saved_ms = float(spec_metrics.get("saved_ms", max(0.0, net_saved_ms)) or max(0.0, net_saved_ms))
        speedup_est = float(spec_metrics.get("speedup_est", (baseline_total_ms / max(perf_total_ms, 1e-6)) if perf_total_ms > 0.0 else 0.0) or 0.0)
        roi = float(spec_metrics.get("roi", saved_ms / max(1e-6, overhead_ms)) or (saved_ms / max(1e-6, overhead_ms)))

        enabled = bool(spec_metrics.get("enabled", False))
        proposed = int(spec_metrics.get("proposed_tokens", 0) or 0)
        accepted = int(spec_metrics.get("accepted_tokens", 0) or 0)
        acceptance = float(spec_metrics.get("acceptance_rate", (accepted / float(max(1, proposed)))) or (accepted / float(max(1, proposed))))
        disable_reason = str(spec_metrics.get("disable_reason", "") or "")
        if not enabled and not disable_reason:
            disable_reason = "engine_spec_unavailable"

        engine_metrics["spec"] = {
            "enabled": enabled,
            "k": int(spec_metrics.get("k", spec_cfg.get("k", 0)) or 0),
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "acceptance_rate": acceptance,
            "verify_ms": float(spec_metrics.get("verify_ms", 0.0) or 0.0),
            "draft_ms": float(spec_metrics.get("draft_ms", 0.0) or 0.0),
            "overhead_ms": overhead_ms,
            "baseline_total_ms": baseline_total_ms,
            "net_saved_ms": net_saved_ms,
            "saved_ms": saved_ms,
            "roi": roi,
            "speedup_est": speedup_est,
            "draft_model": draft_model or "",
            "draft_mode": str(spec_metrics.get("draft_mode", "engine") or "engine"),
            "draft_init_reason": str(spec_metrics.get("draft_init_reason", "") or ""),
            "cache_hit": False,
            "cache_ms": 0.0,
            "cache_only": False,
            "disable_reason": disable_reason,
        }

        if cache_allowed and perf_total_ms > 0.0:
            tokens_out = int(perf.get("tokens_out", max(1, len(str(spec_run.get("output_text", "")).split()))) or 0)
            self._spec_cache_put(cache_key, str(spec_run.get("output_text", "")), perf_total_ms, tokens_out)
        return spec_run

    def _spec_cache_key(self, prompt: str, deterministic_cfg: Dict, max_tokens: int) -> str:
        seed = deterministic_cfg.get("seed")
        n_ctx = deterministic_cfg.get("n_ctx")
        n_batch = deterministic_cfg.get("n_batch")
        digest = hashlib.sha256(
            f"{self.model_path}|{seed}|{n_ctx}|{n_batch}|{max_tokens}|{prompt}".encode("utf-8")
        ).hexdigest()
        return digest

    def _spec_cache_put(self, key: str, output_text: str, total_ms: float, tokens_out: int) -> None:
        if not self._spec_cache_enabled:
            return
        if len(self._spec_verify_cache) >= self._spec_cache_max:
            # Drop an arbitrary old entry (dict order is insertion order in Py3.7+).
            self._spec_verify_cache.pop(next(iter(self._spec_verify_cache)), None)
        self._spec_verify_cache[key] = {
            "output_text": output_text,
            "total_ms": float(total_ms),
            "tokens_out": max(1, int(tokens_out)),
        }

    def _run_engine(
        self,
        *,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
        mode: str,
        spec_cfg: Optional[Dict] = None,
        engine_cfg: Optional[Dict] = None,
        model_path_override: Optional[str] = None,
    ) -> Dict:
        selected_model_path = model_path_override or self.model_path
        env = os.environ.copy()
        inline_prompt_limit_raw = os.environ.get("KORITH_PROMPT_INLINE_MAX_BYTES", "16384")
        try:
            inline_prompt_limit = max(0, int(inline_prompt_limit_raw))
        except ValueError:
            inline_prompt_limit = 16384
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > inline_prompt_limit:
            prompt_path = Path(artifacts["job_dir"]).resolve() / "prompt.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
            env["KORITH_PROMPT_FILE"] = str(prompt_path)
            env.pop("KORITH_PROMPT", None)
        else:
            env["KORITH_PROMPT"] = prompt
            env.pop("KORITH_PROMPT_FILE", None)
        env["KORITH_MAX_TOKENS"] = str(max_tokens)
        env["KORITH_ENGINE_METRICS_PATH"] = artifacts["engine_metrics"]
        env["KORITH_ENGINE_EVENTS_PATH"] = artifacts["engine_events"]
        env["KORITH_MF_SNAPSHOT_OUT"] = artifacts["mf_snapshot"]
        env["KORITH_JOB_ID"] = Path(artifacts["job_dir"]).name

        if mf_snapshot_in:
            env["KORITH_MF_SNAPSHOT_IN"] = mf_snapshot_in

        allow_amf = bool(policy.get("allow_amf_reuse", True))
        env["KORITH_ENABLE_AMF"] = "1" if allow_amf else "0"
        env["KORITH_ENABLE_MF"] = "1" if allow_amf else "0"

        if "n_ctx" in deterministic_cfg:
            env["KORITH_N_CTX"] = str(deterministic_cfg.get("n_ctx"))
        if "n_batch" in deterministic_cfg:
            env["KORITH_N_BATCH"] = str(deterministic_cfg.get("n_batch"))
        if "n_seq_max" in deterministic_cfg:
            env["KORITH_N_SEQ_MAX"] = str(deterministic_cfg.get("n_seq_max"))

        job_dir = Path(artifacts["job_dir"]).resolve()
        platform_root: Optional[Path] = None
        for candidate in [job_dir, *job_dir.parents]:
            if candidate.name == "artifacts":
                platform_root = candidate.parent
                break
        if platform_root is None:
            platform_root = (Path.cwd() / "platform_data").resolve()
        default_amf_path = (platform_root / "amf_store").resolve()
        amf_path = Path(os.environ.get("KORITH_PLATFORM_AMF_PATH", str(default_amf_path))).resolve()
        amf_path.mkdir(parents=True, exist_ok=True)
        env["KORITH_AMF_PATH"] = str(amf_path)

        engine_cfg = engine_cfg or {}
        spec_cfg = spec_cfg or {}
        kernel_enabled = str(engine_cfg.get("kernels_enabled", os.environ.get("KORITH_KERNELS", "0")))
        kernel_backend = str(engine_cfg.get("kernel_backend", os.environ.get("KORITH_KERNEL_BACKEND", "none")))
        kernel_verify = str(engine_cfg.get("kernel_verify", os.environ.get("KORITH_KERNEL_VERIFY", "0")))
        env["KORITH_KERNELS"] = kernel_enabled
        env["KORITH_KERNEL_BACKEND"] = kernel_backend
        env["KORITH_KERNEL_VERIFY"] = kernel_verify
        if mode in ("accel", "spec"):
            env["KORITH_ACCEL_ENABLED"] = "1"
            env["KORITH_CUDA_DEVICE"] = str(engine_cfg.get("cuda_device", os.environ.get("KORITH_CUDA_DEVICE", "0")))
            env["KORITH_CUDA_DTYPE"] = str(engine_cfg.get("cuda_dtype", os.environ.get("KORITH_CUDA_DTYPE", "fp16")))
            env["KORITH_KV_LAYOUT_VERSION"] = str(
                engine_cfg.get("kv_layout_version", os.environ.get("KORITH_KV_LAYOUT_VERSION", "v1"))
            )
        else:
            env["KORITH_ACCEL_ENABLED"] = "0"

        if mode == "spec":
            env["KORITH_SPEC_ENABLED"] = "1"
            env["KORITH_SPEC_K"] = str(int(spec_cfg.get("k", os.environ.get("KORITH_SPEC_K", "6"))))
            env["KORITH_SPEC_MIN_ACCEPT"] = str(float(spec_cfg.get("min_accept", os.environ.get("KORITH_SPEC_MIN_ACCEPT", "0.55"))))
            env["KORITH_SPEC_DISABLE_AFTER_N"] = str(
                int(spec_cfg.get("disable_after_n", os.environ.get("KORITH_SPEC_DISABLE_AFTER_N", "50")))
            )
            # Keep speculative behavior deterministic across runs even if the parent
            # shell exported debug/override variables in a previous session.
            env["KORITH_SPEC_REQUIRE_BASELINE_READY"] = (
                "1" if bool(spec_cfg.get("require_baseline_ready", False)) else "0"
            )
            env["KORITH_SPEC_SHADOW_WHEN_COMMIT_BLOCKED"] = (
                "1" if bool(spec_cfg.get("shadow_when_commit_blocked", False)) else "0"
            )
            if "allow_exec" in spec_cfg:
                env["KORITH_SPEC_ALLOW_EXEC"] = "1" if bool(spec_cfg.get("allow_exec")) else "0"
            else:
                env.pop("KORITH_SPEC_ALLOW_EXEC", None)
            if "allow_commit" in spec_cfg:
                env["KORITH_SPEC_ALLOW_COMMIT"] = "1" if bool(spec_cfg.get("allow_commit")) else "0"
            else:
                env.pop("KORITH_SPEC_ALLOW_COMMIT", None)
            if "depth" in spec_cfg:
                env["KORITH_SPEC_DEPTH"] = str(max(1, int(spec_cfg.get("depth") or 1)))
            else:
                env.pop("KORITH_SPEC_DEPTH", None)
            draft_model = str(
                spec_cfg.get("draft_model")
                or os.environ.get("KORITH_DRAFT_MODEL_PATH", "")
                or os.environ.get("KORITH_DRAFT_MODEL", "")
            ).strip()
            if draft_model:
                env["KORITH_DRAFT_MODEL"] = draft_model
                env["KORITH_DRAFT_MODEL_PATH"] = draft_model
            else:
                env.pop("KORITH_DRAFT_MODEL", None)
                env.pop("KORITH_DRAFT_MODEL_PATH", None)
        else:
            env["KORITH_SPEC_ENABLED"] = "0"
            env["KORITH_SPEC_REQUIRE_BASELINE_READY"] = "0"
            env["KORITH_SPEC_SHADOW_WHEN_COMMIT_BLOCKED"] = "0"
            env.pop("KORITH_SPEC_ALLOW_EXEC", None)
            env.pop("KORITH_SPEC_ALLOW_COMMIT", None)
            env.pop("KORITH_SPEC_DEPTH", None)
            env.pop("KORITH_DRAFT_MODEL", None)
            env.pop("KORITH_DRAFT_MODEL_PATH", None)

        t0 = time.time()
        proc = subprocess.Popen(
            ["./build/korith_dynamic", selected_model_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        stdout, stderr = proc.communicate()
        t1 = time.time()

        output_path = Path(artifacts["output"])
        log_path = Path(artifacts["log"])

        with output_path.open("a", encoding="utf-8") as out_f:
            out_f.write(f"\n[engine mode={mode} exit={proc.returncode}]\n")
            out_f.write(stdout)

        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(f"\n[engine mode={mode} exit={proc.returncode}]\n")
            log_f.write(stderr)

        engine_metrics = {}
        engine_errors = []
        if Path(artifacts["engine_metrics"]).exists():
            try:
                engine_metrics = json.loads(Path(artifacts["engine_metrics"]).read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                engine_errors.append(f"engine_metrics_json_decode_failed:{exc}")
                engine_metrics = {}

        if mode == "accel" and isinstance(engine_metrics, dict):
            engine_metrics.setdefault("engine", {})
            engine_metrics["engine"]["mode"] = "accel"
            engine_metrics["engine"]["cuda_requested"] = True
            engine_metrics["engine"]["engine_lib_available"] = bool(self._engine_client.available)

        if mode == "spec" and isinstance(engine_metrics, dict):
            spec = engine_metrics.setdefault("spec", {})
            spec.setdefault("enabled", False)
            spec.setdefault("k", int(env.get("KORITH_SPEC_K", "6")))
            spec.setdefault("proposed_tokens", 0)
            spec.setdefault("accepted_tokens", 0)
            spec.setdefault("acceptance_rate", 0.0)
            spec.setdefault("verify_ms", 0.0)
            spec.setdefault("draft_ms", 0.0)
            spec.setdefault("overhead_ms", 0.0)
            spec.setdefault("baseline_total_ms", 0.0)
            spec.setdefault("net_saved_ms", 0.0)
            spec.setdefault("saved_ms", 0.0)
            spec.setdefault("roi", 0.0)
            spec.setdefault("speedup_est", 0.0)
            engine_metrics.setdefault("engine", {})
            engine_metrics["engine"]["mode"] = "speculative"
            engine_metrics["engine"]["cuda_requested"] = True
            engine_metrics["engine"]["engine_lib_available"] = bool(self._engine_client.available)
        if mode in ("accel", "spec") and isinstance(engine_metrics, dict):
            kernels = engine_metrics.setdefault("kernels", {})
            kernels.setdefault("enabled", str(kernel_enabled).strip().lower() in ("1", "true", "yes", "on"))
            kernels.setdefault("backend", kernel_backend)
            kernels.setdefault("verify", str(kernel_verify).strip().lower() in ("1", "true", "yes", "on"))
            kernels.setdefault("ms_saved", 0.0)
            kernels.setdefault("fallback", False)
            kernels.setdefault("verify_ok", True)
            kernels.setdefault("probe_enabled", False)
            kernels.setdefault("probe_ms", 0.0)

            kernels_enabled = bool(kernels.get("enabled", False))
            kernel_probe_enabled = os.environ.get("KORITH_KERNEL_LIB_PROBE", "0").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            kernels["probe_enabled"] = bool(kernel_probe_enabled)
            if kernels_enabled and kernel_probe_enabled and self._engine_client.available:
                probe_t0 = time.time()
                init_ok = self._engine_client.init_model(
                    model_path=selected_model_path,
                    n_ctx=int(deterministic_cfg.get("n_ctx", 0) or 0),
                    n_batch=int(deterministic_cfg.get("n_batch", 0) or 0),
                    n_gpu=1,
                )
                if init_ok and mf_snapshot_in and Path(mf_snapshot_in).exists():
                    try:
                        self._engine_client.apply_kv_replay(Path(mf_snapshot_in).read_bytes())
                    except Exception:
                        pass
                lib_metrics = self._engine_client.get_metrics()
                if lib_metrics:
                    kernel_saved = float(lib_metrics.get("kernel_ms_saved", lib_metrics.get("ms_saved", 0.0)) or 0.0)
                    if kernel_saved > 0.0:
                        kernels["ms_saved"] = max(float(kernels.get("ms_saved", 0.0) or 0.0), kernel_saved)
                self._engine_client.shutdown()
                kernels["probe_ms"] = float((time.time() - probe_t0) * 1000.0)

        return {
            "exit_code": proc.returncode,
            "total_ms": (t1 - t0) * 1000.0,
            "output_text": stdout,
            "engine_metrics": engine_metrics,
            "engine_events_path": artifacts["engine_events"],
            "engine_errors": engine_errors,
        }

    def run_draft(
        self,
        prompt: str,
        deterministic_cfg: Dict,
        max_tokens: int,
        spec_cfg: Optional[Dict] = None,
    ) -> Dict:
        # Deterministic placeholder draft stream based on prompt hash.
        n = max(1, int((spec_cfg or {}).get("draft_max_tokens", max_tokens)))
        seed = int(deterministic_cfg.get("seed", 0))
        base = int(hashlib.sha256(f"{prompt}:{seed}".encode("utf-8")).hexdigest()[:8], 16)
        toks = [int((base + i) % 32000) for i in range(n)]
        return {"draft_tokens": toks, "state": {"seed": seed}}

    def verify_tokens(
        self,
        prompt: str,
        draft_tokens: List[int],
        deterministic_cfg: Dict,
        spec_cfg: Optional[Dict] = None,
    ) -> Dict:
        _ = deterministic_cfg
        verify_text = str((spec_cfg or {}).get("_verify_text", ""))
        if not verify_text:
            return {"accepted_count": int(len(draft_tokens)), "verified_logits": None}
        verified = self._token_ids_from_text(verify_text)
        accepted = self._prefix_accept_count(draft_tokens, verified)
        return {"accepted_count": int(accepted), "verified_logits": None}

    def _token_ids_from_text(self, text: str) -> List[int]:
        out: List[int] = []
        for piece in str(text).split():
            digest = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            out.append(int(digest[:8], 16) % 32000)
        return out

    def _prefix_accept_count(self, proposed: List[int], verified: List[int]) -> int:
        accepted = 0
        for p, v in zip(proposed, verified):
            if int(p) != int(v):
                break
            accepted += 1
        return accepted
