from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
import os
import time
import queue
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.adapters.base import BackendAdapter, Capabilities
from platform.adapters.korith_local import Tier1LocalKorithAdapter
from platform.engine.snapshot import (
    snapshot_meta_path,
    validate_snapshot_metadata,
    write_snapshot_metadata,
)
from platform.kernels.registry import resolve_kernel_backend
from platform.ledger.store import SQLiteLedgerStore
from platform.runtime.cluster_worker import (
    ClusterWorker,
    apply_decode_opt_profile_env_defaults,
    enforce_metrics_invariants,
)


class _MockPhase6Adapter(BackendAdapter):
    backend_id = "mock_phase6"
    backend_version = "v1"

    def get_fingerprint(self):
        return {
            "model_hash": "m1",
            "tokenizer_hash": "t1",
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
        }

    def get_capabilities(self):
        return Capabilities(
            kv_replay=True,
            deterministic_seeding=True,
            streaming=False,
            batch_prefill=True,
            logits_access=True,
            verify_tokens=True,
            draft_supported=True,
        )

    def tokenize(self, prompt: str) -> int:
        return max(1, len(prompt.split()))

    def run_baseline(self, prompt, max_tokens, deterministic_cfg, policy, artifacts, mf_snapshot_in):
        tokens = ["A", "B", "C", "D"][: max(1, min(max_tokens, 4))]
        return {
            "exit_code": 0,
            "output_text": " ".join(tokens),
            "total_ms": 10.0,
            "engine_metrics": {
                "amf": {
                    "supported": True,
                    "decision": "hit",
                    "restore_ms": 2.0,
                    "baseline_prefix_ms": 15.0,
                },
                "mf": {"supported": True},
                "perf": {"tokens_out": len(tokens), "total_ms": 10.0, "prefill_ms": 3.0, "decode_ms": 7.0, "avg_tps": 400.0},
            },
        }

    def run_accel(self, prompt, max_tokens, deterministic_cfg, policy, artifacts, mf_snapshot_in, engine_cfg=None):
        out = self.run_baseline(prompt, max_tokens, deterministic_cfg, policy, artifacts, mf_snapshot_in)
        out["engine_metrics"]["engine"] = {"mode": "accel", "accel_enabled": True, "cuda_device": 0, "kv_layout_version": "v1"}
        return out

    def run_speculative(self, prompt, max_tokens, deterministic_cfg, policy, artifacts, mf_snapshot_in, spec_cfg=None, engine_cfg=None):
        verify = self.run_accel(prompt, max_tokens, deterministic_cfg, policy, artifacts, mf_snapshot_in, engine_cfg=engine_cfg)
        verify["engine_metrics"]["engine"]["mode"] = "speculative"
        verify["engine_metrics"]["spec"] = {
            "enabled": True,
            "k": int((spec_cfg or {}).get("k", 6)),
            "proposed_tokens": 4,
            "accepted_tokens": 4,
            "acceptance_rate": 1.0,
            "verify_ms": 4.0,
            "draft_ms": 1.0,
            "speedup_est": 1.2,
        }
        return verify


class _CacheOnlySpecAdapter(Tier1LocalKorithAdapter):
    def __init__(self) -> None:
        super().__init__(model_path="models/llama-3.2-1b-q4_k_m.gguf")

    def _run_engine(self, **kwargs):
        return {
            "exit_code": 0,
            # Deliberately much larger than perf.total_ms to verify cache baseline source.
            "total_ms": 2200.0,
            "output_text": "alpha beta gamma",
            "engine_metrics": {
                "perf": {
                    "tokens_out": 64,
                    "total_ms": 860.0,
                    "prefill_ms": 0.0,
                    "decode_ms": 860.0,
                    "avg_tps": 300.0,
                },
                "amf": {"supported": True, "decision": "hit"},
                "mf": {"supported": True},
            },
            "engine_events_path": "",
            "engine_errors": [],
        }


class Phase6Tests(unittest.TestCase):
    def test_korith_local_is_spec_capable(self):
        adapter = Tier1LocalKorithAdapter(model_path="models/llama-3.2-1b-q4_k_m.gguf")
        caps = adapter.get_capabilities()
        self.assertTrue(caps.verify_tokens)
        self.assertTrue(caps.draft_supported)

    def test_decode_opt_profile_sets_default_runtime_toggles(self):
        keys = (
            "KORITH_DECODE_OPT_PROFILE",
            "KORITH_ACCEL_ENABLED",
            "KORITH_KERNELS",
            "KORITH_KERNEL_BACKEND",
            "KORITH_KERNEL_VERIFY",
            "KORITH_DECODE_CACHE_ENABLED",
            "KORITH_DECODE_CACHE_REQUIRE_DETERMINISTIC",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_DECODE_OPT_PROFILE"] = "1"
            for k in keys:
                if k == "KORITH_DECODE_OPT_PROFILE":
                    continue
                os.environ.pop(k, None)
            enabled = apply_decode_opt_profile_env_defaults()
            self.assertTrue(enabled)
            self.assertEqual(os.environ.get("KORITH_ACCEL_ENABLED"), "1")
            self.assertEqual(os.environ.get("KORITH_KERNELS"), "1")
            self.assertEqual(os.environ.get("KORITH_KERNEL_BACKEND"), "cuda")
            self.assertEqual(os.environ.get("KORITH_KERNEL_VERIFY"), "0")
            self.assertEqual(os.environ.get("KORITH_DECODE_CACHE_ENABLED"), "1")
            self.assertEqual(os.environ.get("KORITH_DECODE_CACHE_REQUIRE_DETERMINISTIC"), "1")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_korith_local_large_prompt_uses_prompt_file_env(self):
        adapter = Tier1LocalKorithAdapter(model_path="models/llama-3.2-1b-q4_k_m.gguf")
        prompt = " ".join(["token"] * 64)
        captured: dict = {}

        class _FakeProc:
            def __init__(self, *args, **kwargs):
                captured["env"] = dict(kwargs.get("env", {}))
                metrics_path = Path(captured["env"]["KORITH_ENGINE_METRICS_PATH"])
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(
                    json.dumps(
                        {
                            "perf": {
                                "tokens_out": 1,
                                "total_ms": 1.0,
                                "prefill_ms": 0.2,
                                "decode_ms": 0.8,
                                "avg_tps": 1.0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                self.returncode = 0

            def communicate(self):
                return ("ok", "")

        old_limit = os.environ.get("KORITH_PROMPT_INLINE_MAX_BYTES")
        try:
            os.environ["KORITH_PROMPT_INLINE_MAX_BYTES"] = "16"
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                artifacts = {
                    "engine_metrics": str(tdp / "engine_metrics.json"),
                    "engine_events": str(tdp / "engine_events.jsonl"),
                    "mf_snapshot": str(tdp / "mf_snapshot.bin"),
                    "job_dir": str(tdp / "job"),
                    "output": str(tdp / "output.txt"),
                    "log": str(tdp / "run.log"),
                }
                Path(artifacts["job_dir"]).mkdir(parents=True, exist_ok=True)
                with patch("platform.adapters.korith_local.subprocess.Popen", side_effect=_FakeProc):
                    out = adapter._run_engine(
                        prompt=prompt,
                        max_tokens=4,
                        deterministic_cfg={"seed": 1, "n_ctx": 1024, "n_batch": 64},
                        policy={"allow_amf_reuse": True, "allow_spec": False},
                        artifacts=artifacts,
                        mf_snapshot_in=None,
                        mode="baseline",
                    )
                self.assertEqual(int(out["exit_code"]), 0)
                self.assertIn("env", captured)
                env = captured["env"]
                self.assertNotIn("KORITH_PROMPT", env)
                self.assertIn("KORITH_PROMPT_FILE", env)
                prompt_path = Path(str(env["KORITH_PROMPT_FILE"]))
                self.assertTrue(prompt_path.exists())
                self.assertEqual(prompt_path.read_text(encoding="utf-8"), prompt)
        finally:
            if old_limit is None:
                os.environ.pop("KORITH_PROMPT_INLINE_MAX_BYTES", None)
            else:
                os.environ["KORITH_PROMPT_INLINE_MAX_BYTES"] = old_limit

    def test_korith_local_small_prompt_uses_inline_prompt_env(self):
        adapter = Tier1LocalKorithAdapter(model_path="models/llama-3.2-1b-q4_k_m.gguf")
        prompt = "short prompt"
        captured: dict = {}

        class _FakeProc:
            def __init__(self, *args, **kwargs):
                captured["env"] = dict(kwargs.get("env", {}))
                metrics_path = Path(captured["env"]["KORITH_ENGINE_METRICS_PATH"])
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(
                    json.dumps(
                        {
                            "perf": {
                                "tokens_out": 1,
                                "total_ms": 1.0,
                                "prefill_ms": 0.2,
                                "decode_ms": 0.8,
                                "avg_tps": 1.0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                self.returncode = 0

            def communicate(self):
                return ("ok", "")

        old_limit = os.environ.get("KORITH_PROMPT_INLINE_MAX_BYTES")
        try:
            os.environ["KORITH_PROMPT_INLINE_MAX_BYTES"] = "4096"
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                artifacts = {
                    "engine_metrics": str(tdp / "engine_metrics.json"),
                    "engine_events": str(tdp / "engine_events.jsonl"),
                    "mf_snapshot": str(tdp / "mf_snapshot.bin"),
                    "job_dir": str(tdp / "job"),
                    "output": str(tdp / "output.txt"),
                    "log": str(tdp / "run.log"),
                }
                Path(artifacts["job_dir"]).mkdir(parents=True, exist_ok=True)
                with patch("platform.adapters.korith_local.subprocess.Popen", side_effect=_FakeProc):
                    out = adapter._run_engine(
                        prompt=prompt,
                        max_tokens=4,
                        deterministic_cfg={"seed": 1, "n_ctx": 1024, "n_batch": 64},
                        policy={"allow_amf_reuse": True, "allow_spec": False},
                        artifacts=artifacts,
                        mf_snapshot_in=None,
                        mode="baseline",
                    )
            self.assertEqual(int(out["exit_code"]), 0)
            self.assertIn("env", captured)
            env = captured["env"]
            self.assertIn("KORITH_PROMPT", env)
            self.assertEqual(str(env["KORITH_PROMPT"]), prompt)
            self.assertNotIn("KORITH_PROMPT_FILE", env)
        finally:
            if old_limit is None:
                os.environ.pop("KORITH_PROMPT_INLINE_MAX_BYTES", None)
            else:
                os.environ["KORITH_PROMPT_INLINE_MAX_BYTES"] = old_limit

    def test_deterministic_baseline_vs_accel_exact_tokens(self):
        adapter = _MockPhase6Adapter()
        args = {
            "prompt": "deterministic input",
            "max_tokens": 4,
            "deterministic_cfg": {"seed": 123, "n_ctx": 8192, "n_batch": 512},
            "policy": {"allow_amf_reuse": True, "allow_spec": False},
            "artifacts": {},
            "mf_snapshot_in": None,
        }
        base = adapter.run_baseline(**args)
        accel = adapter.run_accel(**args, engine_cfg={"cuda_device": 0, "kv_layout_version": "v1"})
        self.assertEqual(base["output_text"], accel["output_text"])

    def test_snapshot_export_import_metadata_validation(self):
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "mf_snapshot.bin"
            snap.write_bytes(b"\x00\x01\x02snapshot")
            write_snapshot_metadata(
                snap,
                fingerprint_hash="fp1",
                model_hash="m1",
                tokenizer_hash="t1",
                backend_id="korith_cuda",
                n_ctx=8192,
                kv_layout_version="v1",
                created_at="2026-02-07T00:00:00Z",
            )
            ok, reason, _ = validate_snapshot_metadata(
                snap,
                fingerprint_hash="fp1",
                model_hash="m1",
                tokenizer_hash="t1",
                kv_layout_version="v1",
                n_ctx=8192,
            )
            self.assertTrue(ok, msg=reason)

            bad, bad_reason, _ = validate_snapshot_metadata(
                snap,
                fingerprint_hash="fp1",
                model_hash="m1",
                tokenizer_hash="t1",
                kv_layout_version="v2",
                n_ctx=8192,
            )
            self.assertFalse(bad)
            self.assertEqual(bad_reason, "snapshot_layout_mismatch")

    def test_snapshot_cache_copy_keeps_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            source = base / "source" / "mf_snapshot.bin"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"\x01\x02\x03snapshot")
            write_snapshot_metadata(
                source,
                fingerprint_hash="fp-cache",
                model_hash="m-cache",
                tokenizer_hash="t-cache",
                backend_id="korith_local",
                n_ctx=65536,
                kv_layout_version="v1",
                created_at="2026-02-15T00:00:00Z",
            )

            worker = ClusterWorker.__new__(ClusterWorker)
            worker._snapshot_vram_max_bytes = 10 * 1024 * 1024
            worker._snapshot_ram_max_bytes = 20 * 1024 * 1024
            worker._snapshot_vram_cache_max_bytes = 200 * 1024 * 1024
            worker._snapshot_ram_cache_max_bytes = 200 * 1024 * 1024
            worker._snapshot_nvme_cache_max_bytes = 200 * 1024 * 1024
            worker._snapshot_vram_dir = base / "cache" / "vram"
            worker._snapshot_ram_dir = base / "cache" / "ram"
            worker._snapshot_nvme_dir = base / "cache" / "nvme"
            worker._snapshot_vram_dir.mkdir(parents=True, exist_ok=True)
            worker._snapshot_ram_dir.mkdir(parents=True, exist_ok=True)
            worker._snapshot_nvme_dir.mkdir(parents=True, exist_ok=True)

            cached_path, _, _ = ClusterWorker._cache_snapshot_from_file(worker, "fp-cache", source)
            self.assertTrue(snapshot_meta_path(cached_path).exists())
            ok, reason, _ = validate_snapshot_metadata(
                cached_path,
                fingerprint_hash="fp-cache",
                model_hash="m-cache",
                tokenizer_hash="t-cache",
                kv_layout_version="v1",
                n_ctx=65536,
            )
            self.assertTrue(ok, msg=reason)

    def test_speculative_verify_matches_verify_only(self):
        adapter = _MockPhase6Adapter()
        verify_only = adapter.run_accel(
            prompt="verify me",
            max_tokens=4,
            deterministic_cfg={"seed": 123, "n_ctx": 8192, "n_batch": 512},
            policy={"allow_amf_reuse": True, "allow_spec": True},
            artifacts={},
            mf_snapshot_in=None,
            engine_cfg={"cuda_device": 0, "kv_layout_version": "v1"},
        )
        spec = adapter.run_speculative(
            prompt="verify me",
            max_tokens=4,
            deterministic_cfg={"seed": 123, "n_ctx": 8192, "n_batch": 512},
            policy={"allow_amf_reuse": True, "allow_spec": True},
            artifacts={},
            mf_snapshot_in=None,
            spec_cfg={"k": 6},
            engine_cfg={"cuda_device": 0, "kv_layout_version": "v1"},
        )
        self.assertEqual(verify_only["output_text"], spec["output_text"])
        self.assertTrue(spec["engine_metrics"]["spec"]["enabled"])

    def test_perf_sanity_restore_less_than_baseline_prefix(self):
        adapter = _MockPhase6Adapter()
        out = adapter.run_baseline(
            prompt="hit path",
            max_tokens=4,
            deterministic_cfg={"seed": 1, "n_ctx": 8192, "n_batch": 512},
            policy={"allow_amf_reuse": True, "allow_spec": False},
            artifacts={},
            mf_snapshot_in=None,
        )
        amf = out["engine_metrics"]["amf"]
        self.assertLess(float(amf["restore_ms"]), float(amf["baseline_prefix_ms"]))

    def test_kernel_registry_selection(self):
        old = {k: os.environ.get(k) for k in ("KORITH_KERNELS", "KORITH_KERNEL_BACKEND", "KORITH_KERNEL_VERIFY")}
        try:
            os.environ["KORITH_KERNELS"] = "0"
            os.environ["KORITH_KERNEL_BACKEND"] = "none"
            os.environ["KORITH_KERNEL_VERIFY"] = "1"
            noop = resolve_kernel_backend()
            ctx = noop.context()
            self.assertFalse(ctx.enabled)
            self.assertEqual(ctx.backend, "none")
            self.assertTrue(ctx.verify)

            os.environ["KORITH_KERNELS"] = "1"
            os.environ["KORITH_KERNEL_BACKEND"] = "cuda"
            cuda = resolve_kernel_backend()
            cuda_ctx = cuda.context()
            self.assertEqual(cuda_ctx.backend, "cuda")
            self.assertTrue(cuda_ctx.enabled)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_spec_governance_cooldown_transitions(self):
        with tempfile.TemporaryDirectory() as td:
            store = SQLiteLedgerStore(Path(td) / "ledger.sqlite")
            store.init()
            fp = "fp_phase6"
            org = "org_phase6"
            store.upsert_spec_governance(
                fingerprint_hash=fp,
                org_id=org,
                spec_disabled=0,
                reason=None,
                cooldown_until=0.0,
                bad_accept_streak=1,
                updated_at="2026-02-07T00:00:00Z",
            )
            first = store.get_spec_governance(fp)
            self.assertIsNotNone(first)
            self.assertEqual(int(first.get("bad_accept_streak", 0)), 1)

            store.upsert_spec_governance(
                fingerprint_hash=fp,
                org_id=org,
                spec_disabled=1,
                reason="low_acceptance",
                cooldown_until=1234.5,
                bad_accept_streak=5,
                updated_at="2026-02-07T00:00:01Z",
            )
            second = store.get_spec_governance(fp)
            self.assertIsNotNone(second)
            self.assertEqual(int(second.get("spec_disabled", 0)), 1)
            self.assertEqual(str(second.get("reason", "")), "low_acceptance")
            self.assertGreater(float(second.get("cooldown_until", 0.0) or 0.0), 0.0)
            rows = store.list_spec_governance(limit=10, org_id=org)
            self.assertGreaterEqual(len(rows), 1)

    def test_metrics_schema_additive_fields_present(self):
        metrics = {
            "ids": {"job_id": "j1", "created_at": "a", "started_at": "b", "finished_at": "c"},
            "backend": {"backend_id": "korith_cuda", "backend_version": "v1"},
            "model": {"model_id": "m", "model_hash": "mh", "tokenizer_hash": "th"},
            "input": {"prompt_hash": "ph", "sampling_hash": "sh", "prompt_tokens": 10},
            "scheduling": {"worker_id": "w0", "node_id": "n0", "gpu_id": 0, "lane": "SPEC_HIT", "queue_latency_ms": 1.0},
            "amf": {"supported": True, "decision": "hit", "prefix_len": 5, "skipped_tokens": 5, "skip_ratio": 1.0, "restore_ms": 1.0, "baseline_prefix_ms": 2.0, "saved_ms": 1.0, "roi": 1.0},
            "mf": {"supported": True, "min_admit_roi": 1.2, "eviction_pressure": 0.8, "replay_disable_mask": 0, "cooldown_ms": 0, "snapshot_id": "s1"},
            "engine": {"mode": "speculative", "accel_enabled": True, "cuda_device": 0, "kv_layout_version": "v1"},
            "kernels": {
                "enabled": True,
                "backend": "cuda",
                "kernels_applied": True,
                "verify": True,
                "verify_ok": True,
                "fallback": False,
                "decode_ms_actual": 5.0,
                "decode_ms_baseline_est": 8.0,
                "comparable": True,
                "comparable_tag": "shadow_baseline",
                "ms_saved": 3.0,
            },
            "spec": {
                "supported": True,
                "enabled": True,
                "k": 6,
                "proposed_tokens": 16,
                "accepted_tokens": 12,
                "acceptance_rate": 0.75,
                "verify_ms": 2.0,
                "draft_ms": 1.0,
                "overhead_ms": 3.0,
                "baseline_total_ms": 20.0,
                "net_saved_ms": 2.0,
                "saved_ms": 5.0,
                "roi": 1.5,
                "speedup_est": 1.2,
                "cache_hit": False,
                "cache_ms": 0.0,
                "cache_only": False,
                "policy_reason": "active",
                "shape_key": "shape-1",
            },
            "savings": {
                "prefill_saved_ms": 1.0,
                "spec_saved_ms": 2.0,
                "kernels_saved_ms": 3.0,
                "total_saved_ms": 6.0,
            },
            "perf": {"tokens_out": 32, "total_ms": 10.0, "prefill_ms": 2.0, "decode_ms": 8.0, "avg_tps": 3.2},
            "health": {"line": "[KORITH_HEALTH] ..."},
            "errors": [],
        }
        self.assertIn("kernels", metrics)
        self.assertIn("spec", metrics)
        self.assertEqual(metrics["scheduling"]["lane"], "SPEC_HIT")
        self.assertIn("saved_ms", metrics["spec"])
        self.assertIn("overhead_ms", metrics["spec"])
        self.assertIn("baseline_total_ms", metrics["spec"])
        self.assertIn("net_saved_ms", metrics["spec"])
        self.assertIn("roi", metrics["spec"])
        self.assertIn("policy_reason", metrics["spec"])
        self.assertIn("shape_key", metrics["spec"])
        self.assertIn("kernels_applied", metrics["kernels"])
        self.assertIn("decode_ms_actual", metrics["kernels"])
        self.assertIn("decode_ms_baseline_est", metrics["kernels"])
        self.assertIn("savings", metrics)
        self.assertIn("total_saved_ms", metrics["savings"])

    def test_spec_cache_only_uses_perf_baseline_and_cached_tokens(self):
        old = {k: os.environ.get(k) for k in ("KORITH_SPEC_VERIFY_CACHE", "KORITH_SPEC_AUTO_CACHE_ONLY")}
        try:
            os.environ["KORITH_SPEC_VERIFY_CACHE"] = "1"
            os.environ["KORITH_SPEC_AUTO_CACHE_ONLY"] = "1"
            adapter = _CacheOnlySpecAdapter()
            deterministic_cfg = {"seed": 123, "n_ctx": 8192, "n_batch": 512}
            policy = {"allow_amf_reuse": True, "allow_spec": True}
            artifacts = {
                "engine_metrics": "/tmp/engine_metrics.json",
                "engine_events": "/tmp/engine_events.jsonl",
                "mf_snapshot": "/tmp/mf_snapshot.json",
                "job_dir": "/tmp/job_dir",
                "output": "/tmp/output.txt",
                "log": "/tmp/run.log",
            }
            spec_cfg = {"cache_only": True, "k": 2}

            first = adapter.run_speculative(
                prompt="same prompt",
                max_tokens=64,
                deterministic_cfg=deterministic_cfg,
                policy=policy,
                artifacts=artifacts,
                mf_snapshot_in=None,
                spec_cfg=spec_cfg,
                engine_cfg={},
            )
            first_spec = first["engine_metrics"]["spec"]
            self.assertFalse(first_spec["enabled"])
            self.assertEqual(float(first_spec["baseline_total_ms"]), 860.0)

            second = adapter.run_speculative(
                prompt="same prompt",
                max_tokens=64,
                deterministic_cfg=deterministic_cfg,
                policy=policy,
                artifacts=artifacts,
                mf_snapshot_in=None,
                spec_cfg=spec_cfg,
                engine_cfg={},
            )
            second_spec = second["engine_metrics"]["spec"]
            self.assertTrue(second_spec["enabled"])
            self.assertTrue(second_spec["cache_hit"])
            self.assertEqual(float(second_spec["baseline_total_ms"]), 860.0)
            self.assertGreaterEqual(float(second_spec["net_saved_ms"]), 0.0)
            self.assertLessEqual(float(second_spec["net_saved_ms"]), float(second_spec["baseline_total_ms"]))
            self.assertEqual(int(second["engine_metrics"]["perf"]["tokens_out"]), 0)
            self.assertEqual(float(second["engine_metrics"]["perf"]["decode_ms"]), 0.0)
            self.assertGreaterEqual(float(second["engine_metrics"]["perf"]["total_ms"]), 0.0)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_cache_only_invariants_zero_kernel_credit_without_decode(self):
        em = {
            "perf": {
                "tokens_out": 1,
                "total_ms": 0.001,
                "prefill_ms": 0.0,
                "decode_ms": 0.5,
            },
            "spec": {
                "cache_only": True,
                "cache_hit": True,
                "baseline_total_ms": 1200.0,
                "net_saved_ms": 4000.0,
                "saved_ms": 4000.0,
            },
            "kernels": {
                "enabled": True,
                "backend": "cuda",
                "ms_saved": 333.0,
            },
        }
        enforce_metrics_invariants(em, no_decode_cache_hit=True)
        self.assertEqual(int(em["perf"]["tokens_out"]), 0)
        self.assertEqual(float(em["perf"]["decode_ms"]), 0.0)
        self.assertEqual(float(em["kernels"]["ms_saved"]), 0.0)
        self.assertFalse(bool(em["kernels"]["enabled"]))
        self.assertGreaterEqual(float(em["spec"]["net_saved_ms"]), 0.0)
        self.assertLessEqual(float(em["spec"]["net_saved_ms"]), 1200.0)

    def test_kernel_policy_cooldown_after_repeated_low_savings(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._kernel_policy_enabled = True
        worker._kernel_policy_min_saved_ms = 2.0
        worker._kernel_policy_bad_streak_max = 2
        worker._kernel_policy_cooldown_ms = 60000
        worker._kernel_policy_probe_every = 8
        worker._kernel_policy_state = {}

        allowed_first, reason_first = ClusterWorker._kernel_policy_allow(worker, shape_key="shape-1", lane="MISS")
        self.assertTrue(allowed_first)
        self.assertEqual(reason_first, "active")

        low_savings_metrics = {
            "kernels": {
                "kernels_applied": True,
                "comparable": True,
                "fallback": False,
                "ms_saved": 0.25,
            }
        }
        ClusterWorker._update_kernel_policy_state(worker, shape_key="shape-1", engine_metrics=low_savings_metrics)
        ClusterWorker._update_kernel_policy_state(worker, shape_key="shape-1", engine_metrics=low_savings_metrics)

        allowed_second, reason_second = ClusterWorker._kernel_policy_allow(worker, shape_key="shape-1", lane="MISS")
        self.assertFalse(allowed_second)
        self.assertEqual(reason_second, "cooldown")

    def test_kernel_policy_allows_probe_during_cooldown(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._kernel_policy_enabled = True
        worker._kernel_policy_min_saved_ms = 2.0
        worker._kernel_policy_bad_streak_max = 2
        worker._kernel_policy_cooldown_ms = 60000
        worker._kernel_policy_probe_every = 3
        worker._kernel_policy_state = {
            "shape-2": {
                "requests": 2.0,
                "cooldown_until": time.time() + 30.0,
            }
        }

        allowed, reason = ClusterWorker._kernel_policy_allow(worker, shape_key="shape-2", lane="MISS")
        self.assertTrue(allowed)
        self.assertEqual(reason, "probe")

    def test_decode_governor_spec_cooldown_after_bad_streak(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_governor_enabled = True
        worker._decode_governor_probe_every = 12
        worker._decode_governor_spec_state = {}
        worker._spec_cfg = {"min_accept": 0.6}
        worker._spec_min_roi = 1.0
        worker._spec_bad_max = 2
        worker._spec_cooldown_ms = 60000

        allowed_first, reason_first = ClusterWorker._decode_governor_spec_allow(worker, shape_key="shape-s", lane="SPEC_MISS")
        self.assertTrue(allowed_first)
        self.assertEqual(reason_first, "active")

        bad_metrics = {
            "enabled": True,
            "proposed_tokens": 8,
            "accepted_tokens": 1,
            "acceptance_rate": 0.1,
            "roi": 0.2,
            "net_saved_ms": -5.0,
        }
        ClusterWorker._decode_governor_spec_update(worker, shape_key="shape-s", spec_metrics=bad_metrics)
        ClusterWorker._decode_governor_spec_update(worker, shape_key="shape-s", spec_metrics=bad_metrics)

        allowed_second, reason_second = ClusterWorker._decode_governor_spec_allow(worker, shape_key="shape-s", lane="SPEC_MISS")
        self.assertFalse(allowed_second)
        self.assertEqual(reason_second, "governor_cooldown")

    def test_decode_governor_spec_probe_during_cooldown(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_governor_enabled = True
        worker._decode_governor_probe_every = 3
        worker._decode_governor_spec_state = {
            "shape-s2": {
                "requests": 2,
                "cooldown_until": time.time() + 30.0,
            }
        }
        worker._spec_cfg = {"min_accept": 0.6}
        worker._spec_min_roi = 1.0
        worker._spec_bad_max = 2
        worker._spec_cooldown_ms = 60000

        allowed, reason = ClusterWorker._decode_governor_spec_allow(worker, shape_key="shape-s2", lane="SPEC_MISS")
        self.assertTrue(allowed)
        self.assertEqual(reason, "probe")

    def test_decode_governor_spec_k_uses_shape_state_and_bounds(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_governor_enabled = True
        worker._spec_adaptive_k_enabled = True
        worker._spec_k_min = 2
        worker._spec_k_max = 8
        worker._decode_governor_spec_state = {"shape-k": {"k_current": 10}}

        k = ClusterWorker._decode_governor_spec_k(
            worker,
            shape_key="shape-k",
            lane="SPEC_MISS",
            requested_k=6,
            max_tokens=5,
        )
        self.assertEqual(k, 5)
        self.assertEqual(worker._decode_governor_spec_state["shape-k"]["k_current"], 5)

    def test_decode_governor_spec_update_adapts_k(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_governor_enabled = True
        worker._decode_governor_probe_every = 12
        worker._decode_governor_spec_state = {"shape-adapt": {"k_current": 6}}
        worker._spec_cfg = {"min_accept": 0.6, "k": 6}
        worker._spec_min_roi = 1.0
        worker._spec_bad_max = 3
        worker._spec_cooldown_ms = 60000
        worker._spec_adaptive_k_enabled = True
        worker._spec_k_min = 2
        worker._spec_k_max = 8
        worker._spec_adaptive_k_down_accept = 0.65
        worker._spec_adaptive_k_down_roi = 0.9
        worker._spec_adaptive_k_up_accept = 0.9
        worker._spec_adaptive_k_up_roi = 1.1

        bad_metrics = {
            "enabled": True,
            "k": 6,
            "proposed_tokens": 8,
            "accepted_tokens": 2,
            "acceptance_rate": 0.25,
            "roi": 0.5,
            "net_saved_ms": -2.0,
        }
        ClusterWorker._decode_governor_spec_update(worker, shape_key="shape-adapt", spec_metrics=bad_metrics)
        self.assertEqual(worker._decode_governor_spec_state["shape-adapt"]["k_current"], 5)

        good_metrics = {
            "enabled": True,
            "k": 5,
            "proposed_tokens": 8,
            "accepted_tokens": 8,
            "acceptance_rate": 1.0,
            "roi": 1.5,
            "net_saved_ms": 10.0,
        }
        ClusterWorker._decode_governor_spec_update(worker, shape_key="shape-adapt", spec_metrics=good_metrics)
        self.assertEqual(worker._decode_governor_spec_state["shape-adapt"]["k_current"], 6)

    def test_decode_scheduler_default_prefers_queue_order(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_scheduler_enabled = False
        worker._queue_budget = {"SPEC_HIT": 1, "HIT": 1, "SPEC_MISS": 1, "MISS": 1}
        worker._spec_priority = 4
        worker._hit_priority = 3
        worker._miss_priority = 1
        worker._lane_decode_ema_ms = {}
        q_map = {
            "SPEC_HIT": queue.Queue(),
            "HIT": queue.Queue(),
            "SPEC_MISS": queue.Queue(),
            "MISS": queue.Queue(),
        }
        q_map["HIT"].put("hit")
        q_map["MISS"].put("miss")
        lane = ClusterWorker._select_lane_to_run(worker, ("SPEC_HIT", "HIT", "SPEC_MISS", "MISS"), q_map)
        self.assertEqual(lane, "HIT")

    def test_decode_scheduler_prefers_lower_decode_ema(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_scheduler_enabled = True
        worker._queue_budget = {"SPEC_HIT": 1, "HIT": 1, "SPEC_MISS": 1, "MISS": 1}
        worker._spec_priority = 4
        worker._hit_priority = 3
        worker._miss_priority = 1
        worker._lane_decode_ema_ms = {"HIT": 900.0, "MISS": 200.0}
        q_map = {
            "SPEC_HIT": queue.Queue(),
            "HIT": queue.Queue(),
            "SPEC_MISS": queue.Queue(),
            "MISS": queue.Queue(),
        }
        q_map["HIT"].put("hit")
        q_map["MISS"].put("miss")
        lane = ClusterWorker._select_lane_to_run(worker, ("SPEC_HIT", "HIT", "SPEC_MISS", "MISS"), q_map)
        self.assertEqual(lane, "MISS")

    def test_decode_scheduler_shape_aware_prefers_faster_shape_front(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_scheduler_enabled = True
        worker._decode_shape_aware_scheduler = True
        worker._queue_budget = {"SPEC_HIT": 1, "HIT": 1, "SPEC_MISS": 1, "MISS": 1}
        worker._spec_priority = 4
        worker._hit_priority = 3
        worker._miss_priority = 1
        worker._lane_decode_ema_ms = {"HIT": 900.0, "MISS": 900.0}
        worker._shape_decode_ema_ms = {"shape-fast": 180.0}
        q_map = {
            "SPEC_HIT": queue.Queue(),
            "HIT": queue.Queue(),
            "SPEC_MISS": queue.Queue(),
            "MISS": queue.Queue(),
        }
        q_map["HIT"].put(type("QItem", (), {"payload": {"job": {"routing_decision": {"shape_key": "shape-fast"}}}})())
        q_map["MISS"].put(type("QItem", (), {"payload": {"job": {"routing_decision": {"shape_key": "shape-slow"}}}})())
        lane = ClusterWorker._select_lane_to_run(worker, ("SPEC_HIT", "HIT", "SPEC_MISS", "MISS"), q_map)
        self.assertEqual(lane, "HIT")

    def test_decode_adaptive_budget_rebalances_to_faster_lane(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_adaptive_budget_enabled = True
        worker._decode_shape_aware_scheduler = False
        worker._decode_budget_depth_weight = 0.35
        worker._spec_priority = 4
        worker._hit_priority = 3
        worker._miss_priority = 1
        worker._queue_budget = {"SPEC_HIT": 0, "HIT": 0, "SPEC_MISS": 0, "MISS": 0}
        worker._lane_decode_ema_ms = {"HIT": 1200.0, "MISS": 200.0}
        worker._shape_decode_ema_ms = {}
        q_map = {
            "SPEC_HIT": queue.Queue(),
            "HIT": queue.Queue(),
            "SPEC_MISS": queue.Queue(),
            "MISS": queue.Queue(),
        }
        q_map["HIT"].put("hit-a")
        q_map["HIT"].put("hit-b")
        q_map["MISS"].put("miss-a")

        ClusterWorker._reset_queue_budget(worker, q_map)
        self.assertGreaterEqual(int(worker._queue_budget["MISS"]), int(worker._queue_budget["HIT"]))

    def test_decode_scheduler_replay_local_bonus_prefers_local_hit(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_scheduler_enabled = True
        worker._decode_shape_aware_scheduler = False
        worker._queue_budget = {"MISS": 1, "HIT": 1}
        worker._spec_priority = 1
        worker._hit_priority = 1
        worker._miss_priority = 1
        worker._lane_decode_ema_ms = {"MISS": 500.0, "HIT": 500.0}
        worker._shape_decode_ema_ms = {}
        q_map = {
            "MISS": queue.Queue(),
            "HIT": queue.Queue(),
        }
        q_map["MISS"].put(type("QItem", (), {"payload": {"job": {"routing_decision": {"replay_local": False}}}})())
        q_map["HIT"].put(type("QItem", (), {"payload": {"job": {"routing_decision": {"replay_local": True}}}})())
        lane = ClusterWorker._select_lane_to_run(worker, ("MISS", "HIT"), q_map)
        self.assertEqual(lane, "HIT")

    def test_vllm_decode_budget_tokens_shape_aware_ratio(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._vllm_decode_budget_min_tokens = 96
        worker._vllm_decode_budget_ratio_default = 1.0
        worker._vllm_decode_budget_ratio_by_lane = {
            "SPEC_HIT": 1.0,
            "HIT": 1.0,
            "SPEC_MISS": 1.0,
            "MISS": 1.0,
        }
        worker._lane_decode_ema_ms = {"MISS": 1000.0}
        worker._shape_decode_ema_ms = {"shape-fast": 700.0, "shape-slow": 1300.0}
        fast = ClusterWorker._vllm_decode_budget_tokens(worker, lane="MISS", shape_key="shape-fast", max_tokens=256)
        slow = ClusterWorker._vllm_decode_budget_tokens(worker, lane="MISS", shape_key="shape-slow", max_tokens=256)
        self.assertGreaterEqual(fast, slow)
        self.assertLessEqual(fast, 256)
        self.assertGreaterEqual(slow, 96)

    def test_build_vllm_runtime_contract_has_required_fields(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._vllm_runtime_contract_enabled = True
        worker._vllm_decode_budget_min_tokens = 96
        worker._vllm_decode_budget_ratio_default = 1.0
        worker._vllm_decode_budget_ratio_by_lane = {
            "SPEC_HIT": 1.0,
            "HIT": 1.0,
            "SPEC_MISS": 1.0,
            "MISS": 1.0,
        }
        worker._lane_decode_ema_ms = {}
        worker._shape_decode_ema_ms = {}
        contract = ClusterWorker._build_vllm_runtime_contract(
            worker,
            lane="HIT",
            shape_key="shape-a",
            max_tokens=192,
            queue_latency_ms=4.25,
            has_snapshot=True,
            vllm_priority=1,
            routing_decision={"replay_local": True, "snapshot_tier": "vram", "prompt_tokens": 1024},
        )
        self.assertEqual(contract["lane"], "HIT")
        self.assertEqual(contract["shape_key"], "shape-a")
        self.assertEqual(int(contract["target_tokens"]), 192)
        self.assertGreaterEqual(int(contract["decode_budget_tokens"]), 96)
        self.assertEqual(contract["replay_state"], "restore")
        self.assertTrue(bool(contract["replay_local"]))
        self.assertEqual(contract["snapshot_tier"], "vram")
        self.assertEqual(int(contract["prompt_tokens"]), 1024)
        self.assertEqual(int(contract["priority"]), 1)
        self.assertEqual(int(contract["spec_enabled"]), 0)
        self.assertEqual(int(contract["spec_k"]), 0)
        self.assertEqual(int(contract["contract_version"]), 1)

    def test_kernel_timing_estimate_strict_mode_requires_shadow_baseline(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_calibration_mode = "strict"
        worker._kernel_decode_baseline_ema = {}
        worker._kernel_gpu_arch = "ga"
        worker.gpu_id = 0

        job = {
            "deterministic_cfg": {"n_batch": 1, "n_ctx": 4096},
            "model": {"model_id": "m1"},
            "fingerprint": {"model_hash": "h1"},
            "prompt_tokens": 128,
        }
        metrics = {
            "perf": {"decode_ms": 80.0, "tokens_out": 64},
            "spec": {"enabled": False, "cache_hit": False},
            "kernels": {"kernels_applied": True},
        }
        ClusterWorker._update_kernel_timing_estimate(
            worker,
            job=job,
            run_mode="accel",
            engine_metrics=metrics,
            shadow_decode_baseline_ms=0.0,
            shadow_tokens_out=0,
        )
        self.assertFalse(bool(metrics["kernels"].get("comparable", False)))
        self.assertEqual(str(metrics["kernels"].get("comparable_tag", "")), "shadow_required")
        self.assertEqual(float(metrics["kernels"].get("ms_saved", 0.0)), 0.0)

        ClusterWorker._update_kernel_timing_estimate(
            worker,
            job=job,
            run_mode="accel",
            engine_metrics=metrics,
            shadow_decode_baseline_ms=120.0,
            shadow_tokens_out=64,
        )
        self.assertTrue(bool(metrics["kernels"].get("comparable", False)))
        self.assertEqual(str(metrics["kernels"].get("comparable_tag", "")), "shadow_baseline")
        self.assertAlmostEqual(float(metrics["kernels"].get("ms_saved", 0.0)), 40.0, places=6)

    def test_kernel_baseline_key_scopes_by_execution_backend(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._kernel_gpu_arch = "ga"
        worker.gpu_id = 0
        base_job = {
            "deterministic_cfg": {"n_batch": 1, "n_ctx": 4096},
            "model": {"model_id": "m1"},
            "fingerprint": {"model_hash": "h1"},
            "prompt_tokens": 128,
        }
        local_job = dict(base_job)
        local_job["backend_id"] = "korith_local"
        vllm_job = dict(base_job)
        vllm_job["backend_id"] = "vllm"
        local_key = ClusterWorker._kernel_baseline_key(worker, job=local_job, tokens_out=64)
        vllm_key = ClusterWorker._kernel_baseline_key(worker, job=vllm_job, tokens_out=64)
        self.assertNotEqual(local_key, vllm_key)

    def test_compose_metrics_drops_kernel_credit_when_comparability_requirements_fail(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker.worker_id = "w0"
        worker.node_id = "n0"
        worker.gpu_id = 0
        worker._engine_cfg = {"kv_layout_version": "v1"}
        worker._kernel_ctx = type("KernelCtx", (), {"backend": "cuda"})()

        job = {
            "job_id": "j3b",
            "created_at": "a",
            "org_id": "default",
            "request_id": "r3b",
            "backend_id": "korith_local",
            "backend_version": "korith_dynamic_v1",
            "model": {"model_id": "m3"},
            "fingerprint": {"model_hash": "mh3", "tokenizer_hash": "th3"},
            "prompt_hash": "ph3",
            "sampling_hash": "sh3",
            "prompt_tokens": 16,
        }
        result = {
            "exit_code": 0,
            "total_ms": 20.0,
            "engine_metrics": {
                "amf": {"supported": True, "decision": "hit", "saved_ms": 5.0},
                "mf": {"supported": True},
                "perf": {"tokens_out": 32, "total_ms": 20.0, "prefill_ms": 4.0, "decode_ms": 16.0, "avg_tps": 2.0},
                "spec": {"enabled": True, "saved_ms": 9.0},
                "kernels": {
                    "enabled": True,
                    "kernels_applied": True,
                    "comparable": True,
                    "comparable_requirements": {
                        "same_model": True,
                        "same_shape": True,
                        "same_generation_len": False,
                        "same_backend_path": True,
                        "same_non_spec_path": True,
                    },
                    "backend": "cuda",
                    "ms_saved": 3.0,
                },
            },
        }
        metrics = ClusterWorker._compose_metrics(
            worker,
            job=job,
            result=result,
            lane="SPEC_HIT",
            queue_latency_ms=1.0,
            started_at="b",
            finished_at="c",
            caps=None,
        )
        self.assertEqual(float(metrics["savings"]["kernels_saved_ms"]), 0.0)
        self.assertEqual(float(metrics["savings"]["spec_saved_ms"]), 9.0)

    def test_compose_metrics_savings_components_are_separate(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker.worker_id = "w0"
        worker.node_id = "n0"
        worker.gpu_id = 0
        worker._engine_cfg = {"kv_layout_version": "v1"}
        worker._kernel_ctx = type("KernelCtx", (), {"backend": "cuda"})()

        job = {
            "job_id": "j2",
            "created_at": "a",
            "org_id": "default",
            "request_id": "r2",
            "backend_id": "korith_local",
            "backend_version": "korith_dynamic_v1",
            "model": {"model_id": "m2"},
            "fingerprint": {"model_hash": "mh2", "tokenizer_hash": "th2"},
            "prompt_hash": "ph2",
            "sampling_hash": "sh2",
            "prompt_tokens": 16,
        }
        result = {
            "exit_code": 0,
            "total_ms": 20.0,
            "engine_metrics": {
                "amf": {"supported": True, "decision": "hit", "saved_ms": 5.0},
                "mf": {"supported": True},
                "perf": {"tokens_out": 32, "total_ms": 20.0, "prefill_ms": 4.0, "decode_ms": 16.0, "avg_tps": 2.0},
                "spec": {"enabled": True, "saved_ms": 7.0},
                "kernels": {"enabled": True, "kernels_applied": True, "backend": "cuda", "ms_saved": 3.0},
            },
        }
        metrics = ClusterWorker._compose_metrics(
            worker,
            job=job,
            result=result,
            lane="SPEC_HIT",
            queue_latency_ms=1.0,
            started_at="b",
            finished_at="c",
            caps=None,
        )
        self.assertEqual(float(metrics["savings"]["prefill_saved_ms"]), 5.0)
        self.assertEqual(float(metrics["savings"]["kernels_saved_ms"]), 3.0)
        self.assertEqual(float(metrics["savings"]["spec_saved_ms"]), 4.0)
        self.assertEqual(float(metrics["savings"]["total_saved_ms"]), 12.0)

    def test_compose_metrics_drops_kernel_credit_when_not_applied(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker.worker_id = "w0"
        worker.node_id = "n0"
        worker.gpu_id = 0
        worker._engine_cfg = {"kv_layout_version": "v1"}
        worker._kernel_ctx = type("KernelCtx", (), {"backend": "cuda"})()

        job = {
            "job_id": "j3",
            "created_at": "a",
            "org_id": "default",
            "request_id": "r3",
            "backend_id": "korith_local",
            "backend_version": "korith_dynamic_v1",
            "model": {"model_id": "m3"},
            "fingerprint": {"model_hash": "mh3", "tokenizer_hash": "th3"},
            "prompt_hash": "ph3",
            "sampling_hash": "sh3",
            "prompt_tokens": 16,
        }
        result = {
            "exit_code": 0,
            "total_ms": 20.0,
            "engine_metrics": {
                "amf": {"supported": True, "decision": "hit", "saved_ms": 5.0},
                "mf": {"supported": True},
                "perf": {"tokens_out": 32, "total_ms": 20.0, "prefill_ms": 4.0, "decode_ms": 16.0, "avg_tps": 2.0},
                "spec": {"enabled": True, "saved_ms": 9.0},
                "kernels": {"enabled": True, "kernels_applied": False, "comparable": True, "backend": "cuda", "ms_saved": 3.0},
            },
        }
        metrics = ClusterWorker._compose_metrics(
            worker,
            job=job,
            result=result,
            lane="SPEC_HIT",
            queue_latency_ms=1.0,
            started_at="b",
            finished_at="c",
            caps=None,
        )
        self.assertEqual(float(metrics["savings"]["prefill_saved_ms"]), 5.0)
        self.assertEqual(float(metrics["savings"]["kernels_saved_ms"]), 0.0)
        self.assertEqual(float(metrics["savings"]["spec_saved_ms"]), 9.0)
        self.assertEqual(float(metrics["savings"]["total_saved_ms"]), 14.0)

    def test_compose_metrics_infers_vllm_amf_from_lane_and_miss_ema(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker.worker_id = "w0"
        worker.node_id = "n0"
        worker.gpu_id = 0
        worker._engine_cfg = {"kv_layout_version": "v1"}
        worker._kernel_ctx = type("KernelCtx", (), {"backend": "cuda"})()
        worker._vllm_amf_infer_by_lane = True
        worker._vllm_amf_miss_ema = {}

        caps = Capabilities(
            kv_replay=True,
            deterministic_seeding=True,
            streaming=True,
            batch_prefill=True,
            logits_access=False,
            verify_tokens=False,
            draft_supported=False,
        )

        base_job = {
            "created_at": "a",
            "org_id": "default",
            "backend_id": "korith_local",
            "backend_version": "korith_dynamic_v1",
            "execution_backend_id": "vllm",
            "model": {"model_id": "m"},
            "execution_model": {"model_id": "m", "endpoint": "http://127.0.0.1:28010"},
            "fingerprint": {"model_hash": "mh", "tokenizer_hash": "th"},
            "prompt_hash": "ph",
            "sampling_hash": "sh",
            "fingerprint_hash": "fp",
            "prompt_tokens": 1000,
        }

        miss_job = dict(base_job)
        miss_job["job_id"] = "miss"
        miss_job["request_id"] = "r-miss"
        miss_result = {
            "exit_code": 0,
            "total_ms": 1000.0,
            "engine_metrics": {
                "amf": {"supported": False, "decision": "unavailable", "saved_ms": 0.0},
                "mf": {"supported": True},
                "perf": {"tokens_out": 64, "total_ms": 1000.0, "prefill_ms": 0.0, "decode_ms": 1000.0, "avg_tps": 64.0},
                "spec": {"enabled": False},
                "kernels": {"enabled": True, "kernels_applied": True, "backend": "cuda", "ms_saved": 0.0},
            },
        }
        miss_metrics = ClusterWorker._compose_metrics(
            worker,
            job=miss_job,
            result=miss_result,
            lane="MISS",
            queue_latency_ms=1.0,
            started_at="b",
            finished_at="c",
            caps=caps,
        )
        self.assertTrue(bool(miss_metrics["amf"]["supported"]))
        self.assertEqual(str(miss_metrics["amf"]["decision"]), "miss")

        hit_job = dict(base_job)
        hit_job["job_id"] = "hit"
        hit_job["request_id"] = "r-hit"
        hit_result = {
            "exit_code": 0,
            "total_ms": 300.0,
            "engine_metrics": {
                "amf": {"supported": False, "decision": "unavailable", "saved_ms": 0.0},
                "mf": {"supported": True},
                "perf": {"tokens_out": 64, "total_ms": 300.0, "prefill_ms": 0.0, "decode_ms": 300.0, "avg_tps": 213.0},
                "spec": {"enabled": False},
                "kernels": {"enabled": True, "kernels_applied": True, "backend": "cuda", "ms_saved": 0.0},
            },
        }
        hit_metrics = ClusterWorker._compose_metrics(
            worker,
            job=hit_job,
            result=hit_result,
            lane="HIT",
            queue_latency_ms=1.0,
            started_at="b",
            finished_at="c",
            caps=caps,
        )
        self.assertTrue(bool(hit_metrics["amf"]["supported"]))
        self.assertEqual(str(hit_metrics["amf"]["decision"]), "hit")
        self.assertGreater(float(hit_metrics["amf"]["saved_ms"]), 600.0)
        self.assertGreater(float(hit_metrics["savings"]["prefill_saved_ms"]), 600.0)
        self.assertEqual(float(hit_metrics["savings"]["kernels_saved_ms"]), 0.0)

    def test_resolve_snapshot_input_path_accepts_existing_path_when_node_id_changes(self):
        class _Restore:
            def __init__(self):
                self.last_set = None

            def set(self, fingerprint_hash, snapshot_path):
                self.last_set = (fingerprint_hash, snapshot_path)

        class _SnapshotIndex:
            def __init__(self, rows):
                self.rows = rows
                self.upserts = []

            def get_locations(self, fingerprint_hash, org_id):
                return list(self.rows)

            def upsert_location(self, **kwargs):
                self.upserts.append(kwargs)

        with tempfile.TemporaryDirectory() as td:
            snap_path = Path(td) / "fp.bin"
            snap_path.write_bytes(b"abc")

            worker = ClusterWorker.__new__(ClusterWorker)
            worker.node_id = "node-new"
            worker.worker_id = "worker-new"
            worker.restore_store = _Restore()
            worker.snapshot_index = _SnapshotIndex(
                [
                    {
                        "snapshot_id": "snap-old",
                        "node_id": "node-old",
                        "snapshot_path": str(snap_path),
                    }
                ]
            )

            resolved = ClusterWorker._resolve_snapshot_input_path(
                worker,
                fingerprint_hash="fp",
                org_id="default",
                candidate_path=None,
            )
            self.assertEqual(resolved, str(snap_path))
            self.assertEqual(worker.restore_store.last_set, ("fp", str(snap_path)))
            self.assertEqual(len(worker.snapshot_index.upserts), 1)
            self.assertEqual(worker.snapshot_index.upserts[0]["node_id"], "node-new")

    def test_resolve_snapshot_input_path_prefers_same_node_without_reindex(self):
        class _Restore:
            def __init__(self):
                self.last_set = None

            def set(self, fingerprint_hash, snapshot_path):
                self.last_set = (fingerprint_hash, snapshot_path)

        class _SnapshotIndex:
            def __init__(self, rows):
                self.rows = rows
                self.upserts = []

            def get_locations(self, fingerprint_hash, org_id):
                return list(self.rows)

            def upsert_location(self, **kwargs):
                self.upserts.append(kwargs)

        with tempfile.TemporaryDirectory() as td:
            snap_path = Path(td) / "fp.bin"
            snap_path.write_bytes(b"abc")

            worker = ClusterWorker.__new__(ClusterWorker)
            worker.node_id = "node-a"
            worker.worker_id = "worker-a"
            worker.restore_store = _Restore()
            worker.snapshot_index = _SnapshotIndex(
                [
                    {
                        "snapshot_id": "snap-a",
                        "node_id": "node-a",
                        "snapshot_path": str(snap_path),
                    }
                ]
            )

            resolved = ClusterWorker._resolve_snapshot_input_path(
                worker,
                fingerprint_hash="fp",
                org_id="default",
                candidate_path=None,
            )
            self.assertEqual(resolved, str(snap_path))
            self.assertEqual(worker.restore_store.last_set, ("fp", str(snap_path)))
            self.assertEqual(worker.snapshot_index.upserts, [])

    def test_promote_snapshot_input_moves_nvme_to_vram_and_records_index(self):
        class _Restore:
            def __init__(self):
                self.last_set = None

            def set(self, fingerprint_hash, snapshot_path):
                self.last_set = (fingerprint_hash, snapshot_path)

        class _SnapshotIndex:
            def __init__(self):
                self.upserts = []

            def upsert_location(self, **kwargs):
                self.upserts.append(kwargs)

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            vram = base / "vram"
            ram = base / "ram"
            nvme = base / "nvme"
            for d in (vram, ram, nvme):
                d.mkdir(parents=True, exist_ok=True)
            source = nvme / "fp.bin"
            source.write_bytes(b"12345678")

            worker = ClusterWorker.__new__(ClusterWorker)
            worker.node_id = "node-a"
            worker.worker_id = "worker-a"
            worker.restore_store = _Restore()
            worker.snapshot_index = _SnapshotIndex()
            worker._snapshot_vram_dir = vram
            worker._snapshot_ram_dir = ram
            worker._snapshot_nvme_dir = nvme
            worker._snapshot_vram_max_bytes = 1024
            worker._snapshot_ram_max_bytes = 1024 * 1024
            worker._snapshot_vram_cache_max_bytes = 0
            worker._snapshot_ram_cache_max_bytes = 0
            worker._snapshot_nvme_cache_max_bytes = 0
            events = []
            worker._append_event = lambda _ep, et, payload: events.append((et, payload))

            promoted = ClusterWorker._maybe_promote_snapshot_input(
                worker,
                fingerprint_hash="fp",
                org_id="default",
                snapshot_path=str(source),
                events_path=base / "events.jsonl",
                job_id="j1",
                request_id="r1",
            )
            self.assertEqual(promoted, str(vram / "fp.bin"))
            self.assertTrue((vram / "fp.bin").exists())
            self.assertEqual(worker.restore_store.last_set, ("fp", str(vram / "fp.bin")))
            self.assertEqual(len(worker.snapshot_index.upserts), 1)
            self.assertEqual(worker.snapshot_index.upserts[0]["snapshot_path"], str(vram / "fp.bin"))
            event_types = [e[0] for e in events]
            self.assertIn("SNAPSHOT_PROMOTE", event_types)

    def test_snapshot_vram_budget_evicts_oldest_cached_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            vram = base / "vram"
            ram = base / "ram"
            nvme = base / "nvme"
            for d in (vram, ram, nvme):
                d.mkdir(parents=True, exist_ok=True)

            worker = ClusterWorker.__new__(ClusterWorker)
            worker._snapshot_vram_dir = vram
            worker._snapshot_ram_dir = ram
            worker._snapshot_nvme_dir = nvme
            worker._snapshot_vram_max_bytes = 1024
            worker._snapshot_ram_max_bytes = 1024 * 1024
            worker._snapshot_vram_cache_max_bytes = 12
            worker._snapshot_ram_cache_max_bytes = 0
            worker._snapshot_nvme_cache_max_bytes = 0

            ClusterWorker._cache_snapshot_from_blob(worker, "fp1", b"abcdefghij")
            self.assertTrue((vram / "fp1.bin").exists())
            ClusterWorker._cache_snapshot_from_blob(worker, "fp2", b"1234567890")
            self.assertFalse((vram / "fp1.bin").exists())
            self.assertTrue((vram / "fp2.bin").exists())

    def test_is_kv_transfer_snapshot_detects_native_payload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mf_snapshot.bin"
            path.write_text(json.dumps({"kv_transfer_params": {"cache_id": "abc"}}), encoding="utf-8")
            worker = ClusterWorker.__new__(ClusterWorker)
            self.assertTrue(ClusterWorker._is_kv_transfer_snapshot(worker, str(path)))

    def test_is_kv_transfer_snapshot_rejects_binary_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mf_snapshot.bin"
            path.write_bytes(b"\x00\x01\x02\x03")
            worker = ClusterWorker.__new__(ClusterWorker)
            self.assertFalse(ClusterWorker._is_kv_transfer_snapshot(worker, str(path)))

    def test_try_vllm_oom_fallback_retries_local_miss_on_vllm(self):
        class _FallbackAdapter:
            backend_version = "vllm-test"

            def get_capabilities(self):
                return Capabilities(kv_replay=False, deterministic_seeding=True)

            def get_fingerprint(self):
                return {"model_hash": "vm", "tokenizer_hash": "vt"}

            def run_baseline(self, prompt, max_tokens, deterministic_cfg, policy, artifacts, mf_snapshot_in):
                _ = (prompt, max_tokens, deterministic_cfg, policy, artifacts, mf_snapshot_in)
                return {
                    "exit_code": 0,
                    "output_text": "ok",
                    "total_ms": 12.0,
                    "engine_metrics": {
                        "perf": {"tokens_out": 4, "total_ms": 12.0, "prefill_ms": 2.0, "decode_ms": 10.0, "avg_tps": 333.0},
                        "amf": {"supported": False, "decision": "unavailable"},
                        "mf": {"supported": False},
                        "spec": {"enabled": False},
                    },
                }

        class _AdapterRegistry:
            def get_adapter(self, jobspec):
                self.last = jobspec
                return _FallbackAdapter()

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "run.log"
            log_path.write_text("cudaMalloc failed: out of memory", encoding="utf-8")
            worker = ClusterWorker.__new__(ClusterWorker)
            worker._vllm_oom_fallback_enabled = True
            worker._vllm_oom_fallback_all_lanes = False
            worker.adapter_registry = _AdapterRegistry()
            events = []
            worker._append_event = lambda _ep, et, payload: events.append((et, payload))

            old_env = {k: os.environ.get(k) for k in ("KORITH_VLLM_ENDPOINT", "KORITH_VLLM_MODEL_ID", "KORITH_VLLM_BACKEND_ID")}
            try:
                os.environ["KORITH_VLLM_ENDPOINT"] = "http://127.0.0.1:8000"
                os.environ["KORITH_VLLM_MODEL_ID"] = "vllm-model"
                os.environ["KORITH_VLLM_BACKEND_ID"] = "vllm"
                job = {
                    "job_id": "job-oom-fallback",
                    "org_id": "org1",
                    "backend_id": "korith_local",
                    "execution_backend_id": "korith_local",
                    "routing_decision": {},
                }
                fallback = ClusterWorker._try_vllm_oom_fallback(
                    worker,
                    job=job,
                    lane="MISS",
                    result={"exit_code": 1, "engine_errors": ["cudaMalloc failed: out of memory"]},
                    prompt="hello",
                    max_tokens=16,
                    deterministic_cfg={"seed": 1},
                    policy={"allow_amf_reuse": True, "allow_spec": False},
                    artifacts={"log": str(log_path), "output": str(Path(td) / "output.txt"), "job_dir": str(Path(td))},
                    mf_snapshot_in=None,
                    events_path=Path(td) / "events.jsonl",
                    request_id="r1",
                )
                self.assertIsNotNone(fallback)
                self.assertEqual(job["execution_backend_id"], "vllm")
                self.assertEqual(job["routing_decision"]["execution_reason"], "worker_oom_fallback")
                self.assertEqual(int(fallback["result"]["exit_code"]), 0)
                event_types = [row[0] for row in events]
                self.assertIn("BACKEND_FALLBACK", event_types)
                self.assertIn("BACKEND_FALLBACK_SUCCESS", event_types)
            finally:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_try_vllm_oom_fallback_skips_hit_lane_by_default(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._vllm_oom_fallback_enabled = True
        worker._vllm_oom_fallback_all_lanes = False
        worker.adapter_registry = object()
        worker._append_event = lambda *_args, **_kwargs: None
        with tempfile.TemporaryDirectory() as td:
            out = ClusterWorker._try_vllm_oom_fallback(
                worker,
                job={"backend_id": "korith_local", "execution_backend_id": "korith_local"},
                lane="HIT",
                result={"exit_code": 1, "engine_errors": ["out of memory"]},
                prompt="p",
                max_tokens=8,
                deterministic_cfg={},
                policy={},
                artifacts={"log": str(Path(td) / "run.log")},
                mf_snapshot_in=None,
                events_path=Path(td) / "events.jsonl",
                request_id="r2",
            )
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
