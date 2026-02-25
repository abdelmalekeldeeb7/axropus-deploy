from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.adapters.vllm_openai import VllmOpenAIAdapter


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class VllmAdapterTests(unittest.TestCase):
    def test_capabilities_enable_kv_replay_when_mf_enabled(self):
        old = os.environ.get("KORITH_VLLM_MF_ENABLED")
        try:
            os.environ["KORITH_VLLM_MF_ENABLED"] = "1"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            caps = adapter.get_capabilities()
            self.assertTrue(bool(caps.kv_replay))
        finally:
            if old is None:
                os.environ.pop("KORITH_VLLM_MF_ENABLED", None)
            else:
                os.environ["KORITH_VLLM_MF_ENABLED"] = old

    def test_capabilities_enable_spec_when_env_enabled(self):
        old = os.environ.get("KORITH_VLLM_ENABLE_SPEC_DECODE")
        try:
            os.environ["KORITH_VLLM_ENABLE_SPEC_DECODE"] = "1"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            caps = adapter.get_capabilities()
            self.assertTrue(bool(caps.verify_tokens))
            self.assertTrue(bool(caps.draft_supported))
        finally:
            if old is None:
                os.environ.pop("KORITH_VLLM_ENABLE_SPEC_DECODE", None)
            else:
                os.environ["KORITH_VLLM_ENABLE_SPEC_DECODE"] = old

    def test_amf_full_hit_from_cached_tokens(self):
        adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
        payload = {
            "choices": [{"text": "x y"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            with patch("platform.adapters.vllm_openai.request.urlopen", return_value=_FakeHTTPResponse(payload)):
                res = adapter.run_baseline(
                    prompt="hello world",
                    max_tokens=16,
                    deterministic_cfg={"temperature": 0.0},
                    policy={"allow_amf_reuse": True, "allow_spec": False},
                    artifacts={"output": str(out)},
                    mf_snapshot_in=None,
                )
        amf = res["engine_metrics"]["amf"]
        self.assertTrue(bool(amf["supported"]))
        self.assertEqual(str(amf["decision"]), "hit")
        self.assertEqual(int(amf["prefix_len"]), 20)
        self.assertAlmostEqual(float(amf["skip_ratio"]), 1.0, places=6)
        self.assertEqual(int(res["engine_metrics"]["perf"]["tokens_out"]), 2)

    def test_amf_estimated_prefill_savings_feed_perf_and_savings(self):
        adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
        payload = {
            "choices": [{"text": "x y"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            with patch("platform.adapters.vllm_openai.request.urlopen", return_value=_FakeHTTPResponse(payload)):
                with patch("platform.adapters.vllm_openai.time.time", side_effect=[10.0, 10.5]):
                    res = adapter.run_baseline(
                        prompt="hello world",
                        max_tokens=16,
                        deterministic_cfg={"temperature": 0.0},
                        policy={"allow_amf_reuse": True, "allow_spec": False},
                        artifacts={"output": str(out)},
                        mf_snapshot_in=None,
                    )
        amf = res["engine_metrics"]["amf"]
        perf = res["engine_metrics"]["perf"]
        self.assertEqual(str(amf.get("estimate_method", "")), "token_proportional")
        self.assertGreater(float(amf.get("baseline_prefix_ms", 0.0)), 0.0)
        self.assertGreater(float(amf.get("saved_ms", 0.0)), 0.0)
        self.assertEqual(float(perf.get("prefill_ms", 0.0)), 0.0)
        self.assertGreater(float(perf.get("decode_ms", 0.0)), 0.0)
        self.assertAlmostEqual(
            float(perf.get("prefill_ms", 0.0)) + float(perf.get("decode_ms", 0.0)),
            float(perf.get("total_ms", 0.0)),
            places=4,
        )

    def test_amf_miss_when_cached_tokens_zero(self):
        adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
        payload = {
            "choices": [{"text": "x y z"}],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            with patch("platform.adapters.vllm_openai.request.urlopen", return_value=_FakeHTTPResponse(payload)):
                res = adapter.run_baseline(
                    prompt="hello world",
                    max_tokens=16,
                    deterministic_cfg={"temperature": 0.0},
                    policy={"allow_amf_reuse": True, "allow_spec": False},
                    artifacts={"output": str(out)},
                    mf_snapshot_in=None,
                )
        amf = res["engine_metrics"]["amf"]
        self.assertTrue(bool(amf["supported"]))
        self.assertEqual(str(amf["decision"]), "miss")
        self.assertEqual(int(amf["prefix_len"]), 0)
        self.assertAlmostEqual(float(amf["skip_ratio"]), 0.0, places=6)

    def test_amf_unavailable_without_cached_tokens_field(self):
        adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
        payload = {
            "choices": [{"text": "x y z"}],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 3,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            with patch("platform.adapters.vllm_openai.request.urlopen", return_value=_FakeHTTPResponse(payload)):
                res = adapter.run_baseline(
                    prompt="hello world",
                    max_tokens=16,
                    deterministic_cfg={"temperature": 0.0},
                    policy={"allow_amf_reuse": True, "allow_spec": False},
                    artifacts={"output": str(out)},
                    mf_snapshot_in=None,
                )
        amf = res["engine_metrics"]["amf"]
        self.assertFalse(bool(amf["supported"]))
        self.assertEqual(str(amf["decision"]), "unavailable")

    def test_mf_snapshot_written_from_kv_transfer_params(self):
        old = os.environ.get("KORITH_VLLM_MF_ENABLED")
        try:
            os.environ["KORITH_VLLM_MF_ENABLED"] = "1"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            payload = {
                "choices": [{"text": "x y z"}],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "kv_transfer_params": {"connector": "nixl", "kv_role": "kv_both"},
            }
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "out.txt"
                snap = Path(td) / "snap.json"
                with patch("platform.adapters.vllm_openai.request.urlopen", return_value=_FakeHTTPResponse(payload)):
                    res = adapter.run_baseline(
                        prompt="hello world",
                        max_tokens=16,
                        deterministic_cfg={"temperature": 0.0},
                        policy={"allow_amf_reuse": True, "allow_spec": False},
                        artifacts={"output": str(out), "mf_snapshot": str(snap)},
                        mf_snapshot_in=None,
                    )
                self.assertTrue(snap.exists())
                stored = json.loads(snap.read_text(encoding="utf-8"))
                self.assertIn("kv_transfer_params", stored)
                mf = res["engine_metrics"]["mf"]
                self.assertTrue(bool(mf["supported"]))
                self.assertTrue(bool(mf["snapshot_emitted"]))
                self.assertTrue(bool(mf["snapshot_id"]))
        finally:
            if old is None:
                os.environ.pop("KORITH_VLLM_MF_ENABLED", None)
            else:
                os.environ["KORITH_VLLM_MF_ENABLED"] = old

    def test_mf_restore_loads_snapshot_into_request_payload(self):
        old = os.environ.get("KORITH_VLLM_MF_ENABLED")
        try:
            os.environ["KORITH_VLLM_MF_ENABLED"] = "1"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            seen = {}

            def _open(req):
                seen["payload"] = json.loads(req.data.decode("utf-8"))
                return _FakeHTTPResponse(
                    {
                        "choices": [{"text": "x"}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 10},
                        },
                    }
                )

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                out = td_path / "out.txt"
                snap = td_path / "in_snap.json"
                snap.write_text(json.dumps({"kv_transfer_params": {"cache_id": "abc123"}}), encoding="utf-8")
                with patch("platform.adapters.vllm_openai.request.urlopen", side_effect=_open):
                    res = adapter.run_baseline(
                        prompt="hello world",
                        max_tokens=16,
                        deterministic_cfg={"temperature": 0.0},
                        policy={"allow_amf_reuse": True, "allow_spec": False},
                        artifacts={"output": str(out), "mf_snapshot": str(td_path / "out_snap.json")},
                        mf_snapshot_in=str(snap),
                    )
                self.assertIn("kv_transfer_params", seen["payload"])
                self.assertEqual(str(seen["payload"]["kv_transfer_params"]["cache_id"]), "abc123")
                self.assertTrue(bool(res["engine_metrics"]["mf"]["restored"]))
        finally:
            if old is None:
                os.environ.pop("KORITH_VLLM_MF_ENABLED", None)
            else:
                os.environ["KORITH_VLLM_MF_ENABLED"] = old

    def test_custom_lane_body_overrides_are_applied(self):
        keys = (
            "KORITH_VLLM_EXTRA_BODY_JSON",
            "KORITH_VLLM_EXTRA_BODY_HIT_JSON",
            "KORITH_VLLM_MAX_TOKENS_CAP",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_EXTRA_BODY_JSON"] = json.dumps({"ignore_eos": True, "top_k": 40})
            os.environ["KORITH_VLLM_EXTRA_BODY_HIT_JSON"] = json.dumps({"min_tokens": 32, "top_k": 8})
            os.environ["KORITH_VLLM_MAX_TOKENS_CAP"] = "64"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            seen = {}

            def _open(req):
                seen["payload"] = json.loads(req.data.decode("utf-8"))
                return _FakeHTTPResponse(
                    {
                        "choices": [{"text": "x"}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                    }
                )

            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "out.txt"
                with patch("platform.adapters.vllm_openai.request.urlopen", side_effect=_open):
                    adapter.run_baseline(
                        prompt="hello world",
                        max_tokens=256,
                        deterministic_cfg={"temperature": 0.0},
                        policy={"allow_amf_reuse": True, "allow_spec": False, "_lane": "HIT"},
                        artifacts={"output": str(out)},
                        mf_snapshot_in=None,
                    )
            payload = seen["payload"]
            self.assertTrue(bool(payload.get("ignore_eos")))
            self.assertEqual(int(payload.get("top_k", 0)), 8)
            self.assertEqual(int(payload.get("max_tokens", 0)), 64)
            self.assertEqual(int(payload.get("min_tokens", 0)), 32)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_run_speculative_sets_vllm_spec_payload(self):
        keys = (
            "KORITH_VLLM_ENABLE_SPEC_DECODE",
            "KORITH_VLLM_SPECULATIVE_MODEL",
            "KORITH_VLLM_SPECULATIVE_NUM_TOKENS",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_ENABLE_SPEC_DECODE"] = "1"
            os.environ["KORITH_VLLM_SPECULATIVE_MODEL"] = "draft-model"
            os.environ["KORITH_VLLM_SPECULATIVE_NUM_TOKENS"] = "6"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            seen = {}

            def _open(req):
                seen["payload"] = json.loads(req.data.decode("utf-8"))
                return _FakeHTTPResponse(
                    {
                        "choices": [{"text": "x y z"}],
                        "usage": {
                            "prompt_tokens": 30,
                            "completion_tokens": 3,
                            "prompt_tokens_details": {"cached_tokens": 0},
                            "accepted_speculative_tokens": 2,
                            "proposed_speculative_tokens": 3,
                        },
                    }
                )

            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "out.txt"
                res = None
                with patch("platform.adapters.vllm_openai.request.urlopen", side_effect=_open):
                    res = adapter.run_speculative(
                        prompt="hello world",
                        max_tokens=16,
                        deterministic_cfg={"temperature": 0.0},
                        policy={"allow_amf_reuse": True, "allow_spec": True, "_lane": "SPEC_MISS"},
                        artifacts={"output": str(out)},
                        mf_snapshot_in=None,
                        spec_cfg={"k": 7},
                    )
            payload = seen["payload"]
            self.assertEqual(str(payload.get("speculative_model", "")), "draft-model")
            self.assertEqual(int(payload.get("num_speculative_tokens", 0)), 7)
            xargs = payload.get("vllm_xargs", {})
            self.assertIsInstance(xargs, dict)
            self.assertEqual(int(xargs.get("korith_spec_enabled", 0)), 1)
            self.assertEqual(int(xargs.get("korith_spec_k", 0)), 7)
            spec = res["engine_metrics"]["spec"]
            self.assertTrue(bool(spec.get("enabled", False)))
            self.assertEqual(int(spec.get("proposed_tokens", 0)), 3)
            self.assertEqual(int(spec.get("accepted_tokens", 0)), 2)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_run_speculative_applies_lane_specific_spec_overrides(self):
        keys = (
            "KORITH_VLLM_ENABLE_SPEC_DECODE",
            "KORITH_VLLM_SPECULATIVE_MODEL_SPEC_HIT",
            "KORITH_VLLM_SPECULATIVE_NUM_TOKENS_SPEC_HIT",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_ENABLE_SPEC_DECODE"] = "1"
            os.environ["KORITH_VLLM_SPECULATIVE_MODEL_SPEC_HIT"] = "draft-hit"
            os.environ["KORITH_VLLM_SPECULATIVE_NUM_TOKENS_SPEC_HIT"] = "9"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            seen = {}

            def _open(req):
                seen["payload"] = json.loads(req.data.decode("utf-8"))
                return _FakeHTTPResponse(
                    {
                        "choices": [{"text": "ok"}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                    }
                )

            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "out.txt"
                with patch("platform.adapters.vllm_openai.request.urlopen", side_effect=_open):
                    adapter.run_speculative(
                        prompt="hello world",
                        max_tokens=16,
                        deterministic_cfg={"temperature": 0.0},
                        policy={"allow_amf_reuse": True, "allow_spec": True, "_lane": "SPEC_HIT"},
                        artifacts={"output": str(out)},
                        mf_snapshot_in=None,
                        spec_cfg={},
                    )

            payload = seen["payload"]
            self.assertEqual(str(payload.get("speculative_model", "")), "draft-hit")
            self.assertEqual(int(payload.get("num_speculative_tokens", 0)), 9)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_custom_payload_policy_sets_priority_when_enabled(self):
        keys = ("KORITH_VLLM_PRIORITY_SCHED",)
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_PRIORITY_SCHED"] = "1"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            seen = {}

            def _open(req):
                seen["payload"] = json.loads(req.data.decode("utf-8"))
                return _FakeHTTPResponse(
                    {
                        "choices": [{"text": "ok"}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                    }
                )

            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "out.txt"
                with patch("platform.adapters.vllm_openai.request.urlopen", side_effect=_open):
                    adapter.run_baseline(
                        prompt="hello world",
                        max_tokens=16,
                        deterministic_cfg={"temperature": 0.0},
                        policy={"allow_amf_reuse": True, "allow_spec": False, "_lane": "MISS", "_vllm_priority": 7},
                        artifacts={"output": str(out)},
                        mf_snapshot_in=None,
                    )
            payload = seen["payload"]
            self.assertEqual(int(payload.get("priority", -1)), 7)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_custom_payload_policy_emits_runtime_contract_vllm_xargs(self):
        keys = ("KORITH_VLLM_RUNTIME_CONTRACT_ENABLED",)
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_RUNTIME_CONTRACT_ENABLED"] = "1"
            adapter = VllmOpenAIAdapter(endpoint="http://127.0.0.1:8000", model_id="m")
            seen = {}

            def _open(req):
                seen["payload"] = json.loads(req.data.decode("utf-8"))
                return _FakeHTTPResponse(
                    {
                        "choices": [{"text": "ok"}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                    }
                )

            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "out.txt"
                with patch("platform.adapters.vllm_openai.request.urlopen", side_effect=_open):
                    adapter.run_baseline(
                        prompt="hello world",
                        max_tokens=64,
                        deterministic_cfg={"temperature": 0.0},
                        policy={
                            "allow_amf_reuse": False,
                            "allow_spec": False,
                            "_lane": "MISS",
                            "_shape_key": "shape-a",
                            "_vllm_contract": {
                                "lane": "MISS",
                                "shape_key": "shape-a",
                                "target_tokens": 64,
                                "decode_budget_tokens": 48,
                                "replay_state": "miss",
                                "replay_local": True,
                                "snapshot_tier": "vram",
                                "prompt_tokens": 1024,
                                "queue_latency_ms": 12.5,
                                "priority": 3,
                                "spec_enabled": True,
                                "spec_k": 6,
                                "spec_min_accept": 0.7,
                                "spec_disable_after_n": 48,
                                "spec_cache_only": False,
                                "contract_version": 1,
                            },
                        },
                        artifacts={"output": str(out)},
                        mf_snapshot_in=None,
                    )
            payload = seen["payload"]
            xargs = payload.get("vllm_xargs", {})
            self.assertIsInstance(xargs, dict)
            self.assertEqual(str(xargs.get("korith_lane", "")), "MISS")
            self.assertEqual(str(xargs.get("korith_shape_key", "")), "shape-a")
            self.assertEqual(int(xargs.get("korith_target_tokens", 0)), 64)
            self.assertEqual(int(xargs.get("korith_decode_budget_tokens", 0)), 48)
            self.assertEqual(str(xargs.get("korith_replay_state", "")), "miss")
            self.assertEqual(int(xargs.get("korith_replay_local", 0)), 1)
            self.assertEqual(str(xargs.get("korith_snapshot_tier", "")), "vram")
            self.assertEqual(int(xargs.get("korith_prompt_tokens", 0)), 1024)
            self.assertEqual(int(xargs.get("korith_priority", -1)), 3)
            self.assertEqual(int(xargs.get("korith_spec_enabled", 0)), 1)
            self.assertEqual(int(xargs.get("korith_spec_k", 0)), 6)
            self.assertAlmostEqual(float(xargs.get("korith_spec_min_accept", 0.0)), 0.7, places=6)
            self.assertEqual(int(xargs.get("korith_spec_disable_after_n", 0)), 48)
            self.assertEqual(int(xargs.get("korith_spec_cache_only", 0)), 0)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
