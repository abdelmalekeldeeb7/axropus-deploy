"""Tests for platform/reasoning/reasoning_model.py (Component 2).

All tests run without a GPU by mocking transformers and torch.
"""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from platform.reasoning.reasoning_model import (
    DecisionVector,
    UnifiedDecisionVector,
    ReasoningModel,
    _REQUIRED_KEYS,
    _safe_fallback,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _zero_tensor() -> np.ndarray:
    return np.zeros(64, dtype=np.float32)


def _rand_tensor() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random(64).astype(np.float32)


def _valid_two_section_dict() -> dict:
    return {
        "prefill": {
            "cache_admission": True,
            "admission_priority": 0.8,
            "eviction_target": None,
            "pre_warm_predictions": ["abc123"],
            "route_decision": "gpu-0",
            "batch_group": None,
            "throttle_back_pressure": 0.0,
        },
        "decode": {
            "spec_decode_enable": True,
            "spec_decode_depth": 4,
            "spec_mode": "v2_clean",
            "spec_v3_block_size": 8,
            "draft_temperature": 0.8,
            "kv_compression_enable": False,
            "kv_compression_ratio": 1.0,
            "decode_batch_priority": 2,
            "early_stop_confidence": 0.0,
            "cuda_graph_hint": True,
            "adaptive_depth_override": False,
        },
    }


def _valid_json_bytes() -> bytes:
    return json.dumps(_valid_two_section_dict()).encode()


def _make_model(**kwargs) -> ReasoningModel:
    return ReasoningModel(
        model_name="test-model",
        device="cpu",
        max_inference_ms=kwargs.get("max_inference_ms", 5000.0),
        load_in_4bit=kwargs.get("load_in_4bit", False),
    )


# ── _safe_fallback ─────────────────────────────────────────────────────────────

class TestSafeFallback:
    def test_is_fallback_true(self):
        dv = _safe_fallback(10.0)
        assert dv.is_fallback is True

    def test_inference_ms(self):
        dv = _safe_fallback(42.5)
        assert dv.inference_ms == pytest.approx(42.5)

    def test_neutral_values(self):
        dv = _safe_fallback()
        assert dv.cache_admission is True
        assert dv.eviction_target is None
        assert dv.pre_warm_predictions == []
        assert dv.route_decision is None
        assert dv.throttle_back_pressure == pytest.approx(0.0)
        assert dv.spec_decode_depth == 4
        assert dv.adaptive_depth_override is False   # Rust scheduler stays in control

    def test_all_fields_present(self):
        dv = _safe_fallback()
        # UnifiedDecisionVector fields
        assert hasattr(dv, "cache_admission")
        assert hasattr(dv, "admission_priority")
        assert hasattr(dv, "eviction_target")
        assert hasattr(dv, "pre_warm_predictions")
        assert hasattr(dv, "route_decision")
        assert hasattr(dv, "batch_group")
        assert hasattr(dv, "throttle_back_pressure")
        assert hasattr(dv, "inference_ms")
        assert hasattr(dv, "is_fallback")
        assert hasattr(dv, "domain")
        assert hasattr(dv, "spec_decode_enable")
        assert hasattr(dv, "spec_decode_depth")
        assert hasattr(dv, "spec_mode")
        assert hasattr(dv, "adaptive_depth_override")
        assert hasattr(dv, "kv_compression_enable")


# ── DecisionVector ─────────────────────────────────────────────────────────────

class TestDecisionVector:
    def test_to_dict_has_all_keys(self):
        dv = _safe_fallback()
        d = dv.to_dict()
        assert "cache_admission" in d
        assert "admission_priority" in d
        assert "eviction_target" in d
        assert "pre_warm_predictions" in d
        assert "route_decision" in d
        assert "batch_group" in d
        assert "spec_decode_enable" in d
        assert "spec_decode_depth" in d
        assert "spec_mode" in d
        assert "adaptive_depth_override" in d
        assert "throttle_back_pressure" in d
        assert "inference_ms" in d
        assert "is_fallback" in d
        assert "domain" in d

    def test_to_dict_json_serialisable(self):
        dv = _safe_fallback(5.0)
        assert json.dumps(dv.to_dict())  # must not raise

    def test_pre_warm_predictions_is_list(self):
        dv = _safe_fallback()
        assert isinstance(dv.pre_warm_predictions, list)


# ── decide() when not loaded ───────────────────────────────────────────────────

class TestDecideNotLoaded:
    def test_returns_fallback(self):
        m = _make_model()
        dv = m.decide(_zero_tensor())
        assert dv.is_fallback is True

    def test_inference_ms_is_zero(self):
        m = _make_model()
        dv = m.decide(_zero_tensor())
        assert dv.inference_ms == pytest.approx(0.0)


# ── _tensor_to_prompt ──────────────────────────────────────────────────────────

class TestTensorToPrompt:
    def test_zero_tensor_produces_empty_sections(self):
        m = _make_model()
        prefill_json, decode_json = m._tensor_to_prompt(_zero_tensor())
        # All zero slots should be omitted (below 0.001 threshold)
        assert json.loads(prefill_json) == {}
        assert json.loads(decode_json) == {}

    def test_returns_tuple_of_two_strings(self):
        m = _make_model()
        result = m._tensor_to_prompt(_zero_tensor())
        assert isinstance(result, tuple)
        assert len(result) == 2
        prefill_json, decode_json = result
        assert isinstance(prefill_json, str)
        assert isinstance(decode_json, str)

    def test_nonzero_prefill_slots_included(self):
        m = _make_model()
        t = _zero_tensor()
        t[2] = 0.998   # hit_rate slot
        t[10] = 0.42   # storage_util slot
        prefill_json, _ = m._tensor_to_prompt(t)
        obj = json.loads(prefill_json)
        assert "hit_rate" in obj
        assert "storage_util" in obj
        assert obj["hit_rate"] == pytest.approx(0.998, abs=0.001)

    def test_nonzero_decode_slots_included(self):
        m = _make_model()
        t = _zero_tensor()
        t[32] = 0.75   # ACCEPTANCE_EMA
        t[35] = 0.50   # ENTROPY_EMA
        _, decode_json = m._tensor_to_prompt(t)
        obj = json.loads(decode_json)
        assert "accept_ema" in obj
        assert "entropy" in obj

    def test_small_values_omitted(self):
        m = _make_model()
        t = _zero_tensor()
        t[2] = 0.0005  # below 0.001 threshold
        prefill_json, _ = m._tensor_to_prompt(t)
        obj = json.loads(prefill_json)
        assert "hit_rate" not in obj

    def test_returns_valid_json(self):
        m = _make_model()
        prefill_json, decode_json = m._tensor_to_prompt(_rand_tensor())
        assert isinstance(json.loads(prefill_json), dict)
        assert isinstance(json.loads(decode_json), dict)

    def test_values_rounded_to_4dp(self):
        m = _make_model()
        t = _zero_tensor()
        t[2] = 0.99876543
        prefill_json, _ = m._tensor_to_prompt(t)
        obj = json.loads(prefill_json)
        assert str(obj["hit_rate"]) == str(round(0.99876543, 4))


# ── _parse_output ──────────────────────────────────────────────────────────────

class TestParseOutput:
    def _valid_dict(self) -> dict:
        return _valid_two_section_dict()

    def test_valid_json_returns_dict(self):
        text = json.dumps(self._valid_dict())
        result = ReasoningModel._parse_output(text)
        assert result is not None
        assert isinstance(result, dict)

    def test_invalid_json_returns_none(self):
        result = ReasoningModel._parse_output("not json at all")
        assert result is None

    def test_empty_string_returns_none(self):
        result = ReasoningModel._parse_output("")
        assert result is None

    def test_missing_prefill_section_returns_none(self):
        d = {"decode": self._valid_dict()["decode"]}
        result = ReasoningModel._parse_output(json.dumps(d))
        assert result is None

    def test_missing_decode_section_returns_none(self):
        d = {"prefill": self._valid_dict()["prefill"]}
        result = ReasoningModel._parse_output(json.dumps(d))
        assert result is None

    def test_missing_prefill_subkey_returns_none(self):
        d = self._valid_dict()
        del d["prefill"]["cache_admission"]
        result = ReasoningModel._parse_output(json.dumps(d))
        assert result is None

    def test_missing_decode_subkey_returns_none(self):
        d = self._valid_dict()
        del d["decode"]["spec_decode_enable"]
        result = ReasoningModel._parse_output(json.dumps(d))
        assert result is None

    def test_extracts_from_surrounding_text(self):
        d = self._valid_dict()
        text = f"Here is my decision:\n{json.dumps(d)}\nThat's it."
        result = ReasoningModel._parse_output(text)
        assert result is not None

    def test_no_opening_brace_returns_none(self):
        result = ReasoningModel._parse_output('"key": "value"')
        assert result is None

    def test_all_required_keys_present(self):
        d = self._valid_dict()
        result = ReasoningModel._parse_output(json.dumps(d))
        for key in _REQUIRED_KEYS:
            assert key in result


# ── _dict_to_decision_vector ───────────────────────────────────────────────────

class TestDictToDecisionVector:
    def _base(self) -> dict:
        return {
            "prefill": {
                "cache_admission": True,
                "admission_priority": 0.75,
                "eviction_target": "prefix_abc",
                "pre_warm_predictions": ["h1", "h2", "h3"],
                "route_decision": "gpu-0",
                "batch_group": "grp1",
                "throttle_back_pressure": 0.3,
            },
            "decode": {
                "spec_decode_enable": False,
                "spec_decode_depth": 6,
                "spec_mode": "v3_block",
                "spec_v3_block_size": 16,
                "draft_temperature": 0.7,
                "kv_compression_enable": False,
                "kv_compression_ratio": 1.0,
                "decode_batch_priority": 3,
                "early_stop_confidence": 0.0,
                "cuda_graph_hint": True,
                "adaptive_depth_override": False,
            },
        }

    def test_basic_mapping(self):
        dv = ReasoningModel._dict_to_decision_vector(self._base(), 15.0)
        assert dv.cache_admission is True
        assert dv.admission_priority == pytest.approx(0.75)
        assert dv.eviction_target == "prefix_abc"
        assert dv.pre_warm_predictions == ["h1", "h2", "h3"]
        assert dv.route_decision == "gpu-0"
        assert dv.batch_group == "grp1"
        assert dv.spec_decode_enable is False
        assert dv.spec_decode_depth == 6
        assert dv.spec_mode == "v3_block"
        assert dv.throttle_back_pressure == pytest.approx(0.3)
        assert dv.inference_ms == pytest.approx(15.0)
        assert dv.is_fallback is False
        assert dv.domain == "unified"

    def test_admission_priority_clamped(self):
        d = self._base()
        d["prefill"]["admission_priority"] = 99.9
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.admission_priority == pytest.approx(1.0)

    def test_admission_priority_negative_clamped(self):
        d = self._base()
        d["prefill"]["admission_priority"] = -5.0
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.admission_priority == pytest.approx(0.0)

    def test_spec_decode_depth_clamped_max(self):
        d = self._base()
        d["decode"]["spec_decode_depth"] = 100
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.spec_decode_depth == 16

    def test_spec_decode_depth_clamped_min(self):
        d = self._base()
        d["decode"]["spec_decode_depth"] = 0
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.spec_decode_depth == 1

    def test_null_eviction_target(self):
        d = self._base()
        d["prefill"]["eviction_target"] = None
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.eviction_target is None

    def test_pre_warm_truncated_to_5(self):
        d = self._base()
        d["prefill"]["pre_warm_predictions"] = ["a", "b", "c", "d", "e", "f", "g"]
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert len(dv.pre_warm_predictions) == 5

    def test_pre_warm_non_list_becomes_empty(self):
        d = self._base()
        d["prefill"]["pre_warm_predictions"] = "not-a-list"
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.pre_warm_predictions == []

    def test_throttle_clamped(self):
        d = self._base()
        d["prefill"]["throttle_back_pressure"] = 2.5
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.throttle_back_pressure == pytest.approx(1.0)

    def test_invalid_spec_mode_defaults_to_v2_clean(self):
        d = self._base()
        d["decode"]["spec_mode"] = "nonsense_mode"
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.spec_mode == "v2_clean"

    def test_adaptive_depth_override_false_by_default(self):
        d = self._base()
        d["decode"]["adaptive_depth_override"] = False
        dv = ReasoningModel._dict_to_decision_vector(d, 0.0)
        assert dv.adaptive_depth_override is False


# ── decide() with mocked model ────────────────────────────────────────────────

def _make_loaded_model_mocks(generated_text: str):
    """
    Returns (mock_model, mock_tokenizer) that simulate transformers objects.
    `generated_text` is what the model "generates".
    """
    import torch

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 2
    mock_tokenizer.eos_token_id = 2

    # Tokenizer returns a simple input_ids tensor
    mock_enc = MagicMock()
    mock_enc.input_ids = torch.zeros((1, 10), dtype=torch.long)
    mock_tokenizer.return_value = mock_enc
    mock_tokenizer.side_effect = None

    # decode returns the generated text
    mock_tokenizer.decode.return_value = generated_text

    # model.generate returns a tensor where first 10 tokens = input
    gen_ids = torch.zeros((1, 20), dtype=torch.long)
    mock_model = MagicMock()
    mock_model.generate.return_value = gen_ids

    # model() for prewarm
    mock_past = tuple(
        (torch.zeros(1, 8, 10, 64), torch.zeros(1, 8, 10, 64))
        for _ in range(32)
    )
    mock_fwd = MagicMock()
    mock_fwd.past_key_values = mock_past
    mock_model.return_value = mock_fwd

    return mock_model, mock_tokenizer


class TestDecideWithMockedModel:
    def _valid_output(self) -> str:
        return json.dumps({
            "prefill": {
                "cache_admission": True,
                "admission_priority": 0.9,
                "eviction_target": None,
                "pre_warm_predictions": ["abc", "def"],
                "route_decision": "gpu-1",
                "batch_group": None,
                "throttle_back_pressure": 0.1,
            },
            "decode": {
                "spec_decode_enable": True,
                "spec_decode_depth": 5,
                "spec_mode": "v2_clean",
                "spec_v3_block_size": 8,
                "draft_temperature": 0.8,
                "kv_compression_enable": False,
                "kv_compression_ratio": 1.0,
                "decode_batch_priority": 2,
                "early_stop_confidence": 0.0,
                "cuda_graph_hint": True,
                "adaptive_depth_override": False,
            },
        })

    def _load_model(self, m: ReasoningModel, mock_model, mock_tokenizer) -> None:
        """Force-load without hitting transformers/torch."""
        import torch
        m._model = mock_model
        m._tokenizer = mock_tokenizer
        # Minimal KV cache
        mock_past = tuple(
            (torch.zeros(1, 8, 5, 64), torch.zeros(1, 8, 5, 64))
            for _ in range(4)
        )
        m._system_kv_cache = mock_past
        m._loaded = True

    def test_valid_output_produces_non_fallback(self):
        import torch
        m = _make_model(max_inference_ms=5000.0)
        mock_model, mock_tokenizer = _make_loaded_model_mocks(self._valid_output())
        self._load_model(m, mock_model, mock_tokenizer)

        mock_enc = MagicMock()
        mock_enc.input_ids = torch.zeros((1, 10), dtype=torch.long)
        mock_tokenizer.return_value = mock_enc

        dv = m.decide(_rand_tensor())
        assert dv.is_fallback is False
        assert dv.cache_admission is True
        assert dv.admission_priority == pytest.approx(0.9, abs=0.001)
        assert dv.route_decision == "gpu-1"
        assert len(dv.pre_warm_predictions) == 2
        assert dv.spec_decode_depth == 5
        assert dv.adaptive_depth_override is False

    def test_invalid_json_output_returns_fallback(self):
        import torch
        m = _make_model(max_inference_ms=5000.0)
        mock_model, mock_tokenizer = _make_loaded_model_mocks("this is not json")
        self._load_model(m, mock_model, mock_tokenizer)

        mock_enc = MagicMock()
        mock_enc.input_ids = torch.zeros((1, 10), dtype=torch.long)
        mock_tokenizer.return_value = mock_enc

        dv = m.decide(_rand_tensor())
        assert dv.is_fallback is True

    def test_generate_exception_returns_fallback(self):
        import torch
        m = _make_model(max_inference_ms=5000.0)
        mock_model, mock_tokenizer = _make_loaded_model_mocks("")
        self._load_model(m, mock_model, mock_tokenizer)
        mock_model.generate.side_effect = RuntimeError("OOM")

        mock_enc = MagicMock()
        mock_enc.input_ids = torch.zeros((1, 10), dtype=torch.long)
        mock_tokenizer.return_value = mock_enc

        dv = m.decide(_rand_tensor())
        assert dv.is_fallback is True

    def test_timeout_returns_fallback(self):
        """Simulate timeout by making generate sleep longer than the deadline."""
        import time as _t
        import torch

        m = _make_model(max_inference_ms=5000.0)
        mock_model, mock_tokenizer = _make_loaded_model_mocks(self._valid_output())
        self._load_model(m, mock_model, mock_tokenizer)
        # Override deadline to 1ms AFTER construction (avoids any floor clamping)
        m._max_inference_ms = 1.0

        gen_ids = torch.zeros((1, 20), dtype=torch.long)

        def slow_generate(*args, **kwargs):
            _t.sleep(0.01)  # 10ms — always exceeds 1ms deadline
            return gen_ids

        mock_model.generate.side_effect = slow_generate

        mock_enc = MagicMock()
        mock_enc.input_ids = torch.zeros((1, 10), dtype=torch.long)
        mock_tokenizer.return_value = mock_enc

        dv = m.decide(_rand_tensor())
        assert dv.is_fallback is True
        assert dv.inference_ms > 1.0

    def test_inference_ms_is_recorded(self):
        import torch
        m = _make_model(max_inference_ms=5000.0)
        mock_model, mock_tokenizer = _make_loaded_model_mocks(self._valid_output())
        self._load_model(m, mock_model, mock_tokenizer)

        mock_enc = MagicMock()
        mock_enc.input_ids = torch.zeros((1, 10), dtype=torch.long)
        mock_tokenizer.return_value = mock_enc

        dv = m.decide(_rand_tensor())
        assert dv.inference_ms >= 0.0
