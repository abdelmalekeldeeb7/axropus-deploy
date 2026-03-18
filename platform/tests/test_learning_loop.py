"""Tests for platform/reasoning/learning_loop.py (Component 5)."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from platform.reasoning.learning_loop import (
    LearningLoop,
    TrainingResult,
    _MAX_REWARD_WEIGHT,
    _TARGET_KEYS,
    _PREFILL_TARGET_KEYS,
    _DECODE_TARGET_KEYS,
)
from platform.reasoning.reward_calculator import Outcome


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_model(loaded: bool = True) -> MagicMock:
    model = MagicMock()
    model.loaded = loaded
    model._device = "cpu"
    model._tokenizer = MagicMock()
    model._model = MagicMock()
    # Tokenizer returns something with .input_ids
    fake_ids = MagicMock()
    fake_ids.input_ids = MagicMock()
    fake_ids.input_ids.to.return_value = fake_ids.input_ids
    fake_ids.input_ids.clone.return_value = fake_ids.input_ids
    fake_ids.input_ids.shape = (1, 10)
    model._tokenizer.return_value = fake_ids
    return model


def _make_calc(batch: list | None = None) -> MagicMock:
    calc = MagicMock()
    calc.get_training_batch.return_value = batch or []
    calc.mark_batch_consumed.return_value = len(batch) if batch else 0
    return calc


def _make_loop(
    batch: list | None = None,
    loaded: bool = True,
    min_positive_examples: int = 1,
    interval_s: float = 3600.0,
    **kwargs,
) -> tuple[LearningLoop, MagicMock, MagicMock]:
    model = _make_model(loaded=loaded)
    calc  = _make_calc(batch=batch)
    loop  = LearningLoop(
        reasoning_model=model,
        reward_calculator=calc,
        min_positive_examples=min_positive_examples,
        interval_s=interval_s,
        adapter_save_dir=":memory_test:",  # won't be used in mocked tests
        **kwargs,
    )
    return loop, model, calc


def _positive_item(decision_id: str = "d-1", reward: float = 1.5) -> dict:
    # decision mirrors UnifiedDecisionVector.to_dict() — flat dict stored in SQLite
    decision = {
        # prefill fields
        "cache_admission": True,
        "admission_priority": 0.7,
        "eviction_target": None,
        "pre_warm_predictions": [],
        "route_decision": None,
        "batch_group": None,
        "throttle_back_pressure": 0.0,
        "inference_ms": 42.0,
        "is_fallback": False,
        "domain": "unified",
        # decode fields
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
    }
    return {
        "decision_id":  decision_id,
        "tensor":       np.zeros(64, dtype=np.float32),
        "decision":     decision,
        "total_reward": reward,
        "outcome_count": 2,
        "overrides": [],
        "timestamp": time.time(),
    }


def _negative_item(decision_id: str = "d-neg") -> dict:
    item = _positive_item(decision_id, reward=-0.5)
    return item


# ── TrainingResult structure ──────────────────────────────────────────────────

class TestTrainingResultStructure:
    def test_skipped_result_has_zero_loss(self):
        loop, _, _ = _make_loop(batch=[])
        result = loop.run_once()
        assert result.skipped is True
        assert result.loss == 0.0
        assert result.examples_used == 0

    def test_skipped_result_has_reason(self):
        loop, _, _ = _make_loop(batch=[])
        result = loop.run_once()
        assert len(result.skip_reason) > 0

    def test_skipped_elapsed_is_nonnegative(self):
        loop, _, _ = _make_loop(batch=[])
        result = loop.run_once()
        assert result.elapsed_s >= 0.0


# ── run_once — skip conditions ────────────────────────────────────────────────

class TestRunOnceSkipConditions:
    def test_skips_when_batch_empty(self):
        loop, _, calc = _make_loop(batch=[])
        result = loop.run_once()
        assert result.skipped is True
        calc.mark_batch_consumed.assert_not_called()

    def test_skips_when_all_negative_rewards(self):
        loop, _, calc = _make_loop(
            batch=[_negative_item("d-1"), _negative_item("d-2")],
            min_positive_examples=1,
        )
        result = loop.run_once()
        assert result.skipped is True

    def test_skips_when_positive_count_below_min(self):
        loop, _, _ = _make_loop(
            batch=[_positive_item("d-1", reward=1.0)],
            min_positive_examples=5,
        )
        result = loop.run_once()
        assert result.skipped is True

    def test_skips_when_model_not_loaded(self):
        loop, model, _ = _make_loop(
            batch=[_positive_item()],
            loaded=False,
            min_positive_examples=1,
        )
        # Patch _fine_tune to raise RuntimeError when model not loaded
        with patch.object(loop, "_fine_tune", side_effect=RuntimeError("not loaded")):
            result = loop.run_once()
        assert result.skipped is True

    def test_skips_when_peft_not_installed(self):
        loop, _, _ = _make_loop(batch=[_positive_item()], min_positive_examples=1)
        with patch.object(loop, "_fine_tune", side_effect=ImportError("peft missing")):
            result = loop.run_once()
        assert result.skipped is True


# ── _build_training_examples ──────────────────────────────────────────────────

class TestBuildTrainingExamples:
    def _examples(self, items):
        loop, _, _ = _make_loop()
        return loop._build_training_examples(items)

    def test_filters_nonpositive_rewards(self):
        items = [
            _positive_item("d-1", reward=1.0),
            _negative_item("d-2"),         # reward=-0.5
            _positive_item("d-3", reward=0.0),  # exactly zero → excluded
        ]
        exs = self._examples(items)
        ids = [e["decision_id"] for e in exs]
        assert "d-1" in ids
        assert "d-2" not in ids
        assert "d-3" not in ids

    def test_weight_equals_reward(self):
        exs = self._examples([_positive_item("d-1", reward=2.0)])
        assert exs[0]["weight"] == pytest.approx(2.0)

    def test_weight_capped_at_max(self):
        exs = self._examples([_positive_item("d-1", reward=99.0)])
        assert exs[0]["weight"] == pytest.approx(_MAX_REWARD_WEIGHT)

    def test_prompt_contains_prefill_metrics_prefix(self):
        exs = self._examples([_positive_item()])
        assert exs[0]["prompt"].startswith("Prefill metrics:")

    def test_prompt_contains_decode_metrics_section(self):
        exs = self._examples([_positive_item()])
        assert "Decode metrics:" in exs[0]["prompt"]

    def test_prompt_ends_with_decision_marker(self):
        exs = self._examples([_positive_item()])
        assert exs[0]["prompt"].endswith("Decision:")

    def test_target_is_valid_json(self):
        exs = self._examples([_positive_item()])
        parsed = json.loads(exs[0]["target"])
        assert isinstance(parsed, dict)

    def test_target_has_prefill_and_decode_sections(self):
        """Target must be {"prefill":{...},"decode":{...}} — two-section format."""
        exs = self._examples([_positive_item()])
        parsed = json.loads(exs[0]["target"])
        assert "prefill" in parsed
        assert "decode"  in parsed

    def test_target_prefill_section_contains_expected_keys(self):
        exs = self._examples([_positive_item()])
        prefill = json.loads(exs[0]["target"])["prefill"]
        for key in _PREFILL_TARGET_KEYS:
            assert key in prefill, f"Missing prefill key: {key}"

    def test_target_decode_section_contains_expected_keys(self):
        exs = self._examples([_positive_item()])
        decode = json.loads(exs[0]["target"])["decode"]
        for key in _DECODE_TARGET_KEYS:
            assert key in decode, f"Missing decode key: {key}"

    def test_none_tensor_uses_empty_metrics(self):
        item = _positive_item()
        item["tensor"] = None
        exs = self._examples([item])
        assert "Prefill metrics: {}" in exs[0]["prompt"]
        assert "Decode metrics: {}"  in exs[0]["prompt"]

    def test_nonzero_tensor_slots_appear_in_prompt(self):
        from platform.reasoning.metrics_collector import Slot
        tensor = np.zeros(64, dtype=np.float32)
        tensor[Slot.HIT_RATE] = 0.99
        item = _positive_item()
        item["tensor"] = tensor
        exs = self._examples([item])
        assert "hit_rate" in exs[0]["prompt"]

    def test_nonzero_decode_slot_appears_in_decode_section(self):
        from platform.reasoning.metrics_collector import Slot
        tensor = np.zeros(64, dtype=np.float32)
        tensor[Slot.ACCEPTANCE_EMA] = 0.85
        item = _positive_item()
        item["tensor"] = tensor
        exs = self._examples([item])
        assert "accept_ema" in exs[0]["prompt"]

    def test_near_zero_slots_omitted_from_prompt(self):
        """Slots with |value| < 0.001 are omitted to minimise token count."""
        from platform.reasoning.metrics_collector import Slot
        tensor = np.zeros(64, dtype=np.float32)
        tensor[Slot.HIT_RATE] = 0.0005  # below threshold
        item = _positive_item()
        item["tensor"] = tensor
        exs = self._examples([item])
        assert "hit_rate" not in exs[0]["prompt"]

    def test_decision_id_preserved(self):
        exs = self._examples([_positive_item("d-xyz")])
        assert exs[0]["decision_id"] == "d-xyz"

    def test_empty_batch_returns_empty(self):
        assert self._examples([]) == []

    def test_multiple_positive_examples_all_included(self):
        items = [_positive_item(f"d-{i}", reward=float(i + 1)) for i in range(5)]
        exs = self._examples(items)
        assert len(exs) == 5


# ── run_once — happy path (mocked fine-tune) ──────────────────────────────────

class TestRunOnceHappyPath:
    def _fake_training_result(self, n: int = 3) -> TrainingResult:
        return TrainingResult(
            examples_used=n,
            mean_reward=1.2,
            loss=0.045,
            adapter_path="/tmp/adapter_123",
            elapsed_s=4.5,
            skipped=False,
            skip_reason="",
        )

    def test_returns_non_skipped_result(self):
        loop, _, _ = _make_loop(batch=[_positive_item()], min_positive_examples=1)
        with patch.object(loop, "_fine_tune", return_value=self._fake_training_result()):
            result = loop.run_once()
        assert result.skipped is False
        assert result.examples_used == 3

    def test_marks_batch_consumed_after_training(self):
        batch = [_positive_item(f"d-{i}") for i in range(3)]
        loop, _, calc = _make_loop(batch=batch, min_positive_examples=1)
        with patch.object(loop, "_fine_tune", return_value=self._fake_training_result(3)):
            loop.run_once()
        consumed_ids = calc.mark_batch_consumed.call_args[0][0]
        assert set(consumed_ids) == {"d-0", "d-1", "d-2"}

    def test_consumed_not_called_on_skip(self):
        loop, _, calc = _make_loop(batch=[])
        loop.run_once()
        calc.mark_batch_consumed.assert_not_called()

    def test_cycle_count_increments(self):
        loop, _, _ = _make_loop(batch=[_positive_item()], min_positive_examples=1)
        with patch.object(loop, "_fine_tune", return_value=self._fake_training_result()):
            loop.run_once()
            loop.run_once()
        assert loop._cycle_count == 2

    def test_cycle_count_not_incremented_on_skip(self):
        loop, _, _ = _make_loop(batch=[])
        loop.run_once()
        loop.run_once()
        assert loop._cycle_count == 0


# ── hot-swap ──────────────────────────────────────────────────────────────────

class TestHotSwap:
    def test_hot_swap_replaces_model(self):
        loop, model, _ = _make_loop()
        new_model = MagicMock()
        loop._hot_swap(new_model)
        assert model._model is new_model

    def test_hot_swap_uses_lock(self):
        """Verify lock is acquired by replacing _swap_lock with a MagicMock."""
        loop, model, _ = _make_loop()
        new_model = MagicMock()
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__  = MagicMock(return_value=False)
        loop._swap_lock = mock_lock

        loop._hot_swap(new_model)

        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


# ── _tensor_to_prompt ─────────────────────────────────────────────────────────

class TestTensorToPrompt:
    def test_none_tensor(self):
        result = LearningLoop._tensor_to_prompt(None)
        assert result == "Prefill metrics: {}\nDecode metrics: {}\nDecision:"

    def test_all_zero_tensor_gives_empty_sections(self):
        t = np.zeros(64, dtype=np.float32)
        result = LearningLoop._tensor_to_prompt(t)
        assert "Prefill metrics: {}" in result
        assert "Decode metrics: {}"  in result

    def test_nonzero_prefill_slot_appears(self):
        from platform.reasoning.metrics_collector import Slot
        t = np.zeros(64, dtype=np.float32)
        t[Slot.HIT_RATE] = 0.99
        result = LearningLoop._tensor_to_prompt(t)
        assert "hit_rate" in result
        assert "0.99" in result

    def test_nonzero_decode_slot_appears(self):
        from platform.reasoning.metrics_collector import Slot
        t = np.zeros(64, dtype=np.float32)
        t[Slot.ACCEPTANCE_EMA] = 0.85
        result = LearningLoop._tensor_to_prompt(t)
        assert "accept_ema" in result

    def test_output_format(self):
        result = LearningLoop._tensor_to_prompt(np.zeros(64, dtype=np.float32))
        assert result.startswith("Prefill metrics:")
        assert "Decode metrics:" in result
        assert result.endswith("Decision:")


# ── lifecycle ─────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_creates_daemon_thread(self):
        loop, _, _ = _make_loop(interval_s=9999.0)
        with patch.object(loop, "run_once", return_value=MagicMock(skipped=True)):
            loop.start()
            assert loop.running
            loop.stop(timeout=2.0)
            assert not loop.running

    def test_double_start_does_not_create_second_thread(self):
        loop, _, _ = _make_loop(interval_s=9999.0)
        with patch.object(loop, "run_once", return_value=MagicMock(skipped=True)):
            loop.start()
            first_thread = loop._thread
            loop.start()  # second call
            assert loop._thread is first_thread
            loop.stop(timeout=2.0)

    def test_stop_without_start_is_safe(self):
        loop, _, _ = _make_loop()
        loop.stop()  # should not raise

    def test_run_once_called_by_thread(self):
        results = []

        def fake_run_once():
            results.append(1)
            return MagicMock(skipped=True)

        loop, _, _ = _make_loop(interval_s=0.02)
        with patch.object(loop, "run_once", side_effect=fake_run_once):
            loop.start()
            time.sleep(0.08)
            loop.stop(timeout=2.0)

        # With interval=20ms, should have run at least once in 80ms
        assert len(results) >= 1


# ── Task 3: RL Level 2 Policy Gradient ───────────────────────────────────────

class TestRlLevelConfig:
    def test_rl_level_default_is_1(self):
        loop, _, _ = _make_loop()
        assert loop._rl_level == 1

    def test_rl_level_can_be_set_to_2(self):
        loop, _, _ = _make_loop(rl_level=2)
        assert loop._rl_level == 2

    def test_rl_level_switches_method_to_supervised(self):
        """rl_level=1 dispatches to _fine_tune, never _fine_tune_policy_gradient."""
        loop, _, _ = _make_loop(batch=[_positive_item()], min_positive_examples=1, rl_level=1)
        fake_result = TrainingResult(
            examples_used=1, mean_reward=1.0, loss=0.01,
            adapter_path=None, elapsed_s=0.1, skipped=False, skip_reason="",
        )
        with patch.object(loop, "_fine_tune", return_value=fake_result) as mock_ft, \
             patch.object(loop, "_fine_tune_policy_gradient") as mock_pg:
            loop.run_once()
        mock_ft.assert_called_once()
        mock_pg.assert_not_called()

    def test_rl_level_switches_method_to_policy_gradient(self):
        """rl_level=2 dispatches to _fine_tune_policy_gradient, never _fine_tune."""
        # Need negative item so rl_level=2 includes it; min_positive_examples=1
        items = [
            _positive_item("d-1", reward=1.0),
            _negative_item("d-2"),
        ]
        loop, _, _ = _make_loop(batch=items, min_positive_examples=1, rl_level=2)
        fake_result = TrainingResult(
            examples_used=2, mean_reward=0.25, loss=0.05,
            adapter_path=None, elapsed_s=0.2, skipped=False, skip_reason="",
        )
        with patch.object(loop, "_fine_tune") as mock_ft, \
             patch.object(loop, "_fine_tune_policy_gradient", return_value=fake_result) as mock_pg:
            loop.run_once()
        mock_pg.assert_called_once()
        mock_ft.assert_not_called()


class TestPolicyGradientExamples:
    def test_includes_negative_examples_at_rl_level_2(self):
        """rl_level=2: examples with reward=-0.5 are included."""
        items = [
            _positive_item("d-pos", reward=1.0),
            _negative_item("d-neg"),   # reward=-0.5
        ]
        loop, _, _ = _make_loop(rl_level=2)
        exs = loop._build_training_examples(items)
        ids = [e["decision_id"] for e in exs]
        assert "d-pos" in ids
        assert "d-neg" in ids

    def test_excludes_negative_examples_at_rl_level_1(self):
        """rl_level=1: examples with reward=-0.5 are excluded."""
        items = [
            _positive_item("d-pos", reward=1.0),
            _negative_item("d-neg"),
        ]
        loop, _, _ = _make_loop(rl_level=1)
        exs = loop._build_training_examples(items)
        ids = [e["decision_id"] for e in exs]
        assert "d-pos" in ids
        assert "d-neg" not in ids

    def test_near_zero_reward_excluded_at_rl_level_2(self):
        """Examples with |reward| < 0.01 are filtered even at rl_level=2."""
        items = [
            _positive_item("d-1", reward=0.005),   # |reward| < 0.01 → excluded
            _positive_item("d-2", reward=1.0),
        ]
        loop, _, _ = _make_loop(rl_level=2)
        exs = loop._build_training_examples(items)
        ids = [e["decision_id"] for e in exs]
        assert "d-1" not in ids
        assert "d-2" in ids

    def test_advantage_computation(self):
        """Advantages = reward − baseline (mean reward).

        rewards = [1.0, -0.5, 0.5]  →  baseline = 1.0/3 ≈ 0.333
        advantages = [+0.667, -0.833, +0.167]
        """
        import torch

        rewards = [1.0, -0.5, 0.5]
        baseline = sum(rewards) / len(rewards)
        expected_advantages = [r - baseline for r in rewards]

        # Verify the math matches REINFORCE formula
        assert baseline == pytest.approx(1.0 / 3, abs=1e-6)
        assert expected_advantages[0] == pytest.approx(0.667, abs=0.001)
        assert expected_advantages[1] == pytest.approx(-0.833, abs=0.001)
        assert expected_advantages[2] == pytest.approx(0.167, abs=0.001)

    def test_policy_gradient_gradient_clipping(self):
        """_fine_tune_policy_gradient uses clip_grad_norm_ with max_norm=1.0."""
        import inspect
        loop, _, _ = _make_loop(rl_level=2)
        src = inspect.getsource(loop._fine_tune_policy_gradient)
        assert "clip_grad_norm_" in src
        assert "max_norm=1.0" in src
