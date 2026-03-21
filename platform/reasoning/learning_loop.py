"""
Component 5: Continuous Learning Loop
======================================
Hourly LoRA fine-tuning that lets the reasoning model learn from its own decisions.

How it works
------------
1. Background daemon thread wakes every `interval_s` (default: 3600 s).
2. Calls RewardCalculator.get_training_batch() — only decisions with ≥1 outcome.
3. Filters to positive-reward examples only (reward > 0).  Negative outcomes
   teach nothing useful for supervised fine-tuning; they are already implicit
   punishment because the model won't see those (prompt, decision) pairs again.
4. Builds (prompt → JSON_decision) supervised pairs, weighted by reward magnitude.
5. Applies LoRA fine-tuning (rank 16, alpha 16) on the live model — same weights
   that are serving inference.
6. Saves the trained adapter to disk, then hot-swaps it into the running model
   without restarting.  The base model weights never change.
7. Marks the consumed decisions in SQLite so they aren't replayed.

Hot-swap safety
---------------
CPython's GIL makes attribute assignment atomic at the object level, so replacing
`reasoning_model._model` is safe for the common case.  For production multi-GPU
setups the Orchestrator should gate the swap between requests using the model's
internal lock (Component 6 adds this coordination).

LoRA note
---------
peft (HuggingFace PEFT library) is required for fine-tuning.  If not installed,
run_once() logs a warning and returns a skipped TrainingResult.  Inference is
unaffected — the base model continues running.

Usage
-----
    loop = LearningLoop(
        reasoning_model=model,
        reward_calculator=calc,
        interval_s=3600.0,
    )
    loop.start()          # begins background fine-tuning thread
    ...
    loop.stop()           # clean shutdown

    # or trigger manually:
    result = loop.run_once()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_LORA_TARGET_MODULES = ["q_proj", "v_proj"]   # attention layers only — fast, effective
_MAX_REWARD_WEIGHT   = 3.0                      # cap per-example weight to prevent outliers
_DEFAULT_LR          = 2e-4
_DEFAULT_ADAPTER_DIR = Path("amf_lora_adapters")

# Prefill keys the model outputs in the "prefill" section of the two-section JSON
_PREFILL_TARGET_KEYS = [
    "cache_admission",
    "admission_priority",
    "eviction_target",
    "pre_warm_predictions",
    "route_decision",
    "batch_group",
    "throttle_back_pressure",
    # Dynamo fleet fields (added in Prompt 3)
    "prefer_local_restore",
    "cross_node_restore_target",
    "persist_priority",
    "replicate_to_nodes",
    "evict_hint_nodes",
]

# Decode keys the model outputs in the "decode" section of the two-section JSON
_DECODE_TARGET_KEYS = [
    "spec_decode_enable",
    "spec_decode_depth",
    "spec_mode",
    "spec_v3_block_size",
    "draft_temperature",
    "kv_compression_enable",
    "kv_compression_ratio",
    "decode_batch_priority",
    "early_stop_confidence",
    "cuda_graph_hint",
    "adaptive_depth_override",
]

# Legacy alias kept so external code that imported _TARGET_KEYS keeps working
_TARGET_KEYS = _PREFILL_TARGET_KEYS + _DECODE_TARGET_KEYS

# Mandatory prefill keys: at least one must be present for an example to be valid.
_MANDATORY_PREFILL_KEYS = frozenset({"cache_admission", "throttle_back_pressure"})


def _extract_two_section(decision: Dict[str, Any]):
    """Normalise a stored decision dict to (prefill_dict, decode_dict).

    Handles two storage formats:

    1. **Flat** (from ``UnifiedDecisionVector.to_dict()``):
       All keys live at the top level — e.g. ``{"cache_admission": True, ...}``.

    2. **Two-section** (already structured):
       ``{"prefill": {...}, "decode": {...}}``.

    Returns ``(None, None)`` when the decision is missing mandatory keys so the
    caller can skip the record gracefully instead of producing empty training pairs.
    """
    if not isinstance(decision, dict):
        return None, None

    if "prefill" in decision and "decode" in decision:
        # Two-section format — extract sub-dicts directly.
        prefill_src = decision.get("prefill") or {}
        decode_src  = decision.get("decode")  or {}
        if not isinstance(prefill_src, dict) or not isinstance(decode_src, dict):
            return None, None
        prefill_target = {k: prefill_src[k] for k in _PREFILL_TARGET_KEYS if k in prefill_src}
        decode_target  = {k: decode_src[k]  for k in _DECODE_TARGET_KEYS  if k in decode_src}
    else:
        # Flat format — all keys live at the top level.
        prefill_target = {k: decision[k] for k in _PREFILL_TARGET_KEYS if k in decision}
        decode_target  = {k: decision[k] for k in _DECODE_TARGET_KEYS  if k in decision}

    # Require at least one mandatory prefill key so we don't train on empty completions.
    if not _MANDATORY_PREFILL_KEYS.intersection(prefill_target):
        return None, None

    return prefill_target, decode_target


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Outcome of one LearningLoop training cycle."""
    examples_used:    int           # number of training examples in this cycle
    mean_reward:      float         # mean total_reward of examples used
    loss:             float         # final training loss (0.0 if skipped)
    adapter_path:     Optional[str] # path where the adapter was saved
    elapsed_s:        float         # wall-clock time for the full cycle
    skipped:          bool          # True = not enough positive examples or peft missing
    skip_reason:      str           # human-readable skip reason


# ── LearningLoop ──────────────────────────────────────────────────────────────

class LearningLoop:
    """
    Hourly LoRA fine-tuning loop that improves the reasoning model from its own
    AMF decisions and their observed outcomes.

    Parameters
    ----------
    reasoning_model : ReasoningModel
        The live model (Component 2).  Must be loaded before run_once() is called.
    reward_calculator : RewardCalculator
        The reward store (Component 4).  Supplies training batches.
    interval_s : float
        Seconds between training cycles.  Default 3600 (hourly).
    batch_size : int
        Max decisions to pull per cycle.
    min_positive_examples : int
        Minimum positive-reward examples required to start training.
        If fewer are available, the cycle is skipped.
    lora_r : int
        LoRA rank.
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        LoRA dropout probability.
    learning_rate : float
        AdamW learning rate for the LoRA parameters.
    adapter_save_dir : str | Path
        Directory where adapter checkpoints are saved.
    """

    def __init__(
        self,
        reasoning_model,
        reward_calculator,
        interval_s: float = 3600.0,
        batch_size: int = 64,
        min_positive_examples: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = _DEFAULT_LR,
        adapter_save_dir: str | Path = _DEFAULT_ADAPTER_DIR,
        rl_level: int = 1,
    ) -> None:
        self._model       = reasoning_model
        self._calc        = reward_calculator
        self._interval_s  = max(1.0, float(interval_s))
        self._batch_size  = int(batch_size)
        self._min_pos     = int(min_positive_examples)
        self._lora_r      = int(lora_r)
        self._lora_alpha  = int(lora_alpha)
        self._lora_dropout = float(lora_dropout)
        self._lr          = float(learning_rate)
        self._adapter_dir = Path(adapter_save_dir)
        self._rl_level    = int(rl_level)

        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._swap_lock   = threading.Lock()  # guards hot-swap of model weights
        self._cycle_count = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background fine-tuning daemon thread."""
        if self._thread and self._thread.is_alive():
            log.warning("[loop] already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="amf-learning-loop",
        )
        self._thread.start()
        log.info("[loop] started (interval=%.0fs, min_pos=%d)", self._interval_s, self._min_pos)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        log.info("[loop] stopped after %d cycles", self._cycle_count)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── main API ──────────────────────────────────────────────────────────────

    def run_once(self) -> TrainingResult:
        """
        Execute one training cycle synchronously.

        Called by the background thread every interval_s, and can be triggered
        manually for testing or forced re-training.

        Returns
        -------
        TrainingResult
        """
        t0 = time.perf_counter()

        # ── pull batch from reward store ──────────────────────────────────────
        batch = self._calc.get_training_batch(limit=self._batch_size, min_outcomes=1)
        if not batch:
            return self._skip(t0, "no scored decisions available")

        # ── build supervised training examples (positive reward only) ─────────
        examples = self._build_training_examples(batch)
        if len(examples) < self._min_pos:
            return self._skip(
                t0,
                f"only {len(examples)} positive examples < min={self._min_pos}",
            )

        # ── fine-tune LoRA adapter ────────────────────────────────────────────
        try:
            if self._rl_level >= 4:
                result = self._fine_tune_dpo(examples, t0)
            elif self._rl_level == 3:
                result = self._fine_tune_ppo(examples, t0)
            elif self._rl_level == 2:
                result = self._fine_tune_policy_gradient(examples, t0)
            else:
                result = self._fine_tune(examples, t0)
        except Exception as exc:
            log.warning("[loop] fine-tune failed: %s", exc, exc_info=True)
            return self._skip(t0, f"fine-tune error: {exc}")

        # ── mark consumed so we don't re-train on the same data ──────────────
        decision_ids = [item["decision_id"] for item in batch]
        self._calc.mark_batch_consumed(decision_ids)

        self._cycle_count += 1
        log.info(
            "[loop] cycle %d done: examples=%d mean_reward=%.3f loss=%.4f elapsed=%.1fs",
            self._cycle_count,
            result.examples_used,
            result.mean_reward,
            result.loss,
            result.elapsed_s,
        )
        return result

    # ── training example construction ─────────────────────────────────────────

    def _build_training_examples(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert scored decisions into supervised (prompt, target, weight) triples.

        Only positive-reward decisions are included.  The reward is clamped to
        [0, MAX_REWARD_WEIGHT] and used as a per-example loss multiplier so the
        model learns more strongly from high-value decisions.

        Parameters
        ----------
        batch : list[dict]
            Output from RewardCalculator.get_training_batch().

        Returns
        -------
        list[dict]  with keys: prompt, target, weight, decision_id
        """
        examples = []
        for item in batch:
            reward = float(item["total_reward"])

            if self._rl_level >= 2:
                # Policy gradient: include all scored decisions (positive AND negative)
                # with abs(reward) >= 0.01 so near-zero rewards don't dilute the signal
                if abs(reward) < 0.01:
                    continue
                weight = abs(reward)  # advantage computation handles sign and scaling
            else:
                # Supervised: only learn from positive outcomes
                if reward <= 0.0:
                    continue
                weight = min(reward, _MAX_REWARD_WEIGHT)

            tensor   = item.get("tensor")
            decision = item.get("decision", {})

            # Build the same two-section prompt that ReasoningModel.decide() produces
            prompt = self._tensor_to_prompt(tensor)

            # Normalise decision to two-section format regardless of how it was stored.
            # Old flat-format decisions (from UnifiedDecisionVector.to_dict()) have all
            # keys at the top level.  New two-section decisions already have
            # {"prefill": {...}, "decode": {...}} structure.  If neither mandatory
            # prefill key is present after normalisation, skip gracefully.
            prefill_target, decode_target = _extract_two_section(decision)
            if prefill_target is None:
                # Malformed or unrecognised format — skip without crashing.
                log.debug(
                    "[loop] skipping decision %s: cannot extract two-section target",
                    item.get("decision_id", "?"),
                )
                continue

            target_str = json.dumps(
                {"prefill": prefill_target, "decode": decode_target},
                separators=(",", ":"),
            )

            examples.append(
                {
                    "prompt":      prompt,
                    "target":      target_str,
                    "weight":      weight,
                    "reward":      reward,
                    "decision_id": item["decision_id"],
                }
            )

        return examples

    # ── fine-tuning ───────────────────────────────────────────────────────────

    def _fine_tune(self, examples: List[Dict[str, Any]], t0: float) -> TrainingResult:
        """
        Apply LoRA fine-tuning to the live model and hot-swap the adapter.

        Requires peft and torch.  Raises ImportError if peft is not installed.
        """
        import torch

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning: pip install peft"
            ) from exc

        if not self._model.loaded:
            raise RuntimeError("ReasoningModel is not loaded — call model.load() first")

        tokenizer = self._model._tokenizer
        base_model = self._model._model
        device     = self._model._device

        # ── wrap base model with LoRA ─────────────────────────────────────────
        lora_cfg = LoraConfig(
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            target_modules=_LORA_TARGET_MODULES,
            lora_dropout=self._lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        peft_model.train()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, peft_model.parameters()),
            lr=self._lr,
        )

        # ── tokenize and train ────────────────────────────────────────────────
        total_loss   = 0.0
        total_weight = 0.0

        for ex in examples:
            full_text   = ex["prompt"] + ex["target"]
            prompt_text = ex["prompt"]
            weight      = ex["weight"]

            full_ids   = tokenizer(full_text,   return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

            prompt_len = prompt_ids.shape[1]

            # Labels: -100 for prompt tokens (masked), real ids for target tokens
            labels = full_ids.clone()
            labels[0, :prompt_len] = -100

            optimizer.zero_grad()
            out  = peft_model(input_ids=full_ids, labels=labels)
            loss = out.loss * weight
            loss.backward()
            optimizer.step()

            total_loss   += loss.item()
            total_weight += weight

        mean_loss   = total_loss / max(len(examples), 1)
        mean_reward = sum(ex["weight"] for ex in examples) / len(examples)

        # ── save adapter and hot-swap ─────────────────────────────────────────
        adapter_path = self._save_adapter(peft_model)
        self._hot_swap(peft_model)

        elapsed_s = time.perf_counter() - t0
        return TrainingResult(
            examples_used=len(examples),
            mean_reward=round(mean_reward, 4),
            loss=round(mean_loss, 6),
            adapter_path=str(adapter_path) if adapter_path else None,
            elapsed_s=round(elapsed_s, 2),
            skipped=False,
            skip_reason="",
        )

    def _fine_tune_policy_gradient(
        self, examples: List[Dict[str, Any]], t0: float
    ) -> TrainingResult:
        """
        REINFORCE policy-gradient fine-tuning (rl_level >= 2).

        All scored decisions (positive AND negative) are used.  Advantage is
        computed per-example as (reward − mean_reward).  Examples with
        |advantage| < 0.01 are filtered out — they carry no useful gradient.

        Loss = −log_prob × advantage    (REINFORCE objective)

        Parameters
        ----------
        examples : list[dict]
            Output of _build_training_examples() at rl_level=2.  Each dict must
            contain "prompt", "target", "reward", and "decision_id" keys.
        t0 : float
            perf_counter timestamp from the start of run_once().
        """
        import torch
        import torch.nn.functional as F  # noqa: N812

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning: pip install peft"
            ) from exc

        if not self._model.loaded:
            raise RuntimeError("ReasoningModel is not loaded — call model.load() first")

        tokenizer  = self._model._tokenizer
        base_model = self._model._model
        device     = self._model._device

        # ── compute baseline and filter near-zero advantages ──────────────────
        rewards  = [ex["reward"] for ex in examples]
        baseline = sum(rewards) / len(rewards)

        trainable = []
        for ex in examples:
            advantage = ex["reward"] - baseline
            if abs(advantage) < 0.01:
                continue
            trainable.append({**ex, "advantage": advantage})

        if not trainable:
            raise RuntimeError(
                "all advantages near zero — no useful policy gradient signal"
            )

        # ── wrap base model with LoRA ─────────────────────────────────────────
        lora_cfg = LoraConfig(
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            target_modules=_LORA_TARGET_MODULES,
            lora_dropout=self._lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        peft_model.train()

        _PG_LR = 1e-5  # policy gradient needs a smaller step than supervised 2e-4
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, peft_model.parameters()),
            lr=_PG_LR,
        )

        # ── REINFORCE training loop ───────────────────────────────────────────
        total_loss = 0.0

        for ex in trainable:
            full_text   = ex["prompt"] + ex["target"]
            prompt_text = ex["prompt"]
            advantage   = ex["advantage"]

            full_ids   = tokenizer(full_text,   return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

            prompt_len = prompt_ids.shape[1]

            # Labels: -100 for prompt tokens (masked), real ids for target tokens
            labels = full_ids.clone()
            labels[0, :prompt_len] = -100

            optimizer.zero_grad()
            out = peft_model(input_ids=full_ids, labels=labels)

            # log_prob: sum of log probs for target token positions only
            # shift_logits/labels align t→t+1 predictions with ground truth
            shift_logits = out.logits[0, :-1, :]    # (seq_len-1, vocab_size)
            shift_labels = full_ids[0, 1:]           # (seq_len-1,)
            target_mask  = (labels[0, 1:] != -100).float()

            log_probs       = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            log_prob        = (token_log_probs * target_mask).sum()

            # REINFORCE: maximise advantage-weighted log_prob → minimise −log_p × adv
            loss = -log_prob * advantage
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, peft_model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item()

        mean_loss   = total_loss / max(len(trainable), 1)
        mean_reward = sum(rewards) / len(rewards)

        # ── save adapter and hot-swap ─────────────────────────────────────────
        adapter_path = self._save_adapter(peft_model)
        self._hot_swap(peft_model)

        elapsed_s = time.perf_counter() - t0
        return TrainingResult(
            examples_used=len(trainable),
            mean_reward=round(mean_reward, 4),
            loss=round(mean_loss, 6),
            adapter_path=str(adapter_path) if adapter_path else None,
            elapsed_s=round(elapsed_s, 2),
            skipped=False,
            skip_reason="",
        )

    def _fine_tune_ppo(
        self, examples: List[Dict[str, Any]], t0: float
    ) -> TrainingResult:
        """
        Proximal Policy Optimization fine-tuning (rl_level == 3).

        Adds a ValueHead to the base model, computes GAE advantages from the
        reward store data, then runs 4 epochs of clipped PPO updates.

        Differences from Level 2 REINFORCE:
        - Value network predicts expected future reward, not just baseline mean
        - PPO clip objective (ε=0.2) prevents large policy updates
        - Entropy bonus (0.01) encourages exploration
        - Value loss (0.5 × MSE) trains the value head jointly
        - Gradient clip max_norm=0.5 (tighter than Level 2's 1.0)

        Parameters
        ----------
        examples : list[dict]
            Output of _build_training_examples() at rl_level >= 3.
        t0 : float
            perf_counter timestamp from the start of run_once().
        """
        import torch
        import torch.nn.functional as F  # noqa: N812

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning: pip install peft"
            ) from exc

        from .rollout_buffer import RolloutBuffer, ValueHead

        if not self._model.loaded:
            raise RuntimeError("ReasoningModel is not loaded — call model.load() first")

        tokenizer  = self._model._tokenizer
        base_model = self._model._model
        device     = self._model._device

        # ── attach value head ─────────────────────────────────────────────────
        hidden_size = base_model.config.hidden_size
        value_head  = ValueHead(hidden_size=hidden_size).build().to(device)

        # ── wrap base model with LoRA ─────────────────────────────────────────
        lora_cfg = LoraConfig(
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            target_modules=_LORA_TARGET_MODULES,
            lora_dropout=self._lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        peft_model.train()
        value_head.train()

        _PPO_LR      = 3e-6
        _PPO_EPSILON = 0.2
        _PPO_EPOCHS  = 4
        _ENT_COEF    = 0.01
        _VAL_COEF    = 0.5

        trainable_params = (
            list(filter(lambda p: p.requires_grad, peft_model.parameters()))
            + list(value_head.parameters())
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=_PPO_LR)

        # ── build rollout buffer from training examples ───────────────────────
        buf = RolloutBuffer(capacity=len(examples))
        rewards = [ex["reward"] for ex in examples]

        # Bootstrap value estimates: use normalised reward as proxy V(s)
        # (proper PPO stores values at inference time; this is the training-loop approx)
        reward_mean = sum(rewards) / len(rewards)
        reward_std  = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
        reward_std  = max(reward_std, 1e-8)

        for ex in examples:
            v_approx = (ex["reward"] - reward_mean) / reward_std
            # old log_prob: approximate from reward magnitude (proxy for a frozen ref)
            old_log_prob = -abs(ex["reward"]) * 0.1
            buf.add(
                state=ex["prompt"],
                action=ex["target"],
                reward=ex["reward"],
                value=float(v_approx),
                log_prob=float(old_log_prob),
            )

        buf.compute_gae()

        # ── PPO update epochs ─────────────────────────────────────────────────
        total_loss = 0.0
        update_count = 0

        for _ in range(_PPO_EPOCHS):
            for batch in buf.sample(batch_size=max(4, len(examples) // 4)):
                optimizer.zero_grad()
                batch_loss = 0.0

                for i, (state, action, old_lp, adv, ret) in enumerate(zip(
                    batch["states"],
                    batch["actions"],
                    batch["log_probs"],
                    batch["advantages"],
                    batch["returns"],
                )):
                    full_text   = state + action
                    prompt_text = state

                    full_ids   = tokenizer(full_text,   return_tensors="pt", add_special_tokens=True).input_ids.to(device)
                    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
                    prompt_len = prompt_ids.shape[1]

                    labels = full_ids.clone()
                    labels[0, :prompt_len] = -100

                    out = peft_model(
                        input_ids=full_ids,
                        labels=labels,
                        output_hidden_states=True,
                    )

                    # ── policy loss ───────────────────────────────────────────
                    shift_logits = out.logits[0, :-1, :]
                    shift_labels = full_ids[0, 1:]
                    target_mask  = (labels[0, 1:] != -100).float()

                    log_probs       = F.log_softmax(shift_logits, dim=-1)
                    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                    new_log_prob    = (token_log_probs * target_mask).sum()

                    old_lp_t = torch.tensor(float(old_lp), device=device)
                    adv_t    = torch.tensor(float(adv),    device=device)

                    ratio   = (new_log_prob - old_lp_t).exp()
                    clipped = ratio.clamp(1.0 - _PPO_EPSILON, 1.0 + _PPO_EPSILON)
                    policy_loss = -torch.min(ratio * adv_t, clipped * adv_t)

                    # ── value loss ────────────────────────────────────────────
                    last_hidden = out.hidden_states[-1]
                    value_pred  = value_head(last_hidden)
                    ret_t       = torch.tensor(float(ret), device=device)
                    value_loss  = F.mse_loss(value_pred.squeeze(), ret_t)

                    # ── entropy bonus ─────────────────────────────────────────
                    probs   = F.softmax(shift_logits, dim=-1)
                    entropy = -(probs * log_probs).sum(-1).mean()

                    loss = policy_loss + _VAL_COEF * value_loss - _ENT_COEF * entropy
                    batch_loss += loss

                batch_loss_mean = batch_loss / max(len(batch["states"]), 1)
                batch_loss_mean.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                optimizer.step()

                total_loss   += batch_loss_mean.item()
                update_count += 1

        mean_loss   = total_loss / max(update_count, 1)
        mean_reward = sum(rewards) / len(rewards)

        adapter_path = self._save_adapter(peft_model)
        self._hot_swap(peft_model)

        elapsed_s = time.perf_counter() - t0
        return TrainingResult(
            examples_used=len(examples),
            mean_reward=round(mean_reward, 4),
            loss=round(mean_loss, 6),
            adapter_path=str(adapter_path) if adapter_path else None,
            elapsed_s=round(elapsed_s, 2),
            skipped=False,
            skip_reason="",
        )

    def _fine_tune_dpo(
        self, examples: List[Dict[str, Any]], t0: float
    ) -> TrainingResult:
        """
        Direct Preference Optimization fine-tuning (rl_level >= 4).

        Uses good/bad decision pairs instead of a value head.  Simpler than PPO
        (no rollout buffer, no GAE) but requires paired positive/negative examples
        for the same metrics state.

        DPO loss:
            logits = β × ((log π(good|s) − log π_ref(good|s))
                         − (log π(bad|s)  − log π_ref(bad|s)))
            loss   = −log_sigmoid(logits)

        The reference policy (π_ref) is approximated by the pre-LoRA base model.
        β = 0.1 is the KL constraint strength.

        Parameters
        ----------
        examples : list[dict]
            Output of _build_training_examples() at rl_level >= 4.
        t0 : float
            perf_counter timestamp from the start of run_once().
        """
        import torch
        import torch.nn.functional as F  # noqa: N812

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning: pip install peft"
            ) from exc

        from .rollout_buffer import build_dpo_pairs

        if not self._model.loaded:
            raise RuntimeError("ReasoningModel is not loaded — call model.load() first")

        pairs = build_dpo_pairs(examples, min_reward_gap=0.5)
        if not pairs:
            raise RuntimeError(
                "no DPO preference pairs found — need both positive and negative examples"
            )

        tokenizer  = self._model._tokenizer
        base_model = self._model._model
        device     = self._model._device

        # ── LoRA model (current policy π) ─────────────────────────────────────
        lora_cfg = LoraConfig(
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            target_modules=_LORA_TARGET_MODULES,
            lora_dropout=self._lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        peft_model.train()

        _DPO_LR   = 5e-7
        _DPO_BETA = 0.1

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, peft_model.parameters()),
            lr=_DPO_LR,
        )

        # ── DPO training loop ─────────────────────────────────────────────────
        total_loss = 0.0

        def _log_prob(model, state: str, action: str) -> "torch.Tensor":
            """Compute sum of target-token log probs under `model`."""
            full_text   = state + action
            prompt_text = state
            full_ids   = tokenizer(full_text,   return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            prompt_len = prompt_ids.shape[1]

            labels = full_ids.clone()
            labels[0, :prompt_len] = -100

            out = model(input_ids=full_ids, labels=labels)
            shift_logits = out.logits[0, :-1, :]
            shift_labels = full_ids[0, 1:]
            target_mask  = (labels[0, 1:] != -100).float()

            lps       = F.log_softmax(shift_logits, dim=-1)
            token_lps = lps.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            return (token_lps * target_mask).sum()

        for pair in pairs:
            optimizer.zero_grad()

            # current policy log probs
            good_logp = _log_prob(peft_model, pair.state, pair.good_action)
            bad_logp  = _log_prob(peft_model, pair.state, pair.bad_action)

            # reference policy (base model, no LoRA) — detached
            with torch.no_grad():
                good_ref = _log_prob(base_model, pair.state, pair.good_action)
                bad_ref  = _log_prob(base_model, pair.state, pair.bad_action)

            # DPO loss
            logits = _DPO_BETA * ((good_logp - good_ref) - (bad_logp - bad_ref))
            loss   = -F.logsigmoid(logits)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, peft_model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item()

        mean_loss   = total_loss / max(len(pairs), 1)
        mean_reward = sum(ex["reward"] for ex in examples) / len(examples)

        adapter_path = self._save_adapter(peft_model)
        self._hot_swap(peft_model)

        elapsed_s = time.perf_counter() - t0
        return TrainingResult(
            examples_used=len(pairs),
            mean_reward=round(mean_reward, 4),
            loss=round(mean_loss, 6),
            adapter_path=str(adapter_path) if adapter_path else None,
            elapsed_s=round(elapsed_s, 2),
            skipped=False,
            skip_reason="",
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _run(self) -> None:
        """Background thread body."""
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                log.error("[loop] unexpected error in cycle: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self._interval_s)

    def _skip(self, t0: float, reason: str) -> TrainingResult:
        elapsed_s = time.perf_counter() - t0
        log.debug("[loop] skipped: %s (%.2fs)", reason, elapsed_s)
        return TrainingResult(
            examples_used=0,
            mean_reward=0.0,
            loss=0.0,
            adapter_path=None,
            elapsed_s=round(elapsed_s, 4),
            skipped=True,
            skip_reason=reason,
        )

    @staticmethod
    def _tensor_to_prompt(tensor) -> str:
        """
        Build the two-section prompt that matches ReasoningModel._build_full_prompt().

        Format:
            Prefill metrics: {...slots 0-31...}
            Decode metrics: {...slots 32-53...}
            Decision:

        Slots with |value| < 0.001 are omitted to minimise token count.
        """
        import numpy as np
        try:
            from .reasoning_model import _PREFILL_SLOT_KEYS, _DECODE_SLOT_KEYS
        except ImportError:
            _PREFILL_SLOT_KEYS = {}
            _DECODE_SLOT_KEYS  = {}

        if tensor is None:
            return "Prefill metrics: {}\nDecode metrics: {}\nDecision:"

        arr = np.asarray(tensor, dtype=np.float32)

        prefill: dict = {}
        for idx, key in _PREFILL_SLOT_KEYS.items():
            if idx < len(arr):
                val = float(arr[idx])
                if abs(val) >= 0.001:
                    prefill[key] = round(val, 4)

        decode: dict = {}
        for idx, key in _DECODE_SLOT_KEYS.items():
            if idx < len(arr):
                val = float(arr[idx])
                if abs(val) >= 0.001:
                    decode[key] = round(val, 4)

        prefill_json = json.dumps(prefill, separators=(",", ":"))
        decode_json  = json.dumps(decode,  separators=(",", ":"))
        return f"Prefill metrics: {prefill_json}\nDecode metrics: {decode_json}\nDecision:"

    def _save_adapter(self, peft_model) -> Optional[Path]:
        """Save LoRA adapter weights to a timestamped directory."""
        try:
            self._adapter_dir.mkdir(parents=True, exist_ok=True)
            adapter_path = self._adapter_dir / f"adapter_{int(time.time())}"
            peft_model.save_pretrained(str(adapter_path))
            log.info("[loop] adapter saved to %s", adapter_path)
            return adapter_path
        except Exception as exc:
            log.warning("[loop] adapter save failed: %s", exc)
            return None

    def _hot_swap(self, peft_model) -> None:
        """
        Atomically replace the inner model with the LoRA-trained version.

        CPython's GIL makes the attribute write atomic.  The swap_lock ensures
        the Orchestrator can safely pause requests during the swap if needed.
        """
        with self._swap_lock:
            self._model._model = peft_model
        log.info("[loop] adapter hot-swapped into live model")
