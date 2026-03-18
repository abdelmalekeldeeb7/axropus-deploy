"""
Level 3 RL Support: RolloutBuffer, ValueHead, GAE
==================================================
Provides the data structures and neural network components needed for
Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO).

RolloutBuffer
-------------
Stores (state, action, reward, value, log_prob) tuples collected during
inference, then computes Generalized Advantage Estimation (GAE) for PPO updates.

ValueHead
---------
A small MLP attached to the last hidden state of the reasoning model.
Predicts expected cumulative reward (value function) for a given metrics state.

DPOPair
-------
A preference pair (state, good_decision, bad_decision) used by DPO training.
The RewardCalculator naturally produces these: positive-reward decisions are
"good", negative-reward decisions are "bad".

Usage
-----
    buf = RolloutBuffer(capacity=2048)
    buf.add(state=tensor, action=token_ids, reward=1.0, value=0.8, log_prob=-0.4)
    ...
    buf.compute_gae()
    for batch in buf.sample(batch_size=16):
        # run PPO update
        ...
    buf.clear()
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


# ── DPO pair ──────────────────────────────────────────────────────────────────

@dataclass
class DPOPair:
    """
    A single preference pair for Direct Preference Optimization.

    Attributes
    ----------
    state : any
        The metrics tensor (64-element float array) at decision time.
    good_action : str
        JSON decision string that led to positive reward.
    bad_action : str
        JSON decision string that led to negative reward.
    good_reward : float
        Total reward of the good decision.
    bad_reward : float
        Total reward of the bad decision.
    """
    state: object
    good_action: str
    bad_action: str
    good_reward: float = 0.0
    bad_reward: float  = 0.0


# ── RolloutBuffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Fixed-capacity buffer for storing PPO rollout data.

    Each entry records:
    - state      : metrics tensor (observation)
    - action     : tokenized decision (action)
    - reward     : observed scalar reward from RewardCalculator
    - value      : value-head estimate V(s) at decision time
    - log_prob   : log π(a|s) under the policy that generated the action
    - advantage  : computed by compute_gae() after collection
    - ret        : discounted return, computed by compute_gae()
    """

    def __init__(self, capacity: int = 2048) -> None:
        self._capacity = int(capacity)
        self.states:     List = []
        self.actions:    List = []
        self.rewards:    List[float] = []
        self.values:     List[float] = []
        self.log_probs:  List[float] = []
        self.advantages: List[float] = []
        self.returns:    List[float] = []

    # ── collection API ────────────────────────────────────────────────────────

    def add(
        self,
        *,
        state,
        action,
        reward: float,
        value: float,
        log_prob: float,
    ) -> None:
        """Append one (s, a, r, v, log_p) tuple. Drops oldest if over capacity."""
        if len(self.states) >= self._capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        """Reset buffer after a PPO update epoch."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()

    # ── GAE ───────────────────────────────────────────────────────────────────

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        """
        Generalized Advantage Estimation (Schulman et al. 2015).

        advantage[t] = δ[t] + (γλ)δ[t+1] + (γλ)²δ[t+2] + ...
        where δ[t] = r[t] + γV[t+1] − V[t]

        Also computes returns[t] = advantage[t] + V[t] for the value loss.

        Must be called before sample().
        """
        n = len(self.rewards)
        if n == 0:
            return

        advantages = [0.0] * n
        last_adv   = 0.0

        for t in reversed(range(n)):
            next_value = self.values[t + 1] if t < n - 1 else 0.0
            delta      = self.rewards[t] + gamma * next_value - self.values[t]
            last_adv   = delta + gamma * lam * last_adv
            advantages[t] = last_adv

        self.advantages = advantages
        self.returns    = [a + v for a, v in zip(advantages, self.values)]

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample(self, batch_size: int = 16) -> Iterator[dict]:
        """
        Yield random mini-batches for PPO update epochs.

        Each yielded dict contains keys:
            states, actions, rewards, values, log_probs, advantages, returns
        as lists of length ≤ batch_size.
        """
        if not self.advantages:
            raise RuntimeError("call compute_gae() before sample()")

        n       = len(self.states)
        indices = list(range(n))
        random.shuffle(indices)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "states":     [self.states[i]    for i in idx],
                "actions":    [self.actions[i]   for i in idx],
                "rewards":    [self.rewards[i]   for i in idx],
                "values":     [self.values[i]    for i in idx],
                "log_probs":  [self.log_probs[i] for i in idx],
                "advantages": [self.advantages[i] for i in idx],
                "returns":    [self.returns[i]   for i in idx],
            }

    # ── statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics for logging."""
        if not self.rewards:
            return {"n": 0, "mean_reward": 0.0, "mean_advantage": 0.0}
        n = len(self.rewards)
        mean_r = sum(self.rewards) / n
        mean_a = (sum(self.advantages) / n) if self.advantages else 0.0
        return {"n": n, "mean_reward": round(mean_r, 4), "mean_advantage": round(mean_a, 4)}


# ── ValueHead ─────────────────────────────────────────────────────────────────

class ValueHead:
    """
    Two-layer MLP value function head.

    Predicts a scalar V(s) from the last-token hidden state of the reasoning
    model.  Attached to the model at PPO training time; not used during normal
    inference (so inference latency is unchanged).

    Parameters
    ----------
    hidden_size : int
        Hidden state dimension of the base LM (e.g. 4096 for Llama 8B).

    Usage
    -----
        import torch
        vh = ValueHead(hidden_size=4096)
        module = vh.build()           # returns torch.nn.Module
        values = module(hidden_states)  # (batch,) tensor of value estimates
    """

    def __init__(self, hidden_size: int = 4096) -> None:
        self._hidden_size = int(hidden_size)

    def build(self):
        """
        Construct and return the torch.nn.Module.

        Separated from __init__ so the class can be instantiated without
        importing torch (e.g. in test environments without GPU).
        """
        import torch
        import torch.nn as nn

        hidden = self._hidden_size

        class _ValueHeadModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1   = nn.Linear(hidden, 256)
                self.linear2   = nn.Linear(256, 1)
                self.activation = nn.GELU()

            def forward(self, hidden_states):
                # hidden_states: (batch, seq_len, hidden_size)
                x = hidden_states[:, -1, :]          # last token
                x = self.activation(self.linear1(x))
                return self.linear2(x).squeeze(-1)   # (batch,)

        return _ValueHeadModule()


# ── DPO pair builder ──────────────────────────────────────────────────────────

def build_dpo_pairs(examples: list, min_reward_gap: float = 0.5) -> List[DPOPair]:
    """
    Build DPO preference pairs from a training batch.

    Strategy: for each "good" example (reward > 0), pair it with the "bad"
    example (reward < 0) whose tensor is most similar.  If no exact same-state
    pair exists, pair across all available good/bad examples.

    Parameters
    ----------
    examples : list[dict]
        Dicts with keys: prompt, target, reward, decision_id.
    min_reward_gap : float
        Minimum |good_reward − bad_reward| for a pair to be used.

    Returns
    -------
    list[DPOPair]
    """
    good = [e for e in examples if e.get("reward", 0.0) > 0.0]
    bad  = [e for e in examples if e.get("reward", 0.0) < 0.0]

    if not good or not bad:
        return []

    pairs: List[DPOPair] = []
    # Round-robin pairing — simple but effective for the reward store structure
    for i, g in enumerate(good):
        b = bad[i % len(bad)]
        gap = abs(g["reward"] - b["reward"])
        if gap < min_reward_gap:
            continue
        pairs.append(DPOPair(
            state       = g.get("prompt", ""),   # use prompt as state proxy
            good_action = g["target"],
            bad_action  = b["target"],
            good_reward = float(g["reward"]),
            bad_reward  = float(b["reward"]),
        ))

    return pairs
