from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Deque, Protocol
from collections import deque


class MetricsView(Protocol):
    watts_avg: float
    tps_rolling: float
    tokens_per_joule: float


class Objective(str, Enum):
    SPEED = "speed"
    EFFICIENCY = "efficiency"
    STABILITY = "stability"
    BALANCED = "balanced"


@dataclass(frozen=True)
class AgentObservation:
    t_s: float
    watts_avg: float
    tps: float
    tokens_per_joule: float


@dataclass(frozen=True)
class AgentAction:
    target_tps: float  # 0 => unthrottled
    objective: Objective
    action: str  # "up" | "down" | "hold"
    reason: str


@dataclass(frozen=True)
class RewardSignal:
    total: float
    speed_term: float
    efficiency_term: float
    stability_penalty: float
    power_penalty: float


@dataclass(frozen=True)
class AgentStep:
    obs: AgentObservation
    action: AgentAction
    reward: RewardSignal
    eff_ema: float
    eff_ratio: float


@dataclass
class ObjectiveWeights:
    w_speed: float
    w_efficiency: float
    w_stability: float
    w_power: float


@dataclass
class AgentConfig:
    # Power control bands (with hysteresis).
    power_cap_w: float = 110.0
    power_high_enter_w: float = 110.0
    power_high_exit_w: float = 106.0
    power_low_enter_w: float = 90.0
    power_low_exit_w: float = 94.0

    # Long-term efficiency trend (EMA time constant).
    eff_trend_tau_s: float = 60.0
    eff_drop_enter_ratio: float = 0.92
    eff_drop_exit_ratio: float = 0.97
    eff_drop_streak: int = 10  # consecutive update ticks

    # Objective switching stability.
    objective_dwell_s: float = 5.0

    # Rate adaptation.
    min_target_tps: float = 1.0
    fallback_tps: float = 50.0
    adjust_cooldown_s: float = 1.0

    # Multiplicative adjustments per objective.
    step_up_speed: float = 1.25
    step_down_speed: float = 0.90
    step_up_efficiency: float = 1.10
    step_down_efficiency: float = 0.85
    step_up_stability: float = 1.05
    step_down_stability: float = 0.75
    step_up_balanced: float = 1.15
    step_down_balanced: float = 0.85

    # Reward weights per objective.
    weights_speed: ObjectiveWeights = field(
        default_factory=lambda: ObjectiveWeights(w_speed=1.0, w_efficiency=0.2, w_stability=0.2, w_power=1.5)
    )
    weights_efficiency: ObjectiveWeights = field(
        default_factory=lambda: ObjectiveWeights(w_speed=0.3, w_efficiency=1.0, w_stability=0.3, w_power=1.5)
    )
    weights_stability: ObjectiveWeights = field(
        default_factory=lambda: ObjectiveWeights(w_speed=0.2, w_efficiency=0.5, w_stability=1.0, w_power=2.0)
    )
    weights_balanced: ObjectiveWeights = field(
        default_factory=lambda: ObjectiveWeights(w_speed=0.7, w_efficiency=0.7, w_stability=0.5, w_power=1.5)
    )


class _Band(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class _EffBand(str, Enum):
    UNKNOWN = "unknown"
    OK = "ok"
    BAD = "bad"


class KorithAgent:
    """
    Goal-oriented inference control agent.

    This is *not* RL; it produces a simple reward signal and keeps a transition
    history so it can evolve into an RL agent later without changing interfaces.

    The only control output is `target_tps` (tokens/sec), enforced by the runtime.
    """

    def __init__(self, cfg: AgentConfig | None = None, *, history_len: int = 2048) -> None:
        self._cfg = cfg or AgentConfig()

        self._objective: Objective = Objective.BALANCED
        self._objective_since_s: float = 0.0

        self._power_band: _Band = _Band.NORMAL
        self._eff_band: _EffBand = _EffBand.UNKNOWN

        self._eff_ema: float | None = None
        self._eff_bad_streak: int = 0

        self._target_tps: float = 0.0
        self._last_adjust_s: float = 0.0

        self._last_obs: AgentObservation | None = None
        self._last_action: AgentAction | None = None

        self._history: Deque[AgentStep] = deque(maxlen=history_len)

    @property
    def target_tps(self) -> float:
        return self._target_tps

    @property
    def objective(self) -> Objective:
        return self._objective

    @property
    def history(self) -> Deque[AgentStep]:
        return self._history

    def step(self, m: MetricsView, *, now_s: float | None = None) -> AgentStep:
        now = time.perf_counter() if now_s is None else float(now_s)
        obs = AgentObservation(
            t_s=now,
            watts_avg=float(getattr(m, "watts_avg", 0.0)),
            tps=float(getattr(m, "tps_rolling", 0.0)),
            tokens_per_joule=float(getattr(m, "tokens_per_joule", 0.0)),
        )

        self._update_power_band(obs.watts_avg)
        self._update_efficiency_trend(obs, now)
        self._select_objective(obs, now)

        action, reason = self._select_action(obs)
        changed = self._apply_action(action, obs, now)

        if changed:
            self._last_adjust_s = now

        act = AgentAction(
            target_tps=self._target_tps,
            objective=self._objective,
            action=action,
            reason=reason,
        )

        reward = self._compute_reward(obs, act)
        step = AgentStep(
            obs=obs,
            action=act,
            reward=reward,
            eff_ema=float(self._eff_ema) if self._eff_ema is not None else 0.0,
            eff_ratio=self._eff_ratio(obs),
        )

        self._history.append(step)
        self._last_obs = obs
        self._last_action = act
        return step

    def _update_power_band(self, watts: float) -> None:
        c = self._cfg
        if self._power_band == _Band.HIGH:
            if watts < c.power_high_exit_w:
                self._power_band = _Band.NORMAL
            return
        if self._power_band == _Band.LOW:
            if watts > c.power_low_exit_w:
                self._power_band = _Band.NORMAL
            return

        if watts > c.power_high_enter_w:
            self._power_band = _Band.HIGH
        elif watts < c.power_low_enter_w:
            self._power_band = _Band.LOW

    def _update_efficiency_trend(self, obs: AgentObservation, now_s: float) -> None:
        c = self._cfg
        tpj = obs.tokens_per_joule
        if tpj <= 0.0:
            return

        if self._eff_ema is None or self._last_obs is None:
            self._eff_ema = tpj
            self._eff_band = _EffBand.OK
            self._eff_bad_streak = 0
            return

        dt = max(0.0, now_s - self._last_obs.t_s)
        tau = max(1e-3, c.eff_trend_tau_s)
        alpha = 1.0 - math.exp(-dt / tau)
        self._eff_ema = (1.0 - alpha) * self._eff_ema + alpha * tpj

        ratio = self._eff_ratio(obs)
        if self._eff_band == _EffBand.BAD:
            if ratio >= c.eff_drop_exit_ratio:
                self._eff_band = _EffBand.OK
            return

        if ratio <= c.eff_drop_enter_ratio:
            self._eff_bad_streak += 1
        else:
            self._eff_bad_streak = 0

        if self._eff_bad_streak >= c.eff_drop_streak:
            self._eff_band = _EffBand.BAD
            self._eff_bad_streak = 0
        else:
            self._eff_band = _EffBand.OK

    def _eff_ratio(self, obs: AgentObservation) -> float:
        if self._eff_ema is None or self._eff_ema <= 0.0:
            return 1.0
        if obs.tokens_per_joule <= 0.0:
            return 1.0
        return obs.tokens_per_joule / self._eff_ema

    def _select_objective(self, obs: AgentObservation, now_s: float) -> None:
        c = self._cfg

        # Safety-critical: if power is in the HIGH band, prioritize stability immediately.
        if self._power_band == _Band.HIGH:
            if self._objective != Objective.STABILITY:
                self._objective = Objective.STABILITY
                self._objective_since_s = now_s
            return

        desired: Objective
        if self._eff_band == _EffBand.BAD:
            desired = Objective.EFFICIENCY
        elif self._power_band == _Band.LOW:
            desired = Objective.SPEED
        else:
            desired = Objective.BALANCED

        if desired == self._objective:
            return

        if (now_s - self._objective_since_s) < c.objective_dwell_s:
            return

        self._objective = desired
        self._objective_since_s = now_s

    def _select_action(self, obs: AgentObservation) -> tuple[str, str]:
        c = self._cfg

        # Immediate downshift if we exceed the soft cap (even if hysteresis hasn't latched).
        if obs.watts_avg > c.power_cap_w:
            return "down", "power_cap"

        if self._power_band == _Band.HIGH:
            return "down", "power_high"
        if self._eff_band == _EffBand.BAD:
            return "down", "eff_drop"
        if self._power_band == _Band.LOW:
            return "up", "power_low"

        return "hold", "stable"

    def _steps(self) -> tuple[float, float]:
        c = self._cfg
        if self._objective == Objective.SPEED:
            return c.step_up_speed, c.step_down_speed
        if self._objective == Objective.EFFICIENCY:
            return c.step_up_efficiency, c.step_down_efficiency
        if self._objective == Objective.STABILITY:
            return c.step_up_stability, c.step_down_stability
        return c.step_up_balanced, c.step_down_balanced

    def _apply_action(self, action: str, obs: AgentObservation, now_s: float) -> bool:
        c = self._cfg
        if action == "hold":
            return False

        if (now_s - self._last_adjust_s) < c.adjust_cooldown_s:
            return False

        step_up, step_down = self._steps()
        prev = self._target_tps

        if action == "down":
            if self._target_tps <= 0.0:
                base = obs.tps if obs.tps > 0.0 else c.fallback_tps
                self._target_tps = max(c.min_target_tps, base * step_down)
            else:
                self._target_tps = max(c.min_target_tps, self._target_tps * step_down)
        elif action == "up":
            if self._target_tps > 0.0:
                self._target_tps = max(c.min_target_tps, self._target_tps * step_up)
            else:
                # Unthrottled stays unthrottled.
                self._target_tps = 0.0

        return self._target_tps != prev

    def _weights(self) -> ObjectiveWeights:
        c = self._cfg
        if self._objective == Objective.SPEED:
            return c.weights_speed
        if self._objective == Objective.EFFICIENCY:
            return c.weights_efficiency
        if self._objective == Objective.STABILITY:
            return c.weights_stability
        return c.weights_balanced

    def _compute_reward(self, obs: AgentObservation, act: AgentAction) -> RewardSignal:
        w = self._weights()
        c = self._cfg

        # Positive terms (log-compressed to avoid dominance by large values).
        speed_term = math.log1p(max(0.0, obs.tps))
        efficiency_term = math.log1p(max(0.0, obs.tokens_per_joule))

        # Penalties.
        prev_target = self._last_action.target_tps if self._last_action is not None else act.target_tps
        delta = abs(act.target_tps - prev_target)
        stability_penalty = delta / max(1.0, prev_target if prev_target > 0.0 else 1.0)

        power_penalty = max(0.0, obs.watts_avg - c.power_cap_w) / max(1e-6, c.power_cap_w)

        total = (
            w.w_speed * speed_term
            + w.w_efficiency * efficiency_term
            - w.w_stability * stability_penalty
            - w.w_power * power_penalty
        )

        return RewardSignal(
            total=total,
            speed_term=speed_term,
            efficiency_term=efficiency_term,
            stability_penalty=stability_penalty,
            power_penalty=power_penalty,
        )
