from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Protocol

class MetricsView(Protocol):
    watts_avg: float
    tps_rolling: float
    tokens_per_joule: float


class PowerBand(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class EfficiencyBand(str, Enum):
    UNKNOWN = "unknown"
    OK = "ok"
    BAD = "bad"


@dataclass(frozen=True)
class PolicyDecision:
    target_tps: float  # 0 => unthrottled
    power_band: PowerBand
    efficiency_band: EfficiencyBand
    action: str  # "up" | "down" | "hold"
    reason: str


@dataclass
class PolicyConfig:
    # Power hysteresis:
    power_high_enter_w: float = 110.0
    power_high_exit_w: float = 106.0
    power_low_enter_w: float = 90.0
    power_low_exit_w: float = 94.0

    # Efficiency control:
    eff_ema_alpha: float = 0.20
    eff_drop_enter_ratio: float = 0.92
    eff_drop_exit_ratio: float = 0.97
    eff_drop_streak: int = 2  # consecutive samples required to enter BAD

    # Rate adaptation:
    min_target_tps: float = 1.0
    step_up: float = 1.15
    step_down: float = 0.85
    adjust_cooldown_s: float = 2.0

    # If we need an initial clamp when transitioning from "unthrottled" to "throttled"
    # but haven't measured TPS yet.
    fallback_tps: float = 50.0


class AdaptiveRatePolicy:
    """
    Rule-based control policy operating only on metrics (no llama internals).

    Outputs a target token generation rate (tokens/sec). The controller enforces
    it via a rate limiter.
    """

    def __init__(self, cfg: PolicyConfig | None = None) -> None:
        self._cfg = cfg or PolicyConfig()
        self._power_band: PowerBand = PowerBand.NORMAL
        self._eff_band: EfficiencyBand = EfficiencyBand.UNKNOWN
        self._eff_ema: float | None = None
        self._eff_bad_streak: int = 0
        self._target_tps: float = 0.0  # 0 => unthrottled
        self._last_adjust_s: float = 0.0

    @property
    def target_tps(self) -> float:
        return self._target_tps

    def update(self, m: MetricsView, *, now_s: float | None = None) -> PolicyDecision:
        now = time.perf_counter() if now_s is None else float(now_s)

        self._update_power_band(float(getattr(m, "watts_avg", 0.0)))
        self._update_efficiency_band(float(getattr(m, "tokens_per_joule", 0.0)))

        action = "hold"
        reason = "stable"

        if self._power_band == PowerBand.HIGH:
            action = "down"
            reason = "power_high"
        elif self._eff_band == EfficiencyBand.BAD:
            action = "down"
            reason = "eff_drop"
        elif self._power_band == PowerBand.LOW:
            action = "up"
            reason = "power_low"

        # Cooldown to avoid oscillation. "power_high" is treated as safety-critical
        # and bypasses cooldown so we can respond quickly to cap violations.
        if action == "up" and (now - self._last_adjust_s) < self._cfg.adjust_cooldown_s:
            action = "hold"
            reason = "cooldown"
        elif action == "down" and reason != "power_high" and (now - self._last_adjust_s) < self._cfg.adjust_cooldown_s:
            action = "hold"
            reason = "cooldown"

        if action == "down":
            if self._apply_down(float(getattr(m, "tps_rolling", 0.0))):
                self._last_adjust_s = now
        elif action == "up":
            if self._apply_up():
                self._last_adjust_s = now

        return PolicyDecision(
            target_tps=self._target_tps,
            power_band=self._power_band,
            efficiency_band=self._eff_band,
            action=action,
            reason=reason,
        )

    def _update_power_band(self, watts: float) -> None:
        c = self._cfg
        if self._power_band == PowerBand.HIGH:
            if watts < c.power_high_exit_w:
                self._power_band = PowerBand.NORMAL
            return

        if self._power_band == PowerBand.LOW:
            if watts > c.power_low_exit_w:
                self._power_band = PowerBand.NORMAL
            return

        # NORMAL
        if watts > c.power_high_enter_w:
            self._power_band = PowerBand.HIGH
        elif watts < c.power_low_enter_w:
            self._power_band = PowerBand.LOW

    def _update_efficiency_band(self, tokens_per_joule: float) -> None:
        c = self._cfg
        if tokens_per_joule <= 0.0:
            return

        if self._eff_ema is None or self._eff_ema <= 0.0:
            self._eff_ema = tokens_per_joule
            self._eff_band = EfficiencyBand.OK
            self._eff_bad_streak = 0
            return

        ratio = tokens_per_joule / self._eff_ema

        if self._eff_band == EfficiencyBand.BAD:
            if ratio >= c.eff_drop_exit_ratio:
                self._eff_band = EfficiencyBand.OK
            # Keep EMA updating so "normal" can drift with long runs.
            self._eff_ema = (1.0 - c.eff_ema_alpha) * self._eff_ema + c.eff_ema_alpha * tokens_per_joule
            return

        # UNKNOWN/OK -> possibly enter BAD after sustained drop.
        if ratio <= c.eff_drop_enter_ratio:
            self._eff_bad_streak += 1
        else:
            self._eff_bad_streak = 0

        if self._eff_bad_streak >= c.eff_drop_streak:
            self._eff_band = EfficiencyBand.BAD
            self._eff_bad_streak = 0
        else:
            self._eff_band = EfficiencyBand.OK

        self._eff_ema = (1.0 - c.eff_ema_alpha) * self._eff_ema + c.eff_ema_alpha * tokens_per_joule

    def _apply_down(self, observed_tps: float) -> bool:
        c = self._cfg
        if self._target_tps <= 0.0:
            base = observed_tps if observed_tps > 0.0 else c.fallback_tps
            new = max(c.min_target_tps, base * c.step_down)
            changed = new != self._target_tps
            self._target_tps = new
            return changed

        new = max(c.min_target_tps, self._target_tps * c.step_down)
        changed = new != self._target_tps
        self._target_tps = new
        return changed

    def _apply_up(self) -> bool:
        c = self._cfg
        if self._target_tps <= 0.0:
            # Already unthrottled; no action required.
            return False
        new = max(c.min_target_tps, self._target_tps * c.step_up)
        changed = new != self._target_tps
        self._target_tps = new
        return changed
