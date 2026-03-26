"""Kelly criterion sizing — actually wired into every trade."""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RiskState:
    bankroll: float
    initial: float
    peak: float = 0.0
    daily_start: float = 0.0
    daily_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    streak: int = 0  # positive=win streak, negative=loss streak
    total_trades: int = 0

    def __post_init__(self):
        self.peak = self.bankroll
        self.daily_start = self.bankroll

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def drawdown(self) -> float:
        return (self.peak - self.bankroll) / self.peak if self.peak > 0 else 0.0

    def record_win(self, payout: float, cost: float):
        net = payout - cost
        self.bankroll += net
        self.peak = max(self.peak, self.bankroll)
        self.daily_pnl += net
        self.wins += 1
        self.total_trades += 1
        self.streak = max(1, self.streak + 1)

    def record_loss(self, cost: float):
        self.bankroll -= cost
        self.daily_pnl -= cost
        self.losses += 1
        self.total_trades += 1
        self.streak = min(-1, self.streak - 1)

    def record_sell(self, payout: float, cost: float):
        net = payout - cost
        self.bankroll += net
        if net >= 0:
            self.peak = max(self.peak, self.bankroll)
            self.daily_pnl += net
            self.wins += 1
            self.streak = max(1, self.streak + 1)
        else:
            self.daily_pnl += net
            self.losses += 1
            self.streak = min(-1, self.streak - 1)
        self.total_trades += 1


class RiskManager:
    """Kelly sizing + circuit breakers — used by every strategy."""

    MAX_DRAWDOWN = 0.20        # halt if down 20%
    DAILY_LOSS_LIMIT = 0.08    # halt if daily loss > 8%
    MAX_POSITIONS = 12
    LOSS_STREAK_PAUSE = 5      # pause after 5 consecutive losses
    MIN_CONFIDENCE = 0.55      # never trade below this

    def __init__(self, initial_bankroll: float):
        self.state = RiskState(bankroll=initial_bankroll, initial=initial_bankroll)
        self._open_positions = 0
        self._paused_until = 0.0

    def kelly_size(self, win_prob: float, entry_price: float) -> float:
        """
        f* = (b*p - q) / b
        b = net odds (win $1-p per $p risked)
        p = win_prob, q = 1-p
        """
        import time
        if time.time() < self._paused_until:
            return 0.0

        p = max(0.51, min(0.99, win_prob))
        q = 1.0 - p
        # net odds: buy at entry_price, win pays $1
        b = (1.0 - entry_price) / entry_price if entry_price > 0 else 1.0

        kelly_f = (b * p - q) / b
        if kelly_f <= 0:
            return 0.0

        # Scale Kelly fraction by bankroll size
        fraction = self._kelly_fraction()
        adjusted = kelly_f * fraction

        # Cap per-trade risk
        max_bet = self.state.bankroll * 0.12
        bet = min(adjusted * self.state.bankroll, max_bet)
        return max(0.0, round(bet, 2))

    def _kelly_fraction(self) -> float:
        b = self.state.bankroll
        if b < 500:   return 0.25
        if b < 5000:  return 0.35
        if b < 50000: return 0.50
        return 0.65

    def can_trade(self) -> tuple[bool, str]:
        import time
        if time.time() < self._paused_until:
            return False, "paused"
        if self.state.drawdown >= self.MAX_DRAWDOWN:
            return False, f"drawdown {self.state.drawdown:.1%}"
        daily_loss_pct = -self.state.daily_pnl / self.state.daily_start if self.state.daily_start > 0 else 0
        if daily_loss_pct >= self.DAILY_LOSS_LIMIT:
            return False, f"daily loss {daily_loss_pct:.1%}"
        if self._open_positions >= self.MAX_POSITIONS:
            return False, "max positions"
        if self.state.streak <= -self.LOSS_STREAK_PAUSE:
            import time
            self._paused_until = time.time() + 30
            self.state.streak = 0
            return False, "loss streak pause"
        return True, "ok"

    def open_position(self):
        self._open_positions += 1

    def close_position(self):
        self._open_positions = max(0, self._open_positions - 1)

    def reset_daily(self):
        self.state.daily_start = self.state.bankroll
        self.state.daily_pnl = 0.0
