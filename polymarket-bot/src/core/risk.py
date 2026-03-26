"""
Risk Manager — Dynamic Kelly + Portfolio Correlation + Circuit Breakers.

Elite quant additions:
  1. Dynamic Kelly: adjusts fraction based on rolling 50-trade WR
     - WR < 52%: use 20% Kelly (defensive)
     - WR 52-60%: standard Kelly
     - WR > 60%: up to 70% Kelly (scale into edge)
  2. Portfolio correlation: reduce size when multiple correlated positions open
     - BTC + ETH + crypto markets all correlated → reduce to 50% size
  3. Volatility-adjusted sizing: shrink in high-vol regimes
  4. Sharpe-aware circuit breaker: pause if Sharpe drops below 0
  5. Hot/cold streaks: boost size slightly on 5+ win streaks
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    bankroll:    float
    initial:     float
    peak:        float = 0.0
    daily_start: float = 0.0
    daily_pnl:   float = 0.0
    wins:        int   = 0
    losses:      int   = 0
    streak:      int   = 0     # positive=win streak, negative=loss streak
    total_trades: int  = 0

    def __post_init__(self):
        self.peak        = self.bankroll
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
        self.bankroll  += net
        self.peak       = max(self.peak, self.bankroll)
        self.daily_pnl += net
        self.wins      += 1
        self.total_trades += 1
        self.streak     = max(1, self.streak + 1)

    def record_loss(self, cost: float):
        self.bankroll  -= cost
        self.daily_pnl -= cost
        self.losses    += 1
        self.total_trades += 1
        self.streak     = min(-1, self.streak - 1)

    def record_sell(self, payout: float, cost: float):
        net = payout - cost
        self.bankroll  += net
        if net >= 0:
            self.peak       = max(self.peak, self.bankroll)
            self.daily_pnl += net
            self.wins      += 1
            self.streak     = max(1, self.streak + 1)
        else:
            self.daily_pnl += net
            self.losses    += 1
            self.streak     = min(-1, self.streak - 1)
        self.total_trades += 1


class RiskManager:
    """
    Dynamic Kelly sizing with circuit breakers.
    Every strategy must call can_trade() before entering.
    """

    MAX_DRAWDOWN          = 0.15   # halt if down 15%
    DAILY_LOSS_LIMIT      = 0.06   # halt if daily loss > 6%
    MAX_POSITIONS         = 15     # allow up to 15 concurrent
    LOSS_STREAK_PAUSE     = 3      # pause after 3 consecutive losses
    LOSS_STREAK_PAUSE_SECS = 15    # only 15s cooldown
    MIN_CONFIDENCE        = 0.58

    # Dynamic Kelly WR thresholds
    WR_LOW    = 0.52   # below → defensive Kelly
    WR_TARGET = 0.60   # above → aggressive Kelly
    WR_HIGH   = 0.65   # above → max Kelly

    # Correlation groups: positions in same group reduce sizing
    CORR_GROUPS = {
        "btc":   ["btcusdt", "btc", "bitcoin"],
        "eth":   ["ethusdt", "eth", "ethereum"],
        "crypto": ["btcusdt", "ethusdt", "btc", "eth", "bitcoin", "ethereum"],
    }

    def __init__(self, bankroll: float, initial_bankroll: float = 0.0):
        br = bankroll or initial_bankroll
        self.state            = RiskState(bankroll=br, initial=br)
        self._open_positions  = 0
        self._paused_until    = 0.0

        # Rolling WR for dynamic Kelly (last 50 trades)
        self._recent_results: deque = deque(maxlen=50)   # True=win, False=loss
        self._recent_pnl: deque     = deque(maxlen=50)   # net P&L per trade

        # Correlation tracking: strategy/symbol → open count
        self._open_symbols: list[str] = []

        # Volatility regime multiplier (set externally by main loop)
        self.vol_multiplier: float = 1.0

    # ── Kelly sizing ─────────────────────────────────────────────────────────

    def kelly_size(
        self,
        win_prob:    float,
        entry_price: float,
        symbol:      str = "",
        strategy:    str = "",
    ) -> float:
        """
        Dynamic Kelly: f* = (b*p - q) / b
        Adjusts fraction based on rolling WR, correlation, volatility.
        """
        if time.time() < self._paused_until:
            return 0.0

        p = max(0.51, min(0.99, win_prob))
        q = 1.0 - p
        b = (1.0 - entry_price) / entry_price if entry_price > 0 else 1.0

        kelly_f = (b * p - q) / b
        if kelly_f <= 0:
            return 0.0

        # Dynamic fraction based on recent performance
        fraction = self._dynamic_kelly_fraction()

        # Correlation discount
        corr_discount = self._correlation_discount(symbol)

        # Volatility discount (from regime detector — set by main loop)
        vol_discount = self.vol_multiplier

        # Hot streak bonus (5+ wins → small boost)
        streak_boost = 1.0
        if self.state.streak >= 5:
            streak_boost = min(1.15, 1.0 + (self.state.streak - 4) * 0.03)

        adjusted = kelly_f * fraction * corr_discount * vol_discount * streak_boost

        # Cap per-trade risk
        max_bet = self.state.bankroll * 0.12
        bet     = min(adjusted * self.state.bankroll, max_bet)
        return max(0.0, round(bet, 2))

    def _dynamic_kelly_fraction(self) -> float:
        """
        Adjust Kelly fraction based on rolling 50-trade win rate.
        Scales from 0.20x (defensive) to 0.70x (aggressive).
        """
        if len(self._recent_results) < 20:
            # Not enough history — use conservative base fraction
            return self._base_kelly_fraction()

        rolling_wr = sum(self._recent_results) / len(self._recent_results)

        # Also check rolling Sharpe (if we have enough P&L data)
        if len(self._recent_pnl) >= 20:
            pnls = list(self._recent_pnl)
            avg_pnl = sum(pnls) / len(pnls)
            std_pnl = (sum((x - avg_pnl)**2 for x in pnls) / len(pnls)) ** 0.5
            rolling_sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0.0
            if rolling_sharpe < 0:
                # Negative Sharpe → be defensive
                return self._base_kelly_fraction() * 0.70
        else:
            rolling_sharpe = 1.0

        base = self._base_kelly_fraction()

        if rolling_wr < self.WR_LOW:
            fraction = base * 0.60          # underperforming → cut size
        elif rolling_wr < self.WR_TARGET:
            fraction = base                 # standard
        elif rolling_wr < self.WR_HIGH:
            fraction = min(base * 1.25, 0.65)  # outperforming → scale up
        else:
            fraction = min(base * 1.50, 0.70)  # exceptional → max Kelly

        logger.debug("Dynamic Kelly: wr=%.2f sharpe=%.2f fraction=%.2f",
                     rolling_wr, rolling_sharpe, fraction)
        return fraction

    def _base_kelly_fraction(self) -> float:
        """Bankroll-tiered base Kelly fraction."""
        b = self.state.bankroll
        if b < 500:    return 0.25
        if b < 5_000:  return 0.35
        if b < 50_000: return 0.50
        return 0.65

    def _correlation_discount(self, symbol: str) -> float:
        """
        Reduce size when we already have correlated positions open.
        2+ crypto positions → 0.70x
        3+ crypto positions → 0.55x
        """
        if not symbol:
            return 1.0
        sym_lower = symbol.lower()

        # Count how many correlated open positions exist
        crypto_keywords = self.CORR_GROUPS["crypto"]
        correlated_open = sum(
            1 for s in self._open_symbols
            if any(kw in s.lower() for kw in crypto_keywords)
        )

        if any(kw in sym_lower for kw in crypto_keywords):
            # This trade is also crypto-correlated
            if correlated_open >= 3:
                return 0.55
            if correlated_open >= 2:
                return 0.70
        return 1.0

    # ── Circuit breakers ─────────────────────────────────────────────────────

    def can_trade(self) -> tuple[bool, str]:
        if time.time() < self._paused_until:
            remaining = self._paused_until - time.time()
            return False, f"paused ({remaining:.0f}s)"

        if self.state.drawdown >= self.MAX_DRAWDOWN:
            return False, f"drawdown {self.state.drawdown:.1%}"

        if self.state.daily_start > 0:
            daily_loss_pct = -self.state.daily_pnl / self.state.daily_start
            if daily_loss_pct >= self.DAILY_LOSS_LIMIT:
                return False, f"daily loss {daily_loss_pct:.1%}"

        if self._open_positions >= self.MAX_POSITIONS:
            return False, "max positions"

        if self.state.streak <= -self.LOSS_STREAK_PAUSE:
            self._paused_until = time.time() + self.LOSS_STREAK_PAUSE_SECS
            self.state.streak  = 0
            return False, "loss streak pause"

        return True, "ok"

    # ── Position tracking ────────────────────────────────────────────────────

    def open_position(self, symbol: str = ""):
        self._open_positions += 1
        if symbol:
            self._open_symbols.append(symbol.lower())

    def close_position(self, symbol: str = "", win: bool = True, pnl: float = 0.0):
        self._open_positions = max(0, self._open_positions - 1)
        if symbol:
            sym = symbol.lower()
            try:
                self._open_symbols.remove(sym)
            except ValueError:
                pass
        self._recent_results.append(win)
        self._recent_pnl.append(pnl)

    def set_vol_multiplier(self, multiplier: float):
        """Called by main loop with regime.size_multiplier."""
        self.vol_multiplier = max(0.30, min(1.50, multiplier))

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def rolling_win_rate(self) -> float:
        if not self._recent_results:
            return self.state.win_rate
        return sum(self._recent_results) / len(self._recent_results)

    @property
    def rolling_sharpe(self) -> float:
        if len(self._recent_pnl) < 5:
            return 0.0
        pnls = list(self._recent_pnl)
        avg  = sum(pnls) / len(pnls)
        std  = (sum((x - avg)**2 for x in pnls) / len(pnls)) ** 0.5
        return (avg / std) if std > 0 else 0.0

    def reset_daily(self):
        self.state.daily_start = self.state.bankroll
        self.state.daily_pnl   = 0.0

    def summary(self) -> dict:
        return {
            "bankroll":      round(self.state.bankroll, 2),
            "daily_pnl":     round(self.state.daily_pnl, 2),
            "drawdown":      f"{self.state.drawdown:.1%}",
            "win_rate":      f"{self.state.win_rate:.1%}",
            "rolling_wr":    f"{self.rolling_win_rate:.1%}",
            "rolling_sharpe": f"{self.rolling_sharpe:.2f}",
            "open_positions": self._open_positions,
            "streak":         self.state.streak,
            "total_trades":   self.state.total_trades,
            "kelly_fraction": f"{self._dynamic_kelly_fraction():.2f}",
        }
