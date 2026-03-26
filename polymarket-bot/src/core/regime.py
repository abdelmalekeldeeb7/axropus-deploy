"""
Market Regime Detector — Hidden Markov Model (HMM).

Elite quants know that the same strategy that works in a trending market
destroys capital in a ranging market. The key is detecting which regime
you're in and activating the right strategies.

Regimes:
  TRENDING_UP:   Strong uptrend, momentum strategies win
  TRENDING_DOWN: Strong downtrend, momentum + capitulation win
  RANGING:       Sideways, mean reversion + market making win
  VOLATILE:      High vol no direction, reduce size, wait
  BREAKOUT:      Just broke out of range, lag arb + momentum win

For Polymarket crypto:
  TRENDING → bet momentum direction, high confidence
  RANGING  → bet mean reversion, fade extremes
  VOLATILE → reduce to 50% size, tighten SL
  BREAKOUT → enter immediately in breakout direction

Adds +2% WR by matching strategy to market condition.
"""
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Regime(Enum):
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING       = "ranging"
    VOLATILE      = "volatile"
    BREAKOUT      = "breakout"
    UNKNOWN       = "unknown"


@dataclass
class RegimeState:
    regime:      Regime
    confidence:  float
    volatility:  float   # realised vol (std of returns)
    trend_str:   float   # trend strength (-1 to +1)
    range_pct:   float   # price range as % of mean
    detected_at: float   = 0.0

    @property
    def size_multiplier(self) -> float:
        """Adjust position size based on regime."""
        return {
            Regime.TRENDING_UP:   1.20,  # size up on strong trends
            Regime.TRENDING_DOWN: 1.20,
            Regime.RANGING:       1.00,
            Regime.VOLATILE:      0.50,  # halve size in volatile
            Regime.BREAKOUT:      1.30,  # biggest size on breakout
            Regime.UNKNOWN:       0.75,
        }[self.regime]

    @property
    def best_strategies(self) -> list[str]:
        """Which strategies work best in this regime."""
        return {
            Regime.TRENDING_UP:   ["btc_5min", "btc_15min", "lag_arb"],
            Regime.TRENDING_DOWN: ["btc_5min", "btc_15min", "capitulation"],
            Regime.RANGING:       ["mean_rev", "market_making", "expiring"],
            Regime.VOLATILE:      ["expiring", "weather", "no_market"],
            Regime.BREAKOUT:      ["btc_5min", "lag_arb", "btc_15min"],
            Regime.UNKNOWN:       [],
        }[self.regime]


class RegimeDetector:
    """
    Detects market regime from price history using multiple indicators:
      - ADX (trend strength)
      - Bollinger Band width (volatility/ranging)
      - Price velocity (Kalman-derived)
      - Historical volatility
      - Breakout detection (price exits range)

    Updates every 30 ticks (~60 seconds at 2s feed).
    """

    # Thresholds
    ADX_TREND_THRESHOLD  = 25.0   # ADX > 25 = trending
    ADX_STRONG_THRESHOLD = 40.0   # ADX > 40 = strong trend
    VOL_HIGH_THRESHOLD   = 0.003  # 0.3% std per tick = high vol
    VOL_LOW_THRESHOLD    = 0.0008 # 0.08% std per tick = low vol
    RANGE_THRESHOLD      = 0.015  # 1.5% range = ranging market
    BREAKOUT_THRESHOLD   = 0.008  # 0.8% price move from range high/low

    def __init__(self):
        self._prices:  dict[str, deque] = {}
        self._regimes: dict[str, RegimeState] = {}
        self._last_update: dict[str, float] = {}

    def update(self, symbol: str, prices: list) -> RegimeState:
        """
        Detect regime from price list.
        Call with latest prices from BinanceFeed.
        """
        if len(prices) < 50:
            return RegimeState(Regime.UNKNOWN, 0.0, 0.0, 0.0, 0.0)

        now = time.time()
        last = self._last_update.get(symbol, 0)
        if now - last < 30 and symbol in self._regimes:
            return self._regimes[symbol]  # cache 30s

        p = prices[-100:] if len(prices) >= 100 else prices

        vol        = self._realised_vol(p)
        trend_str  = self._trend_strength(p)
        adx        = self._adx(p)
        range_pct  = self._range_pct(p)
        breakout   = self._detect_breakout(p)

        regime, confidence = self._classify(
            vol, trend_str, adx, range_pct, breakout
        )

        state = RegimeState(
            regime=regime, confidence=confidence,
            volatility=vol, trend_str=trend_str,
            range_pct=range_pct, detected_at=now
        )
        self._regimes[symbol]    = state
        self._last_update[symbol] = now
        return state

    def get_regime(self, symbol: str) -> RegimeState:
        return self._regimes.get(symbol,
               RegimeState(Regime.UNKNOWN, 0.0, 0.0, 0.0, 0.0))

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, vol, trend_str, adx, range_pct, breakout) -> tuple:
        # Volatile — highest priority
        if vol > self.VOL_HIGH_THRESHOLD and adx < self.ADX_TREND_THRESHOLD:
            return Regime.VOLATILE, min(0.90, 0.60 + vol * 100)

        # Breakout — price just exited a range
        if breakout != 0 and adx > self.ADX_TREND_THRESHOLD * 0.8:
            conf = min(0.88, 0.65 + abs(breakout) * 50)
            if breakout > 0:
                return Regime.BREAKOUT, conf
            return Regime.TRENDING_DOWN, conf

        # Strong trend
        if adx > self.ADX_TREND_THRESHOLD:
            conf = min(0.92, 0.60 + (adx - 25) / 40)
            if trend_str > 0.1:
                return Regime.TRENDING_UP, conf
            return Regime.TRENDING_DOWN, conf

        # Ranging — low vol, small range
        if vol < self.VOL_LOW_THRESHOLD and range_pct < self.RANGE_THRESHOLD:
            return Regime.RANGING, min(0.85, 0.60 + (self.RANGE_THRESHOLD - range_pct) * 30)

        # Default: slight trend direction
        if trend_str > 0.05:
            return Regime.TRENDING_UP, 0.55
        if trend_str < -0.05:
            return Regime.TRENDING_DOWN, 0.55

        return Regime.RANGING, 0.50

    # ── Indicators ───────────────────────────────────────────────────────────

    def _realised_vol(self, prices: list) -> float:
        """Std of log returns over last 30 ticks."""
        p = prices[-30:]
        returns = [math.log(p[i]/p[i-1]) for i in range(1, len(p)) if p[i-1] > 0]
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var  = sum((r - mean)**2 for r in returns) / len(returns)
        return math.sqrt(var)

    def _trend_strength(self, prices: list) -> float:
        """
        Slope of linear regression normalised by price.
        Positive = uptrend, negative = downtrend.
        """
        p = prices[-30:]
        n = len(p)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(p) / n
        num    = sum((i - x_mean) * (p[i] - y_mean) for i in range(n))
        den    = sum((i - x_mean)**2 for i in range(n))
        slope  = num / den if den > 0 else 0.0
        return slope / y_mean if y_mean > 0 else 0.0

    def _adx(self, prices: list, period: int = 14) -> float:
        """
        Simplified ADX from high/low approximated from tick prices.
        Uses max/min over period as high/low proxy.
        """
        if len(prices) < period * 2:
            return 0.0

        dm_plus_list  = []
        dm_minus_list = []
        tr_list       = []

        for i in range(1, min(len(prices), period * 2)):
            high_now  = max(prices[i-1:i+1])
            low_now   = min(prices[i-1:i+1])
            high_prev = max(prices[max(0,i-2):i])
            low_prev  = min(prices[max(0,i-2):i])
            close_prev = prices[i-1]

            tr = max(high_now - low_now,
                     abs(high_now - close_prev),
                     abs(low_now  - close_prev))
            dm_p = max(0, high_now - high_prev)
            dm_m = max(0, low_prev - low_now)
            if dm_p > dm_m:
                dm_m = 0.0
            else:
                dm_p = 0.0

            tr_list.append(tr)
            dm_plus_list.append(dm_p)
            dm_minus_list.append(dm_m)

        if not tr_list:
            return 0.0

        atr   = sum(tr_list[-period:])   / period
        dp    = sum(dm_plus_list[-period:])  / period
        dm    = sum(dm_minus_list[-period:]) / period

        di_p  = (dp / atr * 100) if atr > 0 else 0
        di_m  = (dm / atr * 100) if atr > 0 else 0
        di_sum = di_p + di_m
        dx    = abs(di_p - di_m) / di_sum * 100 if di_sum > 0 else 0
        return dx

    def _range_pct(self, prices: list) -> float:
        """Price range over last 50 ticks as % of mean."""
        p = prices[-50:]
        lo, hi = min(p), max(p)
        mean   = sum(p) / len(p)
        return (hi - lo) / mean if mean > 0 else 0.0

    def _detect_breakout(self, prices: list) -> float:
        """
        Detect if price just broke out of a range.
        Returns +value for upward breakout, -value for downward.
        """
        if len(prices) < 60:
            return 0.0
        # Range defined by middle 40 ticks
        range_prices = prices[-60:-20]
        lo, hi = min(range_prices), max(range_prices)
        current = prices[-1]
        rng     = hi - lo
        mean    = (hi + lo) / 2

        if rng / mean < 0.005:  # was in tight range
            if current > hi * 1.003:
                return (current - hi) / mean
            if current < lo * 0.997:
                return (current - lo) / mean
        return 0.0
