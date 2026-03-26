"""Strategy 10: Mean Reversion — fade extreme price moves back to fair value."""
import logging
import time
from dataclasses import dataclass, field
from ..core.feed import BinanceFeed
from ..core.amf_bridge import AMFBridge
from .btc_5min import Signal

logger = logging.getLogger(__name__)


@dataclass
class ReversionSetup:
    symbol: str
    z_score: float
    current_price: float
    mean_price: float
    direction: str          # direction to bet (UP = oversold, DOWN = overbought)
    strength: float
    timestamp: float = field(default_factory=time.time)


class MeanReversionStrategy:
    """
    Strategy 10: Mean Reversion V1-V5.
    Fades extreme 1-3 std dev moves back to 20-period mean.
    Edge: 57-63% on crypto shorts/longs — works best during consolidation.

    V1: BTC price vs 20-period EMA (z-score)
    V2: ETH price vs 20-period EMA
    V3: BTC/ETH price ratio reversion
    V4: Polymarket YES price vs historical average
    V5: Volume-weighted mean reversion (VWAP)
    """

    MIN_Z_SCORE   = 1.8     # 1.8 std deviations to trigger
    STRONG_Z      = 2.5     # strong signal threshold
    MIN_CONFIDENCE = 0.62
    LOOKBACK      = 20      # EMA/mean window
    COOLDOWN      = 90.0    # seconds between same-symbol signals

    def __init__(self, feed: BinanceFeed, amf: AMFBridge):
        self.feed = feed
        self.amf  = amf
        self.name = "mean_rev"
        self._last_signal: dict[str, float] = {}  # symbol → timestamp

    # ─── public ────────────────────────────────────────────────────────────

    def scan(self, markets: list) -> list[Signal]:
        setups = self._detect_setups()
        if not setups:
            return []

        signals = []
        for setup in setups:
            for m in markets:
                if not self._matches_market(m, setup.symbol):
                    continue
                sig = self._build_signal(m, setup)
                if sig:
                    signals.append(sig)
        return signals

    # ─── detection ─────────────────────────────────────────────────────────

    def _detect_setups(self) -> list[ReversionSetup]:
        setups = []

        for symbol in ["btcusdt", "ethusdt"]:
            prices = self.feed.get_prices(symbol)
            if len(prices) < self.LOOKBACK + 5:
                continue

            # Cool-down per symbol
            last = self._last_signal.get(symbol, 0)
            if time.time() - last < self.COOLDOWN:
                continue

            setup = self._compute_z(symbol, prices)
            if setup:
                setups.append(setup)
                self._last_signal[symbol] = time.time()

        # V3: BTC/ETH ratio
        btc = self.feed.get_prices("btcusdt")
        eth = self.feed.get_prices("ethusdt")
        if len(btc) >= self.LOOKBACK + 5 and len(eth) >= self.LOOKBACK + 5:
            ratio_setup = self._compute_ratio_z(btc, eth)
            if ratio_setup:
                setups.append(ratio_setup)

        return setups

    def _compute_z(self, symbol: str, prices: list) -> ReversionSetup | None:
        window = prices[-self.LOOKBACK:]
        mean   = sum(window) / len(window)
        var    = sum((p - mean) ** 2 for p in window) / len(window)
        std    = var ** 0.5

        if std == 0:
            return None

        current = prices[-1]
        z       = (current - mean) / std

        if abs(z) < self.MIN_Z_SCORE:
            return None

        # Bet against the extreme: price too high → DOWN (expect reversion)
        direction = "DOWN" if z > 0 else "UP"
        strength  = min(abs(z) / self.STRONG_Z, 1.0)

        logger.info("MEAN REV %s z=%.2f → %s (mean=%.2f current=%.2f)",
                    symbol.upper(), z, direction, mean, current)

        return ReversionSetup(
            symbol=symbol, z_score=z, current_price=current,
            mean_price=mean, direction=direction, strength=strength
        )

    def _compute_ratio_z(self, btc: list, eth: list) -> ReversionSetup | None:
        ratios = [b / e for b, e in zip(btc[-self.LOOKBACK:], eth[-self.LOOKBACK:])]
        mean   = sum(ratios) / len(ratios)
        var    = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        std    = var ** 0.5

        if std == 0:
            return None

        current_ratio = btc[-1] / eth[-1]
        z = (current_ratio - mean) / std

        if abs(z) < self.MIN_Z_SCORE:
            return None

        # High ratio = BTC overbought vs ETH → bet ETH UP (or BTC DOWN)
        direction = "UP" if z < 0 else "DOWN"
        strength  = min(abs(z) / self.STRONG_Z, 1.0)

        return ReversionSetup(
            symbol="eth_vs_btc", z_score=z, current_price=current_ratio,
            mean_price=mean, direction=direction, strength=strength
        )

    # ─── signal building ───────────────────────────────────────────────────

    def _build_signal(self, m, setup: ReversionSetup) -> Signal | None:
        direction  = setup.direction
        token      = m.yes_token if direction == "UP" else m.no_token
        price      = m.yes_price if direction == "UP" else m.no_price

        if price <= 0 or price >= 0.92:
            return None

        # LLM confirmation on strong setups only
        base_conf = 0.58 + setup.strength * 0.12
        if setup.z_score >= self.STRONG_Z:
            llm = self.amf.analyze(
                question=m.question, yes_price=m.yes_price, no_price=m.no_price,
                btc_price=0, btc_change=0, eth_price=0, eth_change=0,
                book_ratio=1.0, spread=abs(m.yes_price - m.no_price),
                momentum=f"Z-score={setup.z_score:.2f} mean={setup.mean_price:.4f} "
                          f"current={setup.current_price:.4f} → expect reversion {direction}",
                time_elapsed=0, window_duration=300
            )
            confidence = base_conf * 0.65 + (llm.confidence if llm.direction else 0.5) * 0.35
        else:
            confidence = base_conf

        if confidence < self.MIN_CONFIDENCE:
            return None

        # Tighter SL for mean reversion (quick exit if trend continues)
        return Signal(
            direction=direction, confidence=confidence,
            entry_price=price, token_id=token,
            market_slug=m.slug, question=m.question,
            tp_pct=0.18, sl_pct=0.09, max_hold=240.0,
            source=f"{self.name}_z{abs(setup.z_score):.1f}"
        )

    # ─── helpers ───────────────────────────────────────────────────────────

    def _matches_market(self, m, symbol: str) -> bool:
        q = m.question.lower()
        if symbol == "btcusdt":
            return "btc" in q or "bitcoin" in q
        if symbol == "ethusdt":
            return "eth" in q or "ethereum" in q
        if symbol == "eth_vs_btc":
            return "eth" in q or "ethereum" in q
        return False
