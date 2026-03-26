"""Strategy 2: BTC 15-Minute — trend + order book + LLM."""
import logging
from dataclasses import dataclass
from ..core.feed import BinanceFeed
from ..core.amf_bridge import AMFBridge
from .btc_5min import Signal, compute_momentum_signals

logger = logging.getLogger(__name__)


class BTC15MinStrategy:
    """
    15-min window = more data = higher confidence.
    Uses 15-min klines for trend, combines with 5-min momentum.
    Edge: more time for LLM context = better AMF hit rate.
    """

    MIN_CONFIDENCE = 0.60
    MIN_VOLUME = 1000.0

    def __init__(self, feed: BinanceFeed, amf: AMFBridge):
        self.feed = feed
        self.amf = amf
        self.name = "btc_15min"

    def scan(self, markets: list) -> list[Signal]:
        prices = self.feed.get_prices("btcusdt")
        if len(prices) < 60:
            return []

        # 15-min trend from longer price window
        klines_1m  = self.feed.get_klines("btcusdt", "1m", 15)
        klines_5m  = self.feed.get_klines("btcusdt", "5m", 6)

        mom = compute_momentum_signals(prices, klines_1m, self.feed)
        if not mom["direction"] or mom["confidence"] < self.MIN_CONFIDENCE:
            return []

        # Additional: 15-min trend confirmation from 5m klines
        if len(klines_5m) >= 3:
            last3 = klines_5m[-3:]
            trend_up = all(last3[i]["close"] > last3[i-1]["close"]
                           for i in range(1, 3))
            trend_dn = all(last3[i]["close"] < last3[i-1]["close"]
                           for i in range(1, 3))
            if mom["direction"] == "UP" and not trend_up:
                mom["confidence"] *= 0.85  # penalize if 5m disagrees
            if mom["direction"] == "DOWN" and not trend_dn:
                mom["confidence"] *= 0.85

        btc_now = prices[-1]
        btc_15m = prices[-min(450, len(prices)-1)]
        btc_change = (btc_now - btc_15m) / btc_15m * 100 if btc_15m > 0 else 0
        eth_prices = self.feed.get_prices("ethusdt")
        eth_now = eth_prices[-1] if eth_prices else 0
        eth_change = (eth_now - (eth_prices[-30] if len(eth_prices)>=30 else eth_now)) / eth_now * 100 if eth_now > 0 else 0

        signals = []
        for m in markets:
            if not self._is_15min_btc(m):
                continue
            if m.volume < self.MIN_VOLUME:
                continue

            direction = mom["direction"]
            token = m.yes_token if direction == "UP" else m.no_token
            price = m.yes_price if direction == "UP" else m.no_price
            if price <= 0 or price >= 0.95:
                continue

            llm = self.amf.analyze(
                question=m.question, yes_price=m.yes_price, no_price=m.no_price,
                btc_price=btc_now, btc_change=btc_change,
                eth_price=eth_now, eth_change=eth_change,
                book_ratio=1.0, spread=abs(m.yes_price - m.no_price),
                momentum=f"{direction} {mom['confidence']:.0%} (15m)",
                time_elapsed=0, window_duration=900
            )

            if llm.direction and llm.direction != direction:
                continue

            combined = mom["confidence"] * 0.55 + (llm.confidence if llm.direction else 0.5) * 0.45
            if combined < self.MIN_CONFIDENCE:
                continue

            signals.append(Signal(
                direction=direction, confidence=combined,
                entry_price=price, token_id=token,
                market_slug=m.slug, question=m.question,
                tp_pct=0.22, sl_pct=0.10, max_hold=870.0,
                source=self.name
            ))
        return signals

    def _is_15min_btc(self, m) -> bool:
        q = m.question.lower()
        s = m.slug.lower()
        return ("btc" in q or "bitcoin" in q) and (
            "15" in s or "15-min" in q or "15min" in q
        )
