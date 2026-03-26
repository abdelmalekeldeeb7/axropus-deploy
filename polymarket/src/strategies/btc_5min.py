"""Strategy 1: BTC 5-Minute — 10-signal momentum consensus."""
import time
import logging
from dataclasses import dataclass
from typing import Optional
from ..core.feed import BinanceFeed
from ..core.amf_bridge import AMFBridge

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    direction: str      # UP or DOWN
    confidence: float
    entry_price: float
    token_id: str
    market_slug: str
    question: str
    tp_pct: float = 0.18
    sl_pct: float = 0.10
    max_hold: float = 290.0
    source: str = "btc_5min"


def _ema(prices: list, period: int) -> float:
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = prices[-period]
    for p in prices[-period + 1:]:
        ema = p * k + ema * (1 - k)
    return ema


def compute_momentum_signals(prices: list[float], klines: list[dict],
                              feed: BinanceFeed) -> dict:
    """10-signal consensus — same engine as poly_hft but fixed and tuned."""
    if len(prices) < 30:
        return {"direction": None, "confidence": 0}

    votes = []

    # S1: 10-second momentum
    if len(prices) >= 5:
        chg = (prices[-1] - prices[-5]) / prices[-5]
        if abs(chg) > 0.00002:
            s = min(abs(chg) / 0.0005, 1.0)
            votes.append((1 if chg > 0 else -1, 1.0 + s, "mom_10s"))

    # S2: 30-second momentum
    if len(prices) >= 15:
        chg = (prices[-1] - prices[-15]) / prices[-15]
        if abs(chg) > 0.00003:
            s = min(abs(chg) / 0.001, 1.0)
            votes.append((1 if chg > 0 else -1, 1.2 + s, "mom_30s"))

    # S3: 2-minute trend
    if len(prices) >= 60:
        chg = (prices[-1] - prices[-60]) / prices[-60]
        if abs(chg) > 0.00005:
            s = min(abs(chg) / 0.002, 1.0)
            votes.append((1 if chg > 0 else -1, 1.4 + s, "trend_2m"))

    # S4: 5-minute trend
    if len(prices) >= 150:
        chg = (prices[-1] - prices[-150]) / prices[-150]
        if abs(chg) > 0.0001:
            s = min(abs(chg) / 0.005, 1.0)
            votes.append((1 if chg > 0 else -1, 1.5 + s, "trend_5m"))

    # S5: EMA crossover
    if len(prices) >= 20:
        ema_f = _ema(prices, 5)
        ema_s = _ema(prices, 20)
        gap = abs(ema_f - ema_s) / ema_s if ema_s > 0 else 0
        if gap > 0.00005:
            votes.append((1 if ema_f > ema_s else -1, 1.3, "ema_cross"))

    # S6: Price acceleration
    if len(prices) >= 20:
        mid = len(prices) // 2
        v1 = (prices[mid] - prices[0]) / prices[0] if prices[0] > 0 else 0
        v2 = (prices[-1] - prices[mid]) / prices[mid] if prices[mid] > 0 else 0
        acc = v2 - v1
        if abs(acc) > 0.00002:
            votes.append((1 if acc > 0 else -1, 1.0, "acceleration"))

    # S7: Kline candle trend (4/5 green or red)
    if len(klines) >= 5:
        recent = klines[-5:]
        green = sum(1 for c in recent if c["close"] > c["open"])
        if green >= 4:
            votes.append((1, 1.2, "kline_green"))
        elif green <= 1:
            votes.append((-1, 1.2, "kline_red"))

    # S8: Higher highs / lower lows
    if len(klines) >= 6:
        c1 = [c["close"] for c in klines[-6:-3]]
        c2 = [c["close"] for c in klines[-3:]]
        if c1 and c2:
            if max(c2) > max(c1) and min(c2) > min(c1):
                votes.append((1, 1.0, "higher_highs"))
            elif max(c2) < max(c1) and min(c2) < min(c1):
                votes.append((-1, 1.0, "lower_lows"))

    # S9: Tick consistency (6/7 ticks same direction)
    if len(prices) >= 8:
        recent = prices[-8:]
        ups = sum(1 for i in range(1, 8) if recent[i] > recent[i - 1])
        if ups >= 6:
            votes.append((1, 1.1, "tick_up"))
        elif ups <= 1:
            votes.append((-1, 1.1, "tick_down"))

    # S10: Multi-crypto consensus (BTC+ETH+SOL+BNB) — weight 2.5x
    mc = feed.get_multi_momentum()
    if mc["direction"] and mc["confidence"] >= 0.75:
        w = 2.5 * mc["confidence"]
        votes.append((1 if mc["direction"] == "UP" else -1, w, "multi_crypto"))

    if not votes:
        return {"direction": None, "confidence": 0}

    up_w   = sum(w for v, w, _ in votes if v > 0)
    down_w = sum(w for v, w, _ in votes if v < 0)
    total  = up_w + down_w
    if total == 0:
        return {"direction": None, "confidence": 0}

    max_w  = max(up_w, down_w)
    agreement = (max_w - min(up_w, down_w)) / total
    n_agree = sum(1 for v, w, _ in votes if (v > 0) == (up_w >= down_w))
    depth_bonus = min(n_agree / 8.0, 1.0)
    confidence = agreement * (0.5 + 0.5 * depth_bonus)
    direction  = "UP" if up_w >= down_w else "DOWN"

    return {
        "direction": direction,
        "confidence": round(confidence, 3),
        "votes": len(votes),
        "up_weight": round(up_w, 2),
        "down_weight": round(down_w, 2),
    }


class BTC5MinStrategy:
    """
    Strategy 1: BTC 5-Minute momentum.
    Scans Polymarket for active BTC 5-min markets, fires when
    10-signal consensus confidence >= 0.65 + LLM confirms.
    """

    MIN_CONFIDENCE = 0.65
    MIN_VOLUME = 500.0

    def __init__(self, feed: BinanceFeed, amf: AMFBridge):
        self.feed = feed
        self.amf = amf
        self.name = "btc_5min"

    def scan(self, markets: list) -> list[Signal]:
        """Return signals for all tradeable 5-min BTC markets."""
        prices = self.feed.get_prices("btcusdt")
        if len(prices) < 30:
            return []

        klines = self.feed.get_klines("btcusdt", "1m", 10)
        result = compute_momentum_signals(prices, klines, self.feed)

        if not result["direction"] or result["confidence"] < self.MIN_CONFIDENCE:
            return []

        direction = result["direction"]
        confidence = result["confidence"]
        btc_now = prices[-1]
        btc_1m = prices[-30] if len(prices) >= 30 else prices[0]
        btc_change = (btc_now - btc_1m) / btc_1m * 100

        eth_prices = self.feed.get_prices("ethusdt")
        eth_now = eth_prices[-1] if eth_prices else 0
        eth_1m = eth_prices[-30] if len(eth_prices) >= 30 else eth_now
        eth_change = (eth_now - eth_1m) / eth_1m * 100 if eth_1m > 0 else 0

        signals = []
        for m in markets:
            if not self._is_5min_btc(m):
                continue
            if m.volume < self.MIN_VOLUME:
                continue

            # Pick correct token
            if direction == "UP":
                token = m.yes_token
                price = m.yes_price
            else:
                token = m.no_token
                price = m.no_price

            if price <= 0 or price >= 0.95:
                continue

            # LLM confirmation (AMF-cached)
            llm = self.amf.analyze(
                question=m.question, yes_price=m.yes_price, no_price=m.no_price,
                btc_price=btc_now, btc_change=btc_change,
                eth_price=eth_now, eth_change=eth_change,
                book_ratio=1.0, spread=abs(m.yes_price - m.no_price),
                momentum=f"{direction} {confidence:.0%}",
                time_elapsed=0, window_duration=300
            )

            # Blend momentum + LLM confidence
            if llm.direction and llm.direction != direction:
                continue  # LLM disagrees — skip
            combined = confidence * 0.6 + (llm.confidence if llm.direction else 0.5) * 0.4

            if combined < self.MIN_CONFIDENCE:
                continue

            signals.append(Signal(
                direction=direction, confidence=combined,
                entry_price=price, token_id=token,
                market_slug=m.slug, question=m.question,
                source=self.name
            ))

        return signals

    def _is_5min_btc(self, m) -> bool:
        q = m.question.lower()
        slug = m.slug.lower()
        return ("btc" in q or "bitcoin" in q) and (
            "5" in slug or "5-min" in q or "5min" in q or "5 min" in q
        )
