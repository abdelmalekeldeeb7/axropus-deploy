"""Strategy 3: BTC 1-Hour — macro trend + funding rate + LLM deep context."""
import logging
import requests
from ..core.feed import BinanceFeed
from ..core.amf_bridge import AMFBridge
from .btc_5min import Signal, _ema

logger = logging.getLogger(__name__)


def get_funding_rate() -> float:
    """Binance perpetual funding rate — positive = longs paying = bearish signal."""
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": "BTCUSDT", "limit": 1}, timeout=3
        )
        if r.ok:
            data = r.json()
            if data:
                return float(data[-1].get("fundingRate", 0))
    except Exception:
        pass
    return 0.0


class BTC1HourStrategy:
    """
    Strategy 3: BTC 1-Hour markets.
    Uses hourly klines, funding rate, RSI, macro trend.
    Higher edge due to more context available for LLM.
    """

    MIN_CONFIDENCE = 0.60
    MIN_VOLUME = 2000.0

    def __init__(self, feed: BinanceFeed, amf: AMFBridge):
        self.feed = feed
        self.amf = amf
        self.name = "btc_1hour"

    def scan(self, markets: list) -> list[Signal]:
        klines_1h = self.feed.get_klines("btcusdt", "1h", 24)
        klines_15m = self.feed.get_klines("btcusdt", "15m", 8)
        prices = self.feed.get_prices("btcusdt")

        if len(klines_1h) < 6 or not prices:
            return []

        direction, confidence = self._analyze(klines_1h, klines_15m, prices)
        if not direction or confidence < self.MIN_CONFIDENCE:
            return []

        funding = get_funding_rate()
        # Funding rate > 0.01% = bullish (longs paying, market tilted up)
        # Funding rate < -0.01% = bearish
        if funding > 0.0001 and direction == "DOWN":
            confidence *= 0.90
        elif funding < -0.0001 and direction == "UP":
            confidence *= 0.90

        btc_now = prices[-1]
        btc_1h = prices[-min(1800, len(prices)-1)]
        btc_change = (btc_now - btc_1h) / btc_1h * 100 if btc_1h > 0 else 0
        eth_prices = self.feed.get_prices("ethusdt")
        eth_now = eth_prices[-1] if eth_prices else 0

        signals = []
        for m in markets:
            if not self._is_1hour_btc(m):
                continue
            if m.volume < self.MIN_VOLUME:
                continue

            token = m.yes_token if direction == "UP" else m.no_token
            price = m.yes_price if direction == "UP" else m.no_price
            if price <= 0 or price >= 0.95:
                continue

            llm = self.amf.analyze(
                question=m.question, yes_price=m.yes_price, no_price=m.no_price,
                btc_price=btc_now, btc_change=btc_change,
                eth_price=eth_now, eth_change=0,
                book_ratio=1.0, spread=abs(m.yes_price - m.no_price),
                momentum=f"{direction} {confidence:.0%} funding={funding:.4f}",
                time_elapsed=0, window_duration=3600
            )

            if llm.direction and llm.direction != direction:
                continue

            combined = confidence * 0.50 + (llm.confidence if llm.direction else 0.5) * 0.50
            if combined < self.MIN_CONFIDENCE:
                continue

            signals.append(Signal(
                direction=direction, confidence=combined,
                entry_price=price, token_id=token,
                market_slug=m.slug, question=m.question,
                tp_pct=0.25, sl_pct=0.12, max_hold=3540.0,
                source=self.name
            ))
        return signals

    def _analyze(self, klines_1h, klines_15m, prices) -> tuple:
        closes = [k["close"] for k in klines_1h]

        # EMA trend
        ema_8  = _ema(closes, 8)
        ema_21 = _ema(closes, min(21, len(closes)))
        trend = "UP" if ema_8 > ema_21 else "DOWN"

        # RSI
        rsi = self._rsi(closes, 14)
        if rsi > 70:
            trend_score = -0.3  # overbought → bearish
        elif rsi < 30:
            trend_score = 0.3   # oversold → bullish
        else:
            trend_score = 0.0

        # Recent 4h price change
        chg_4h = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
        momentum = "UP" if chg_4h > 0 else "DOWN"

        if trend == momentum:
            confidence = 0.68 + min(abs(chg_4h) * 10, 0.10)
        else:
            confidence = 0.52

        if trend_score != 0:
            confidence = min(0.85, confidence + abs(trend_score) * 0.1)

        return trend, round(confidence, 3)

    def _rsi(self, closes: list, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains  = [d for d in deltas[-period:] if d > 0]
        losses = [-d for d in deltas[-period:] if d < 0]
        avg_g  = sum(gains) / period if gains else 0
        avg_l  = sum(losses) / period if losses else 0
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return 100 - (100 / (1 + rs))

    def _is_1hour_btc(self, m) -> bool:
        q = m.question.lower()
        s = m.slug.lower()
        return ("btc" in q or "bitcoin" in q) and (
            "hour" in q or "1h" in s or "60" in q or "-1h-" in s
        )
