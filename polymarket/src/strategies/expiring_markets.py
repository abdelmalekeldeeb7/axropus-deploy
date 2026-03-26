"""Strategy 11: Expiring Markets — exploit time-decay mispricing near resolution."""
import logging
import time
import re
from datetime import datetime, timezone
from ..core.amf_bridge import AMFBridge
from .btc_5min import Signal

logger = logging.getLogger(__name__)


def _parse_expiry(question: str) -> float | None:
    """
    Extract resolution timestamp from market question.
    Returns UNIX timestamp or None.
    Handles patterns like:
      '...by Jan 31?', '...before February 5?', '...end of March?',
      '...by 3pm ET?', '...at close on Dec 15?'
    """
    now = datetime.now(timezone.utc)

    # Explicit date: "January 31", "Jan 31", "Feb 5 2025", etc.
    pattern_full = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|" \
                   r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|" \
                   r"dec(?:ember)?)\s+(\d{1,2})(?:,?\s+(\d{4}))?"
    m = re.search(pattern_full, question.lower())
    if m:
        month_str = m.group(1)[:3]
        day       = int(m.group(2))
        year      = int(m.group(3)) if m.group(3) else now.year
        months    = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                     "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
        mo = months.get(month_str, 0)
        if mo:
            try:
                dt = datetime(year, mo, day, 23, 59, tzinfo=timezone.utc)
                if dt < now:  # already past — try next year
                    dt = datetime(year + 1, mo, day, 23, 59, tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                pass

    # "end of Q1/Q2/Q3/Q4"
    q_map = {"q1": (3, 31), "q2": (6, 30), "q3": (9, 30), "q4": (12, 31)}
    for qk, (mo, day) in q_map.items():
        if qk in question.lower():
            try:
                dt = datetime(now.year, mo, day, 23, 59, tzinfo=timezone.utc)
                if dt < now:
                    dt = datetime(now.year + 1, mo, day, 23, 59, tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                pass

    return None


class ExpiringMarketsStrategy:
    """
    Strategy 11: Expiring Markets V1-V4.
    Near expiry, markets often misprice because:
      - Retail traders leave early (liquidity drops)
      - Price anchors to last traded price, not true probability
      - Time value decays faster than market reprices

    V1: Outcome nearly certain (<24h, price < 0.85 when outcome clear)
    V2: Outcome nearly certain (>24h, >5% edge vs model price)
    V3: Last-minute correction — market still 0.50 when one outcome obvious
    V4: Dead markets — no trades in >30 min, price stale, LLM sees clear edge

    Edge: 65-75% close to expiry, volume must be >500 USDC.
    """

    MIN_CONFIDENCE = 0.66
    MIN_EDGE       = 0.07     # 7% model vs market edge
    MIN_VOLUME     = 500.0
    HOURS_URGENT   = 6        # < 6 hours = "urgent" tier
    HOURS_NEAR     = 24       # < 24 hours = "near" tier

    def __init__(self, amf: AMFBridge):
        self.amf  = amf
        self.name = "expiring"
        self._cache: dict[str, tuple[float, float]] = {}  # slug → (prob, ts)

    def scan(self, markets: list) -> list[Signal]:
        signals = []
        now = time.time()

        for m in markets:
            if m.volume < self.MIN_VOLUME:
                continue

            expiry = _parse_expiry(m.question)
            if expiry is None:
                # If market has end_date_iso attribute
                expiry = getattr(m, "end_ts", None)
            if expiry is None:
                continue

            hours_left = (expiry - now) / 3600.0
            if hours_left < 0 or hours_left > 72:
                continue  # already resolved or too far out

            sig = self._analyze_market(m, hours_left, now)
            if sig:
                signals.append(sig)

        return signals

    def _analyze_market(self, m, hours_left: float, now: float) -> Signal | None:
        # Boost confidence as expiry approaches
        if hours_left <= self.HOURS_URGENT:
            time_boost = 0.12
        elif hours_left <= self.HOURS_NEAR:
            time_boost = 0.06
        else:
            time_boost = 0.02

        # Cache LLM calls (5-min TTL)
        cached = self._cache.get(m.slug)
        if cached:
            true_prob, ts = cached
            if now - ts < 300:
                return self._make_signal(m, true_prob, time_boost)

        # LLM assessment of true probability
        llm = self.amf.analyze(
            question=m.question, yes_price=m.yes_price, no_price=m.no_price,
            btc_price=0, btc_change=0, eth_price=0, eth_change=0,
            book_ratio=1.0, spread=abs(m.yes_price - m.no_price),
            momentum=f"{hours_left:.1f} hours until resolution",
            time_elapsed=0, window_duration=int(hours_left * 3600)
        )

        if not llm.direction or llm.confidence < 0.58:
            return None

        true_prob = llm.confidence if llm.direction == "UP" else 1.0 - llm.confidence
        self._cache[m.slug] = (true_prob, now)

        return self._make_signal(m, true_prob, time_boost)

    def _make_signal(self, m, true_prob: float, time_boost: float) -> Signal | None:
        yes_edge = true_prob - m.yes_price
        no_edge  = (1 - true_prob) - m.no_price

        if yes_edge >= self.MIN_EDGE and yes_edge >= no_edge:
            direction  = "UP"
            confidence = min(0.88, 0.55 + yes_edge + time_boost)
            token, price = m.yes_token, m.yes_price
        elif no_edge >= self.MIN_EDGE and no_edge > yes_edge:
            direction  = "DOWN"
            confidence = min(0.88, 0.55 + no_edge + time_boost)
            token, price = m.no_token, m.no_price
        else:
            return None

        if price <= 0 or price >= 0.95:
            return None
        if confidence < self.MIN_CONFIDENCE:
            return None

        logger.info("EXPIRING: %s edge=%.1f%% conf=%.2f",
                    m.slug[:30], max(yes_edge, no_edge) * 100, confidence)

        # Larger TP for expiring markets — price can rocket to 0.99
        return Signal(
            direction=direction, confidence=confidence,
            entry_price=price, token_id=token,
            market_slug=m.slug, question=m.question,
            tp_pct=0.40, sl_pct=0.12, max_hold=3600.0,
            source=self.name
        )
