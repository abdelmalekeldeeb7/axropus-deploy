"""Strategy 8: Spread / Market Making — capture bid-ask spread."""
import logging
import time
import requests
from ..core.amf_bridge import AMFBridge
from ..core.polymarket import PolymarketClient
from .btc_5min import Signal

logger = logging.getLogger(__name__)


class MarketMakingStrategy:
    """
    Strategy 8: Spread capture.
    When YES+NO sum < 0.98 → sum arbitrage (guaranteed profit).
    When spread > 4 cents → fade the spread.
    """

    SUM_ARB_THRESHOLD = 0.98   # buy both when sum < this
    MIN_SPREAD = 0.04           # 4 cent spread to fade
    MIN_EDGE = 0.02

    def __init__(self, amf: AMFBridge, poly=None):
        self.amf = amf
        self.poly = poly
        self.name = "market_making"

    def scan(self, markets: list) -> list[Signal]:
        signals = []

        for m in markets:
            # Sum arbitrage (guaranteed)
            total = m.yes_price + m.no_price
            if total < self.SUM_ARB_THRESHOLD and total > 0.60:
                edge = 1.0 - total
                # Return two signals: buy YES + buy NO
                signals.append(Signal(
                    direction="UP", confidence=0.99,
                    entry_price=m.yes_price, token_id=m.yes_token,
                    market_slug=m.slug, question=m.question,
                    tp_pct=edge * 0.9, sl_pct=0.01, max_hold=86400.0,
                    source="sum_arb"
                ))
                signals.append(Signal(
                    direction="DOWN", confidence=0.99,
                    entry_price=m.no_price, token_id=m.no_token,
                    market_slug=m.slug, question=m.question,
                    tp_pct=edge * 0.9, sl_pct=0.01, max_hold=86400.0,
                    source="sum_arb"
                ))
                logger.info("SUM ARB: %s sum=%.3f edge=%.1f%%",
                            m.slug[:30], total, edge * 100)
                continue

            # Spread fade — buy underpriced side
            spread = abs(m.yes_price - m.no_price)
            if spread < self.MIN_SPREAD:
                continue

            # Determine which side is underpriced vs fair value
            fair = 0.50  # assume 50/50 base for active markets
            yes_edge = fair - m.yes_price
            no_edge  = fair - m.no_price

            if yes_edge >= self.MIN_EDGE:
                signals.append(Signal(
                    direction="UP", confidence=min(0.75, 0.55 + yes_edge),
                    entry_price=m.yes_price, token_id=m.yes_token,
                    market_slug=m.slug, question=m.question,
                    tp_pct=yes_edge * 0.8, sl_pct=0.05, max_hold=1800.0,
                    source="spread_fade"
                ))
            elif no_edge >= self.MIN_EDGE:
                signals.append(Signal(
                    direction="DOWN", confidence=min(0.75, 0.55 + no_edge),
                    entry_price=m.no_price, token_id=m.no_token,
                    market_slug=m.slug, question=m.question,
                    tp_pct=no_edge * 0.8, sl_pct=0.05, max_hold=1800.0,
                    source="spread_fade"
                ))

        return signals
