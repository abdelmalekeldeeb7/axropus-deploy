"""Strategy 7: Copy Trading / Whale Tracking."""
import logging
import time
import requests
from dataclasses import dataclass, field
from ..core.amf_bridge import AMFBridge
from .btc_5min import Signal

logger = logging.getLogger(__name__)

DATA_API = "https://data-api.polymarket.com"

@dataclass
class WhalePosition:
    wallet: str
    market_slug: str
    token_id: str
    side: str
    avg_price: float
    shares: float
    volume_usd: float


class CopyTradingStrategy:
    """
    Strategy 7: Copy top whale wallets.
    Monitors 10 known profitable wallets.
    Enters when whale makes a new position > $100.
    """

    MIN_TRADE_SIZE = 100.0    # only copy trades > $100
    COPY_DELAY = 5.0          # enter 5 seconds after whale
    POLL_INTERVAL = 30.0
    MIN_CONFIDENCE = 0.62

    # Top performing wallets (add real ones here)
    WHALE_WALLETS: list[str] = []

    def __init__(self, amf: AMFBridge, wallets: list[str] = None):
        self.amf = amf
        self.name = "copy_trading"
        self.wallets = wallets or self.WHALE_WALLETS
        self._seen_trades: set[str] = set()
        self._last_poll: dict[str, float] = {}
        self._new_positions: list[WhalePosition] = []

    def add_wallet(self, wallet: str):
        if wallet not in self.wallets:
            self.wallets.append(wallet)

    def poll_whales(self):
        """Background call — fetch new whale positions."""
        for wallet in self.wallets:
            last = self._last_poll.get(wallet, 0)
            if time.time() - last < self.POLL_INTERVAL:
                continue
            self._last_poll[wallet] = time.time()
            self._fetch_recent_trades(wallet)

    def _fetch_recent_trades(self, wallet: str):
        try:
            r = requests.get(
                f"{DATA_API}/trades",
                params={"user": wallet, "limit": 10},
                timeout=5
            )
            if not r.ok:
                return
            for trade in r.json():
                tx = trade.get("transactionHash", "")
                if tx in self._seen_trades:
                    continue
                self._seen_trades.add(tx)

                size = float(trade.get("size", 0))
                price = float(trade.get("price", 0))
                usd = size * price

                if trade.get("side") != "BUY":
                    continue
                if usd < self.MIN_TRADE_SIZE:
                    continue

                self._new_positions.append(WhalePosition(
                    wallet=wallet,
                    market_slug=trade.get("slug", ""),
                    token_id=trade.get("asset", ""),
                    side="YES" if trade.get("outcome", "").lower() in ("yes", "up") else "NO",
                    avg_price=price,
                    shares=size,
                    volume_usd=usd
                ))
                logger.info("Whale %s...%s: %s $%.0f on %s",
                            wallet[:6], wallet[-4:],
                            trade.get("outcome"), usd,
                            trade.get("title", "")[:30])
        except Exception as e:
            logger.debug("whale poll error: %s", e)

    def scan(self, markets: list) -> list[Signal]:
        self.poll_whales()

        signals = []
        market_by_slug = {m.slug: m for m in markets}

        new = list(self._new_positions)
        self._new_positions.clear()

        for wp in new:
            m = market_by_slug.get(wp.market_slug)
            if not m:
                continue

            direction = "UP" if wp.side == "YES" else "DOWN"
            token = m.yes_token if direction == "UP" else m.no_token
            price = m.yes_price if direction == "UP" else m.no_price

            if price <= 0 or price >= 0.92:
                continue

            # Size-based confidence
            if wp.volume_usd >= 1000:
                base_conf = 0.70
            elif wp.volume_usd >= 500:
                base_conf = 0.66
            else:
                base_conf = 0.62

            llm = self.amf.analyze(
                question=m.question, yes_price=m.yes_price, no_price=m.no_price,
                btc_price=0, btc_change=0, eth_price=0, eth_change=0,
                book_ratio=1.0, spread=abs(m.yes_price - m.no_price),
                momentum=f"Whale bought ${wp.volume_usd:.0f} of {wp.side}",
                time_elapsed=0, window_duration=300
            )

            confidence = base_conf * 0.6 + (llm.confidence if llm.direction else 0.5) * 0.4
            if confidence < self.MIN_CONFIDENCE:
                continue

            signals.append(Signal(
                direction=direction, confidence=confidence,
                entry_price=price, token_id=token,
                market_slug=m.slug, question=m.question,
                tp_pct=0.20, sl_pct=0.10, max_hold=600.0,
                source=f"{self.name}_{wp.wallet[:6]}"
            ))

        return signals
