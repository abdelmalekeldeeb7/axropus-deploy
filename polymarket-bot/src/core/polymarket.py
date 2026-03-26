"""Polymarket CLOB client — order placement, position tracking, real order book."""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import requests

logger = logging.getLogger(__name__)


@dataclass
class Market:
    condition_id: str
    question:     str
    slug:         str
    yes_token:    str
    no_token:     str
    yes_price:    float
    no_price:     float
    volume:       float
    end_date_iso: str
    category:     str  = ""
    active:       bool = True


@dataclass
class OrderBook:
    """Real-time CLOB order book snapshot."""
    token_id:     str
    bids:         list   # [(price, size), ...]  sorted desc
    asks:         list   # [(price, size), ...]  sorted asc
    timestamp:    float  = field(default_factory=time.time)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 1.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def bid_volume(self) -> float:
        """Total dollar volume on bid side (top 5 levels)."""
        return sum(p * s for p, s in self.bids[:5])

    @property
    def ask_volume(self) -> float:
        """Total dollar volume on ask side (top 5 levels)."""
        return sum(p * s for p, s in self.asks[:5])

    @property
    def book_ratio(self) -> float:
        """
        bid_volume / ask_volume.
        > 1.0 = more buy pressure  → UP signal
        < 1.0 = more sell pressure → DOWN signal
        """
        av = self.ask_volume
        return self.bid_volume / av if av > 0 else 1.0

    @property
    def depth_imbalance(self) -> float:
        """
        (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Range: -1.0 (all asks) to +1.0 (all bids)
        """
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.0
        return (self.bid_volume - self.ask_volume) / total

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2 if self.bids and self.asks else 0.0

    def signal_strength(self) -> tuple[str, float]:
        """
        Returns (direction, strength) based on order book alone.
        direction: 'UP' | 'DOWN' | 'NEUTRAL'
        strength:  0.0 - 1.0
        """
        imb = self.depth_imbalance
        if abs(imb) < 0.10:
            return "NEUTRAL", 0.0
        direction = "UP" if imb > 0 else "DOWN"
        strength  = min(abs(imb) * 2.0, 1.0)   # scale to 0-1
        return direction, strength


@dataclass
class Position:
    token_id:    str
    market_slug: str
    side:        str    # YES or NO
    shares:      float
    entry_price: float
    cost:        float
    open_time:   float = field(default_factory=time.time)
    strategy:    str   = ""


@dataclass
class Order:
    order_id: str
    token_id: str
    side:     str
    price:    float
    size:     float
    status:   str   = "OPEN"
    filled:   float = 0.0


class PolymarketClient:
    """Polymarket CLOB API wrapper with real order book scanning."""

    CLOB_URL  = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"
    DATA_URL  = "https://data-api.polymarket.com"

    # Order book cache TTL — 3 seconds (fast enough for 15s cycles)
    OB_CACHE_TTL = 3.0

    def __init__(self, private_key: str, proxy_wallet: str,
                 paper_mode: bool = True, mode: str = "paper"):
        self.private_key  = private_key
        self.proxy_wallet = proxy_wallet
        self._paper_mode  = paper_mode
        self.mode         = "paper" if paper_mode else mode
        self._positions:  dict[str, Position]  = {}
        self._ob_cache:   dict[str, tuple[OrderBook, float]] = {}
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def set_paper_mode(self, paper: bool):
        self._paper_mode = paper
        self.mode = "paper" if paper else "live"

    # ── Order Book ────────────────────────────────────────────────────────────

    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """
        Fetch real CLOB order book for a token.
        Cached for OB_CACHE_TTL seconds to avoid hammering API.
        """
        now = time.time()
        cached = self._ob_cache.get(token_id)
        if cached:
            ob, ts = cached
            if now - ts < self.OB_CACHE_TTL:
                return ob

        try:
            r = self._session.get(
                f"{self.CLOB_URL}/book",
                params={"token_id": token_id},
                timeout=2
            )
            if not r.ok:
                return None

            data = r.json()
            bids = [(float(b["price"]), float(b["size"]))
                    for b in data.get("bids", []) if b.get("price") and b.get("size")]
            asks = [(float(a["price"]), float(a["size"]))
                    for a in data.get("asks", []) if a.get("price") and a.get("size")]

            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            ob = OrderBook(token_id=token_id, bids=bids, asks=asks)
            self._ob_cache[token_id] = (ob, now)
            return ob

        except Exception as e:
            logger.debug("order book fetch %s: %s", token_id[:12], e)
            return None

    def get_market_book(self, market: Market) -> tuple[Optional[OrderBook], Optional[OrderBook]]:
        """Fetch YES and NO order books for a market."""
        yes_ob = self.get_order_book(market.yes_token) if market.yes_token else None
        no_ob  = self.get_order_book(market.no_token)  if market.no_token  else None
        return yes_ob, no_ob

    def book_signal(self, market: Market, direction: str) -> dict:
        """
        Get complete order book signal for a direction.
        Returns dict with book_ratio, spread, depth_imbalance, ob_direction, ob_strength.
        Used to replace hardcoded book_ratio=1.0 in all strategies.
        """
        token_id = market.yes_token if direction == "UP" else market.no_token
        ob = self.get_order_book(token_id)

        if ob is None:
            # Fallback — use price spread from market data
            return {
                "book_ratio":       1.0,
                "spread":           abs(market.yes_price - market.no_price),
                "depth_imbalance":  0.0,
                "ob_direction":     "NEUTRAL",
                "ob_strength":      0.0,
                "mid_price":        market.yes_price if direction == "UP" else market.no_price,
                "ob_available":     False,
            }

        ob_dir, ob_str = ob.signal_strength()
        return {
            "book_ratio":       ob.book_ratio,
            "spread":           ob.spread,
            "depth_imbalance":  ob.depth_imbalance,
            "ob_direction":     ob_dir,
            "ob_strength":      ob_str,
            "mid_price":        ob.mid_price,
            "ob_available":     True,
        }

    # ── Market Discovery ──────────────────────────────────────────────────────

    def get_markets(self, keyword: str = "", limit: int = 50) -> list[Market]:
        try:
            params = {"active": "true", "closed": "false", "limit": limit}
            if keyword:
                params["keyword"] = keyword
            r = self._session.get(f"{self.GAMMA_URL}/markets", params=params, timeout=5)
            if not r.ok:
                return []
            markets = []
            for m in r.json().get("data", r.json() if isinstance(r.json(), list) else []):
                try:
                    tokens  = m.get("tokens", [m.get("clobTokenIds", ["", ""])])
                    yes_tok = tokens[0] if tokens else ""
                    no_tok  = tokens[1] if len(tokens) > 1 else ""
                    markets.append(Market(
                        condition_id=m.get("conditionId", m.get("condition_id", "")),
                        question=m.get("question", ""),
                        slug=m.get("slug", ""),
                        yes_token=yes_tok if isinstance(yes_tok, str) else yes_tok.get("token_id", ""),
                        no_token=no_tok   if isinstance(no_tok,  str) else no_tok.get("token_id", ""),
                        yes_price=float(m.get("outcomePrices", [0.5, 0.5])[0]),
                        no_price=float(m.get("outcomePrices",  [0.5, 0.5])[1]),
                        volume=float(m.get("volume", 0)),
                        end_date_iso=m.get("endDateIso", ""),
                        category=m.get("groupItemTitle", ""),
                    ))
                except Exception:
                    continue
            return markets
        except Exception as e:
            logger.warning("get_markets: %s", e)
            return []

    def get_token_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        try:
            r = self._session.get(
                f"{self.CLOB_URL}/price",
                params={"token_id": token_id, "side": side},
                timeout=3
            )
            if r.ok:
                return float(r.json().get("price", 0))
        except Exception:
            pass
        return None

    # ── Order Execution ───────────────────────────────────────────────────────

    def buy(self, token_id: str, price: float, size_usd: float,
            market_slug: str = "", strategy: str = "") -> Optional[Order]:
        if self.mode == "paper":
            return self._paper_buy(token_id, price, size_usd, market_slug, strategy)

        shares = size_usd / price if price > 0 else 0
        if shares < 1:
            return None

        try:
            body = {
                "token_id": token_id,
                "side":     "BUY",
                "price":    round(price, 4),
                "size":     round(shares, 2),
                "type":     "GTC",
            }
            r = self._session.post(f"{self.CLOB_URL}/order", json=body, timeout=5)
            if r.ok:
                data = r.json()
                oid  = data.get("orderID", data.get("order_id", ""))
                self._positions[token_id] = Position(
                    token_id=token_id, market_slug=market_slug,
                    side="YES", shares=shares, entry_price=price,
                    cost=size_usd, strategy=strategy
                )
                logger.info("BUY  %.2f shares @ $%.3f | %s", shares, price, market_slug[:30])
                return Order(order_id=oid, token_id=token_id,
                             side="BUY", price=price, size=shares)
        except Exception as e:
            logger.error("buy: %s", e)
        return None

    def sell(self, token_id: str, shares: float,
             market_slug: str = "") -> Optional[Order]:
        price = self.get_token_price(token_id, "SELL")
        if not price:
            return None

        if self.mode == "paper":
            return self._paper_sell(token_id, shares, price, market_slug)

        try:
            body = {
                "token_id": token_id,
                "side":     "SELL",
                "price":    round(price, 4),
                "size":     round(shares, 2),
                "type":     "GTC",
            }
            r = self._session.post(f"{self.CLOB_URL}/order", json=body, timeout=5)
            if r.ok:
                data = r.json()
                oid  = data.get("orderID", data.get("order_id", ""))
                self._positions.pop(token_id, None)
                logger.info("SELL %.2f shares @ $%.3f | %s", shares, price, market_slug[:30])
                return Order(order_id=oid, token_id=token_id,
                             side="SELL", price=price, size=shares, status="FILLED")
        except Exception as e:
            logger.error("sell: %s", e)
        return None

    def exit_all(self):
        for tid in list(self._positions.keys()):
            pos = self._positions[tid]
            self.sell(tid, pos.shares, pos.market_slug)

    # ── Paper Trading ─────────────────────────────────────────────────────────

    def _paper_buy(self, token_id, price, size_usd, market_slug, strategy) -> Order:
        shares = size_usd / price if price > 0 else 0
        self._positions[token_id] = Position(
            token_id=token_id, market_slug=market_slug,
            side="YES", shares=shares, entry_price=price,
            cost=size_usd, strategy=strategy
        )
        return Order(order_id=f"paper_{int(time.time()*1000)}",
                     token_id=token_id, side="BUY",
                     price=price, size=shares, status="FILLED")

    def _paper_sell(self, token_id, shares, price, market_slug) -> Order:
        self._positions.pop(token_id, None)
        return Order(order_id=f"paper_sell_{int(time.time()*1000)}",
                     token_id=token_id, side="SELL",
                     price=price, size=shares, status="FILLED")

    def get_position(self, token_id: str) -> Optional[Position]:
        return self._positions.get(token_id)

    def get_all_positions(self) -> list[Position]:
        return list(self._positions.values())
