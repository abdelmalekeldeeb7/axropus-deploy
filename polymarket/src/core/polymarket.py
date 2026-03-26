"""Polymarket CLOB client — order placement, position tracking, market data."""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import requests

logger = logging.getLogger(__name__)

@dataclass
class Market:
    condition_id: str
    question: str
    slug: str
    yes_token: str
    no_token: str
    yes_price: float
    no_price: float
    volume: float
    end_date_iso: str
    category: str = ""
    active: bool = True

@dataclass
class Position:
    token_id: str
    market_slug: str
    side: str           # YES or NO
    shares: float
    entry_price: float
    cost: float
    open_time: float = field(default_factory=time.time)
    strategy: str = ""

@dataclass
class Order:
    order_id: str
    token_id: str
    side: str
    price: float
    size: float
    status: str = "OPEN"
    filled: float = 0.0


class PolymarketClient:
    """Polymarket CLOB API wrapper."""

    CLOB_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"
    DATA_URL  = "https://data-api.polymarket.com"

    def __init__(self, private_key: str, proxy_wallet: str, mode: str = "paper"):
        self.private_key = private_key
        self.proxy_wallet = proxy_wallet
        self.mode = mode
        self._positions: dict[str, Position] = {}
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

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
                    tokens = m.get("tokens", [m.get("clobTokenIds", ["", ""])])
                    yes_tok = tokens[0] if tokens else ""
                    no_tok  = tokens[1] if len(tokens) > 1 else ""
                    markets.append(Market(
                        condition_id=m.get("conditionId", m.get("condition_id", "")),
                        question=m.get("question", ""),
                        slug=m.get("slug", ""),
                        yes_token=yes_tok if isinstance(yes_tok, str) else yes_tok.get("token_id", ""),
                        no_token=no_tok  if isinstance(no_tok, str) else no_tok.get("token_id", ""),
                        yes_price=float(m.get("outcomePrices", [0.5, 0.5])[0]),
                        no_price=float(m.get("outcomePrices", [0.5, 0.5])[1]),
                        volume=float(m.get("volume", 0)),
                        end_date_iso=m.get("endDateIso", ""),
                        category=m.get("groupItemTitle", ""),
                    ))
                except Exception:
                    continue
            return markets
        except Exception as e:
            logger.warning("get_markets error: %s", e)
            return []

    def get_token_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get best price for a token."""
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
        """Buy shares. size_usd = dollars to spend."""
        if self.mode == "paper":
            return self._paper_buy(token_id, price, size_usd, market_slug, strategy)

        shares = size_usd / price if price > 0 else 0
        if shares < 1:
            return None

        try:
            body = {
                "token_id": token_id,
                "side": "BUY",
                "price": round(price, 4),
                "size": round(shares, 2),
                "type": "GTC",
            }
            r = self._session.post(
                f"{self.CLOB_URL}/order",
                json=body,
                timeout=5
            )
            if r.ok:
                data = r.json()
                oid = data.get("orderID", data.get("order_id", ""))
                order = Order(order_id=oid, token_id=token_id,
                              side="BUY", price=price, size=shares)
                # Track position
                self._positions[token_id] = Position(
                    token_id=token_id, market_slug=market_slug,
                    side="YES", shares=shares, entry_price=price,
                    cost=size_usd, strategy=strategy
                )
                logger.info("BUY %.2f shares @ $%.3f | %s", shares, price, market_slug[:30])
                return order
        except Exception as e:
            logger.error("buy error: %s", e)
        return None

    def sell(self, token_id: str, shares: float,
             market_slug: str = "") -> Optional[Order]:
        """Sell shares to exit position."""
        price = self.get_token_price(token_id, "SELL")
        if not price:
            return None

        if self.mode == "paper":
            return self._paper_sell(token_id, shares, price, market_slug)

        try:
            body = {
                "token_id": token_id,
                "side": "SELL",
                "price": round(price, 4),
                "size": round(shares, 2),
                "type": "GTC",
            }
            r = self._session.post(f"{self.CLOB_URL}/order", json=body, timeout=5)
            if r.ok:
                data = r.json()
                oid = data.get("orderID", data.get("order_id", ""))
                if token_id in self._positions:
                    del self._positions[token_id]
                logger.info("SELL %.2f shares @ $%.3f | %s", shares, price, market_slug[:30])
                return Order(order_id=oid, token_id=token_id,
                             side="SELL", price=price, size=shares, status="FILLED")
        except Exception as e:
            logger.error("sell error: %s", e)
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
        if token_id in self._positions:
            del self._positions[token_id]
        return Order(order_id=f"paper_sell_{int(time.time()*1000)}",
                     token_id=token_id, side="SELL",
                     price=price, size=shares, status="FILLED")

    def get_position(self, token_id: str) -> Optional[Position]:
        return self._positions.get(token_id)

    def get_all_positions(self) -> list[Position]:
        return list(self._positions.values())
