"""Real-time price feed — Binance WebSocket + REST fallback."""
import asyncio
import json
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import requests
import websocket

logger = logging.getLogger(__name__)

@dataclass
class PriceBar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float

class BinanceFeed:
    """Live BTC/ETH/SOL/BNB price feed via WebSocket with REST fallback."""

    SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt"]
    TICKS = 300  # keep last 300 ticks (~10 min at 2s)

    def __init__(self):
        self._prices: dict[str, deque] = {s: deque(maxlen=self.TICKS) for s in self.SYMBOLS}
        self._klines: dict[str, deque] = {s: deque(maxlen=50) for s in self.SYMBOLS}
        self._last_update: dict[str, float] = {s: 0.0 for s in self.SYMBOLS}
        self._lock = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()
        # Seed with REST prices immediately
        self._seed_rest()
        logger.info("BinanceFeed started")

    def stop(self):
        self._running = False

    def _seed_rest(self):
        for sym in self.SYMBOLS:
            try:
                r = requests.get(
                    f"https://api.binance.us/api/v3/ticker/price?symbol={sym.upper()}",
                    timeout=3
                )
                if r.ok:
                    p = float(r.json()["price"])
                    with self._lock:
                        self._prices[sym].append(p)
                        self._last_update[sym] = time.time()
            except Exception:
                pass

    def _run_ws(self):
        streams = "/".join(f"{s}@trade" for s in self.SYMBOLS)
        url = f"wss://stream.binance.us:9443/stream?streams={streams}"

        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                stream = data.get("stream", "")
                sym = stream.split("@")[0]
                price = float(data["data"]["p"])
                with self._lock:
                    self._prices[sym].append(price)
                    self._last_update[sym] = time.time()
            except Exception:
                pass

        def on_error(ws, err):
            logger.warning("WS error: %s", err)

        def on_close(ws, *args):
            if self._running:
                time.sleep(2)
                self._run_ws()

        ws = websocket.WebSocketApp(url, on_message=on_message,
                                    on_error=on_error, on_close=on_close)
        ws.run_forever()

    def get_prices(self, symbol: str) -> list[float]:
        with self._lock:
            return list(self._prices.get(symbol, []))

    def get_price(self, symbol: str = "btcusdt") -> float:
        prices = self.get_prices(symbol)
        return prices[-1] if prices else 0.0

    def is_stale(self, symbol: str = "btcusdt", max_age: float = 30.0) -> bool:
        return time.time() - self._last_update.get(symbol, 0) > max_age

    def get_klines(self, symbol: str = "btcusdt", interval: str = "1m", limit: int = 10) -> list:
        try:
            r = requests.get(
                "https://api.binance.us/api/v3/klines",
                params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
                timeout=3
            )
            if r.ok:
                raw = r.json()
                return [{"open": float(k[1]), "high": float(k[2]),
                         "low": float(k[3]), "close": float(k[4]),
                         "volume": float(k[5])} for k in raw]
        except Exception:
            pass
        return []

    def get_multi_momentum(self) -> dict:
        """BTC+ETH+SOL+BNB consensus — highest weight signal."""
        votes = []
        for sym in self.SYMBOLS:
            prices = self.get_prices(sym)
            if len(prices) < 10:
                continue
            chg = (prices[-1] - prices[-10]) / prices[-10]
            if abs(chg) > 0.0001:
                votes.append(1 if chg > 0 else -1)

        if not votes:
            return {"direction": None, "confidence": 0}

        up = sum(1 for v in votes if v > 0)
        conf = max(up, len(votes) - up) / len(votes)
        direction = "UP" if up >= len(votes) - up else "DOWN"
        return {"direction": direction, "confidence": conf, "n": len(votes)}
