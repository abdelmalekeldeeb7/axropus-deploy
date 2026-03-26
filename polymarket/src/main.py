"""
Korith-Poly: 12-Strategy Polymarket Trading Orchestrator.

Cycle:
  1. Fetch live markets from Polymarket
  2. Run all 12 strategies in parallel
  3. Deduplicate + rank signals by confidence
  4. Risk check (Kelly sizing, drawdown limits)
  5. Execute top-N signals per cycle
  6. Monitor open positions (TP/SL) in background
"""
import logging
import time
import os
import sys
import threading
from typing import Optional

# ─── Core ──────────────────────────────────────────────────────────────────
from .core.feed       import BinanceFeed
from .core.risk       import RiskManager
from .core.polymarket import PolymarketClient
from .core.amf_bridge import AMFBridge
from .core.executor   import TradeExecutor

# ─── Strategies ────────────────────────────────────────────────────────────
from .strategies.btc_5min        import BTC5MinStrategy
from .strategies.btc_15min       import BTC15MinStrategy
from .strategies.btc_1hour       import BTC1HourStrategy
from .strategies.capitulation    import CapitulationStrategy
from .strategies.no_market       import NoMarketStrategy
from .strategies.weather         import WeatherStrategy
from .strategies.copy_trading    import CopyTradingStrategy
from .strategies.market_making   import MarketMakingStrategy
from .strategies.lag_arbitrage   import LagArbitrageStrategy
from .strategies.mean_reversion  import MeanReversionStrategy
from .strategies.expiring_markets import ExpiringMarketsStrategy
from .strategies.ml_prediction   import MLPredictionStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("korith")


# ─── Config (override via env vars) ───────────────────────────────────────

POLY_KEY         = os.getenv("POLY_PRIVATE_KEY", "")
POLY_PROXY       = os.getenv("POLY_PROXY_WALLET", "")
PAPER_MODE       = os.getenv("PAPER_MODE", "true").lower() == "true"
BANKROLL         = float(os.getenv("BANKROLL", "300.0"))
TRADE_SIZE_USD   = float(os.getenv("TRADE_SIZE_USD", "10.0"))
MAX_SIGNALS_CYCLE = int(os.getenv("MAX_SIGNALS_CYCLE", "3"))
CYCLE_SECONDS    = float(os.getenv("CYCLE_SECONDS", "15.0"))
NEWS_API_KEY     = os.getenv("NEWS_API_KEY", "")
WHALE_WALLETS    = [w for w in os.getenv("WHALE_WALLETS", "").split(",") if w]
LLM_ENDPOINT     = os.getenv("LLM_ENDPOINT", "http://localhost:11434")
DEEPSEEK_KEY     = os.getenv("DEEPSEEK_API_KEY", "")


class Orchestrator:
    """
    Main trading loop.
    Runs all 12 strategies every CYCLE_SECONDS and executes top signals.
    """

    def __init__(self):
        logger.info("=== Korith-Poly starting ===")
        logger.info("Paper mode: %s | Bankroll: $%.0f | Trade size: $%.0f",
                    PAPER_MODE, BANKROLL, TRADE_SIZE_USD)

        # Infrastructure
        self.feed  = BinanceFeed()
        self.risk  = RiskManager(bankroll=BANKROLL)
        self.poly  = PolymarketClient(
            private_key=POLY_KEY, proxy_wallet=POLY_PROXY,
            paper_mode=PAPER_MODE
        )
        self.amf   = AMFBridge(
            endpoint=LLM_ENDPOINT, deepseek_api_key=DEEPSEEK_KEY
        )
        self.exec  = TradeExecutor(
            client=self.poly, risk=self.risk,
            default_size_usd=TRADE_SIZE_USD
        )

        # Build all 12 strategies
        self.strategies = self._build_strategies()

        self._running   = False
        self._cycle_no  = 0
        self._markets   = []
        self._market_ts = 0.0

    def _build_strategies(self) -> list:
        s = [
            BTC5MinStrategy(self.feed, self.amf),           # S1
            BTC15MinStrategy(self.feed, self.amf),           # S2
            BTC1HourStrategy(self.feed, self.amf),           # S3
            CapitulationStrategy(self.feed, self.amf),       # S4
            NoMarketStrategy(self.amf, NEWS_API_KEY),        # S5
            WeatherStrategy(self.amf),                       # S6
            CopyTradingStrategy(self.amf, WHALE_WALLETS),    # S7
            MarketMakingStrategy(self.amf),                  # S8
            LagArbitrageStrategy(self.feed),                 # S9
            MeanReversionStrategy(self.feed, self.amf),      # S10
            ExpiringMarketsStrategy(self.amf),               # S11
            MLPredictionStrategy(self.feed, self.amf),       # S12
        ]
        logger.info("Loaded %d strategies", len(s))
        return s

    # ─── public ────────────────────────────────────────────────────────

    def start(self):
        """Start Binance feed + main loop."""
        self.feed.start()
        time.sleep(2)  # let WS connect

        self._running = True
        logger.info("Main loop started (cycle=%.0fs, max=%d signals)",
                    CYCLE_SECONDS, MAX_SIGNALS_CYCLE)

        try:
            while self._running:
                t0 = time.time()
                self._cycle()
                elapsed = time.time() - t0
                sleep_for = max(0, CYCLE_SECONDS - elapsed)
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self.exec.stop()
        self.feed.stop()
        logger.info("=== Korith-Poly stopped ===")

    # ─── cycle ─────────────────────────────────────────────────────────

    def _cycle(self):
        self._cycle_no += 1
        logger.debug("--- Cycle #%d ---", self._cycle_no)

        markets = self._get_markets()
        if not markets:
            logger.warning("No markets available")
            return

        # Collect signals from all strategies
        all_signals = []
        for strat in self.strategies:
            try:
                sigs = strat.scan(markets)
                all_signals.extend(sigs)
            except Exception as e:
                logger.error("Strategy %s error: %s",
                             getattr(strat, "name", "?"), e)

        if not all_signals:
            return

        # Deduplicate (one signal per market per direction)
        signals = self._deduplicate(all_signals)

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)

        # Risk-check and execute top N
        executed = 0
        for sig in signals:
            if executed >= MAX_SIGNALS_CYCLE:
                break
            ok, reason = self.risk.can_trade()
            if not ok:
                logger.warning("Risk block: %s", reason)
                break

            size = self.risk.kelly_size(sig.confidence, sig.entry_price)
            if size < 2.0:
                continue

            order = self.exec.execute(sig, size_usd=size)
            if order:
                executed += 1

        if executed:
            logger.info("Cycle #%d: %d/%d signals executed",
                        self._cycle_no, executed, len(signals))

    # ─── market fetching ───────────────────────────────────────────────

    def _get_markets(self) -> list:
        now = time.time()
        if now - self._market_ts < 30:
            return self._markets  # cached

        try:
            self._markets = self.poly.get_markets(limit=200)
            self._market_ts = now
        except Exception as e:
            logger.error("Market fetch error: %s", e)
        return self._markets

    # ─── deduplication ─────────────────────────────────────────────────

    def _deduplicate(self, signals: list) -> list:
        """Keep only the highest-confidence signal per (market, direction) pair."""
        seen: dict[str, object] = {}
        for sig in signals:
            key = f"{sig.market_slug}:{sig.direction}"
            if key not in seen or sig.confidence > seen[key].confidence:
                seen[key] = sig
        return list(seen.values())


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    orch = Orchestrator()
    orch.start()


if __name__ == "__main__":
    main()
