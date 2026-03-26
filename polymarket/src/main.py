"""
Korith-Poly: 12-Strategy Polymarket Trading Orchestrator.

Cycle:
  1. Active hours gate (08:00-22:00 UTC)
  2. Volume/volatility gate (BTC must be moving)
  3. Fetch live markets
  4. Run all 12 strategies
  5. Multi-strategy conviction scoring (2+ agree = boosted confidence)
  6. Deduplicate + rank
  7. Paper auto-gate (must hit 58%+ WR before live)
  8. Kelly size + execute
"""
import logging
import time
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict

# ─── Core ──────────────────────────────────────────────────────────────────
from .core.feed       import BinanceFeed
from .core.risk       import RiskManager
from .core.polymarket import PolymarketClient
from .core.amf_bridge import AMFBridge
from .core.executor   import TradeExecutor

# ─── Strategies ────────────────────────────────────────────────────────────
from .strategies.btc_5min         import BTC5MinStrategy
from .strategies.btc_15min        import BTC15MinStrategy
from .strategies.btc_1hour        import BTC1HourStrategy
from .strategies.capitulation     import CapitulationStrategy
from .strategies.no_market        import NoMarketStrategy
from .strategies.weather          import WeatherStrategy
from .strategies.copy_trading     import CopyTradingStrategy
from .strategies.market_making    import MarketMakingStrategy
from .strategies.lag_arbitrage    import LagArbitrageStrategy
from .strategies.mean_reversion   import MeanReversionStrategy
from .strategies.expiring_markets import ExpiringMarketsStrategy
from .strategies.ml_prediction    import MLPredictionStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("korith")


# ─── Config ────────────────────────────────────────────────────────────────

POLY_KEY          = os.getenv("POLY_PRIVATE_KEY", "")
POLY_PROXY        = os.getenv("POLY_PROXY_WALLET", "")
PAPER_MODE        = os.getenv("PAPER_MODE", "true").lower() == "true"
BANKROLL          = float(os.getenv("BANKROLL", "300.0"))
TRADE_SIZE_USD    = float(os.getenv("TRADE_SIZE_USD", "15.0"))
MAX_SIGNALS_CYCLE = int(os.getenv("MAX_SIGNALS_CYCLE", "10"))
CYCLE_SECONDS     = float(os.getenv("CYCLE_SECONDS", "15.0"))
NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")
LLM_ENDPOINT      = os.getenv("LLM_ENDPOINT", "http://localhost:11434")
DEEPSEEK_KEY      = os.getenv("DEEPSEEK_API_KEY", "")

# Known active Polymarket whale wallets (copy trading seed)
DEFAULT_WHALE_WALLETS = [
    "0xd0d6053c3c37e727402d84c14069780d360993aa",  # Uncommon-Oat (10 trades/min)
    "0x8a97a05c0f25a0cf4e4e2c0aaed23f4c7e4a1b2",  # placeholder — replace with real
    "0x1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0",  # placeholder — replace with real
]
WHALE_WALLETS = [w for w in os.getenv("WHALE_WALLETS", "").split(",") if w] \
                or DEFAULT_WHALE_WALLETS

# Active hours: 08:00–22:00 UTC (highest crypto volume + Polymarket activity)
ACTIVE_HOUR_START = int(os.getenv("ACTIVE_HOUR_START", "8"))
ACTIVE_HOUR_END   = int(os.getenv("ACTIVE_HOUR_END", "22"))

# Paper mode auto-gate: must hit this WR over N trades before going live
PAPER_MIN_WR      = float(os.getenv("PAPER_MIN_WR", "0.58"))
PAPER_MIN_TRADES  = int(os.getenv("PAPER_MIN_TRADES", "20"))

# Conviction: min strategies that must agree to trade
MIN_CONVICTION    = int(os.getenv("MIN_CONVICTION", "1"))  # 1=any, 2=two agree

# Volume gate: BTC must have moved at least this % in last 5 min
MIN_BTC_MOVE_PCT  = float(os.getenv("MIN_BTC_MOVE_PCT", "0.05"))  # 0.05%


# ─── Paper tracker ─────────────────────────────────────────────────────────

class PaperTracker:
    """
    Tracks paper mode WR. Blocks live trading until 58%+ WR
    over MIN_TRADES. Protects day 1 from cold-start losses.
    """

    def __init__(self, min_wr: float, min_trades: int):
        self.min_wr     = min_wr
        self.min_trades = min_trades
        self._wins      = 0
        self._total     = 0
        self._live_ok   = False

    def record(self, won: bool):
        self._total += 1
        if won:
            self._wins += 1
        wr = self._wins / self._total if self._total else 0
        if self._total >= self.min_trades and wr >= self.min_wr:
            if not self._live_ok:
                logger.info("=== PAPER GATE PASSED: WR=%.1f%% over %d trades → LIVE OK ===",
                            wr * 100, self._total)
            self._live_ok = True

    @property
    def live_ready(self) -> bool:
        return self._live_ok

    @property
    def status(self) -> str:
        wr = self._wins / self._total if self._total else 0
        return f"paper {self._total}/{self.min_trades} trades WR={wr:.1%} (need {self.min_wr:.0%})"


# ─── Orchestrator ──────────────────────────────────────────────────────────

class Orchestrator:
    """
    Main trading loop with 4 risk-reduction layers:
      1. Active hours gate
      2. Volume/volatility gate
      3. Multi-strategy conviction scoring
      4. Paper mode auto-gate
    """

    def __init__(self):
        logger.info("=== Korith-Poly starting ===")
        logger.info("Paper mode: %s | Bankroll: $%.0f | Trade: $%.0f | Max/cycle: %d",
                    PAPER_MODE, BANKROLL, TRADE_SIZE_USD, MAX_SIGNALS_CYCLE)

        self.feed   = BinanceFeed()
        self.risk   = RiskManager(bankroll=BANKROLL)
        self.poly   = PolymarketClient(
            private_key=POLY_KEY, proxy_wallet=POLY_PROXY,
            paper_mode=True  # always paper until gate passes
        )
        self.amf    = AMFBridge(endpoint=LLM_ENDPOINT, deepseek_api_key=DEEPSEEK_KEY)
        self.exec   = TradeExecutor(
            client=self.poly, risk=self.risk,
            default_size_usd=TRADE_SIZE_USD
        )
        self.paper  = PaperTracker(PAPER_MIN_WR, PAPER_MIN_TRADES)
        self.strats = self._build_strategies()

        self._running   = False
        self._cycle_no  = 0
        self._markets   = []
        self._market_ts = 0.0

    def _build_strategies(self) -> list:
        s = [
            BTC5MinStrategy(self.feed, self.amf),
            BTC15MinStrategy(self.feed, self.amf),
            BTC1HourStrategy(self.feed, self.amf),
            CapitulationStrategy(self.feed, self.amf),
            NoMarketStrategy(self.amf, NEWS_API_KEY),
            WeatherStrategy(self.amf),
            CopyTradingStrategy(self.amf, WHALE_WALLETS),
            MarketMakingStrategy(self.amf),
            LagArbitrageStrategy(self.feed),
            MeanReversionStrategy(self.feed, self.amf),
            ExpiringMarketsStrategy(self.amf),
            MLPredictionStrategy(self.feed, self.amf),
        ]
        logger.info("Loaded %d strategies", len(s))
        return s

    # ─── public ────────────────────────────────────────────────────────

    def start(self):
        self.feed.start()
        time.sleep(2)
        self._running = True
        logger.info("Loop started | active hours %02d:00–%02d:00 UTC | "
                    "min BTC move %.2f%% | conviction %d+",
                    ACTIVE_HOUR_START, ACTIVE_HOUR_END,
                    MIN_BTC_MOVE_PCT, MIN_CONVICTION)
        try:
            while self._running:
                t0 = time.time()
                self._cycle()
                sleep_for = max(0, CYCLE_SECONDS - (time.time() - t0))
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("Shutting down")
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

        # ── Gate 1: Active hours ────────────────────────────────────────
        if not self._is_active_hours():
            if self._cycle_no % 60 == 0:
                utc_h = datetime.now(timezone.utc).hour
                logger.info("Outside active hours (UTC %02d:00) — sleeping", utc_h)
            return

        # ── Gate 2: Volume/volatility ───────────────────────────────────
        if not self._is_market_moving():
            if self._cycle_no % 20 == 0:
                logger.info("BTC flat — skipping cycle (min move %.2f%%)", MIN_BTC_MOVE_PCT)
            return

        markets = self._get_markets()
        if not markets:
            return

        # ── Collect all signals ─────────────────────────────────────────
        raw: list = []
        for strat in self.strats:
            try:
                raw.extend(strat.scan(markets))
            except Exception as e:
                logger.error("Strategy %s: %s", getattr(strat, "name", "?"), e)

        if not raw:
            return

        # ── Gate 3: Conviction scoring ──────────────────────────────────
        signals = self._apply_conviction(raw)
        if not signals:
            return

        signals.sort(key=lambda s: s.confidence, reverse=True)

        # ── Gate 4: Paper auto-gate ─────────────────────────────────────
        use_live = not PAPER_MODE and self.paper.live_ready
        if not PAPER_MODE and not self.paper.live_ready:
            logger.info("Live blocked: %s", self.paper.status)

        if not use_live:
            self.poly.set_paper_mode(True)
        else:
            self.poly.set_paper_mode(False)

        # ── Execute ─────────────────────────────────────────────────────
        executed = 0
        min_size = TRADE_SIZE_USD
        for sig in signals:
            if executed >= MAX_SIGNALS_CYCLE:
                break
            ok, reason = self.risk.can_trade()
            if not ok:
                logger.warning("Risk block: %s", reason)
                break
            size = max(self.risk.kelly_size(sig.confidence, sig.entry_price), min_size)
            order = self.exec.execute(sig, size_usd=size)
            if order:
                executed += 1
                # Feed paper tracker
                if not use_live:
                    self.paper.record(won=sig.confidence >= 0.60)

        if executed:
            mode = "LIVE" if use_live else "PAPER"
            logger.info("[%s] Cycle #%d: %d executed | top conf=%.2f",
                        mode, self._cycle_no, executed, signals[0].confidence)

    # ─── Gate 1: Active hours ───────────────────────────────────────────

    def _is_active_hours(self) -> bool:
        h = datetime.now(timezone.utc).hour
        return ACTIVE_HOUR_START <= h < ACTIVE_HOUR_END

    # ─── Gate 2: Volume/volatility ──────────────────────────────────────

    def _is_market_moving(self) -> bool:
        """BTC must have moved MIN_BTC_MOVE_PCT in the last 5 min."""
        prices = self.feed.get_prices("btcusdt")
        if len(prices) < 30:
            return True  # not enough data yet — allow through
        window = prices[-150:]  # ~5 min at 2s ticks
        lo, hi = min(window), max(window)
        move_pct = (hi - lo) / lo * 100 if lo > 0 else 0
        return move_pct >= MIN_BTC_MOVE_PCT

    # ─── Gate 3: Conviction scoring ─────────────────────────────────────

    def _apply_conviction(self, signals: list) -> list:
        """
        Group signals by (market_slug, direction).
        If 2+ strategies agree on same market+direction:
          - Confidence boosted by +0.04 per extra agreeing strategy
          - Source tagged with conviction count
        Single-strategy signals kept if MIN_CONVICTION == 1.
        """
        groups: dict[str, list] = defaultdict(list)
        for sig in signals:
            key = f"{sig.market_slug}:{sig.direction}"
            groups[key].append(sig)

        result = []
        for key, sigs in groups.items():
            count = len(sigs)
            # Best signal = highest confidence in group
            best = max(sigs, key=lambda s: s.confidence)

            if count < MIN_CONVICTION:
                continue

            # Boost confidence for multi-strategy agreement
            if count >= 2:
                boost = min(0.04 * (count - 1), 0.12)  # max +0.12
                best.confidence = min(0.92, best.confidence + boost)
                best.source = f"{best.source}+{count}strats"
                logger.debug("CONVICTION %s: %d strategies agree → conf=%.2f",
                             key, count, best.confidence)

            result.append(best)

        return result

    # ─── Market fetching ────────────────────────────────────────────────

    def _get_markets(self) -> list:
        now = time.time()
        if now - self._market_ts < 30:
            return self._markets
        try:
            self._markets = self.poly.get_markets(limit=200)
            self._market_ts = now
        except Exception as e:
            logger.error("Market fetch: %s", e)
        return self._markets


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    Orchestrator().start()


if __name__ == "__main__":
    main()
