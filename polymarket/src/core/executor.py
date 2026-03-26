"""Trade executor — percentage-based TP/SL, wired to risk manager."""
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable
from .polymarket import PolymarketClient, Position
from .risk import RiskManager

logger = logging.getLogger(__name__)

FEE = 0.02  # Polymarket 2% fee on winnings

@dataclass
class OpenTrade:
    token_id: str
    market_slug: str
    side: str               # YES or NO
    shares: float
    entry_price: float
    cost: float
    direction: str          # UP or DOWN
    strategy: str
    open_time: float = field(default_factory=time.time)
    # Percentage-based exits (scaled to trade size)
    tp_pct: float = 0.18    # take profit at +18% net
    sl_pct: float = 0.10    # stop loss at -10% net
    max_hold: float = 290.0 # 4m50s — just before 5-min resolution


class TradeExecutor:
    """
    Executes trades with percentage-based TP/SL.

    TP/SL scale with cost — works correctly for $1, $10, $100 trades.
    Monitors positions in background thread and exits when conditions met.
    """

    def __init__(self, client: PolymarketClient, risk: RiskManager):
        self.client = client
        self.risk = risk
        self._open: list[OpenTrade] = []
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self.on_close: Optional[Callable] = None  # callback(trade, pnl, reason)

    def start(self):
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

    def stop(self):
        self._running = False
        self._exit_all()

    def enter(self, token_id: str, price: float, size_usd: float,
              market_slug: str, direction: str, strategy: str,
              tp_pct: float = 0.18, sl_pct: float = 0.10,
              max_hold: float = 290.0) -> bool:
        """Open a position. Returns True if order placed."""

        can, reason = self.risk.can_trade()
        if not can:
            logger.debug("Risk block: %s", reason)
            return False

        # Use Kelly size if larger than requested
        kelly = self.risk.kelly_size(0.57, price)
        size_usd = max(size_usd, kelly) if kelly > 0 else size_usd
        size_usd = min(size_usd, self.risk.state.bankroll * 0.12)

        if size_usd < 0.50:
            return False

        order = self.client.buy(token_id, price, size_usd, market_slug, strategy)
        if not order:
            return False

        shares = size_usd / price if price > 0 else 0
        trade = OpenTrade(
            token_id=token_id, market_slug=market_slug,
            side="YES", shares=shares, entry_price=price,
            cost=size_usd, direction=direction, strategy=strategy,
            tp_pct=tp_pct, sl_pct=sl_pct, max_hold=max_hold
        )

        with self._lock:
            self._open.append(trade)

        self.risk.open_position()
        self.risk.state.bankroll -= size_usd
        logger.info("ENTER %s %.2f @ $%.3f cost=$%.2f [%s]",
                    direction, shares, price, size_usd, strategy[:20])
        return True

    def _monitor_loop(self):
        while self._running:
            try:
                self._check_positions()
            except Exception as e:
                logger.error("monitor error: %s", e)
            time.sleep(2)

    def _check_positions(self):
        with self._lock:
            trades = list(self._open)

        still_open = []
        for trade in trades:
            age = time.time() - trade.open_time
            exited, reason, pnl = self._try_exit(trade, age)
            if exited:
                self.risk.close_position()
                if pnl >= 0:
                    self.risk.state.record_win(trade.cost + pnl, trade.cost)
                else:
                    self.risk.state.record_sell(trade.cost + pnl, trade.cost)
                if self.on_close:
                    self.on_close(trade, pnl, reason)
                logger.info("%s #%s pnl=$%+.2f [%s] age=%.0fs",
                            reason, trade.token_id[:8], pnl,
                            trade.strategy[:15], age)
            else:
                still_open.append(trade)

        with self._lock:
            self._open = still_open

    def _try_exit(self, trade: OpenTrade, age: float) -> tuple[bool, str, float]:
        """Check TP/SL/timeout. Returns (exited, reason, net_pnl)."""

        current_price = self.client.get_token_price(trade.token_id, "SELL")
        if current_price is None:
            if age >= trade.max_hold:
                # Force flat — can't read price, mark as hold-to-resolution
                return True, "RESOLUTION", 0.0
            return False, "", 0.0

        # Net P&L if we sell now (after fee on profit)
        sell_value = current_price * trade.shares
        gross_pnl  = sell_value - trade.cost
        fee        = max(0.0, gross_pnl) * FEE
        net_pnl    = gross_pnl - fee

        tp_target = trade.cost * trade.tp_pct   # e.g. $10 * 0.18 = $1.80
        sl_target = -trade.cost * trade.sl_pct  # e.g. $10 * 0.10 = -$1.00

        if net_pnl >= tp_target:
            order = self.client.sell(trade.token_id, trade.shares, trade.market_slug)
            if order:
                return True, "TAKE_PROFIT", net_pnl
            return False, "", 0.0

        if net_pnl <= sl_target:
            order = self.client.sell(trade.token_id, trade.shares, trade.market_slug)
            if order:
                return True, "STOP_LOSS", net_pnl
            return False, "", 0.0

        if age >= trade.max_hold:
            # Hold to resolution — let market pay $1 or $0
            return True, "RESOLUTION", 0.0

        return False, "", 0.0

    def _exit_all(self):
        with self._lock:
            trades = list(self._open)
        for trade in trades:
            self.client.sell(trade.token_id, trade.shares, trade.market_slug)

    @property
    def open_count(self) -> int:
        return len(self._open)
