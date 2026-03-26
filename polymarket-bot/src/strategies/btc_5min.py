"""Strategy 1: BTC 5-Minute — 10-signal momentum consensus + elite quant stack."""
import time
import logging
from dataclasses import dataclass
from typing import Optional
from ..core.feed import BinanceFeed
from ..core.amf_bridge import AMFBridge
from ..core.polymarket import PolymarketClient

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    direction:   str      # UP or DOWN
    confidence:  float
    entry_price: float
    token_id:    str
    market_slug: str
    question:    str
    tp1_pct:     float = 0.12   # partial exit at +12%
    tp2_pct:     float = 0.18   # full exit at +18%
    tp_pct:      float = 0.18   # legacy alias (used by expiring/other strategies)
    sl_pct:      float = 0.10
    max_hold:    float = 290.0
    source:      str   = "btc_5min"


def _ema(prices: list, period: int) -> float:
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = prices[-period]
    for p in prices[-period + 1:]:
        ema = p * k + ema * (1 - k)
    return ema


def compute_momentum_signals(
    prices: list[float],
    klines: list[dict],
    feed: BinanceFeed
) -> dict:
    """10-signal consensus — same engine as poly_hft but fixed and tuned."""
    if len(prices) < 30:
        return {"direction": None, "confidence": 0}

    votes = []

    # S1: 10-second momentum
    if len(prices) >= 5:
        chg = (prices[-1] - prices[-5]) / prices[-5]
        if abs(chg) > 0.00002:
            s = min(abs(chg) / 0.0005, 1.0)
            votes.append((1 if chg > 0 else -1, 1.0 + s, "mom_10s"))

    # S2: 30-second momentum
    if len(prices) >= 15:
        chg = (prices[-1] - prices[-15]) / prices[-15]
        if abs(chg) > 0.00003:
            s = min(abs(chg) / 0.001, 1.0)
            votes.append((1 if chg > 0 else -1, 1.2 + s, "mom_30s"))

    # S3: 2-minute trend
    if len(prices) >= 60:
        chg = (prices[-1] - prices[-60]) / prices[-60]
        if abs(chg) > 0.00005:
            s = min(abs(chg) / 0.002, 1.0)
            votes.append((1 if chg > 0 else -1, 1.4 + s, "trend_2m"))

    # S4: 5-minute trend
    if len(prices) >= 150:
        chg = (prices[-1] - prices[-150]) / prices[-150]
        if abs(chg) > 0.0001:
            s = min(abs(chg) / 0.005, 1.0)
            votes.append((1 if chg > 0 else -1, 1.5 + s, "trend_5m"))

    # S5: EMA crossover
    if len(prices) >= 20:
        ema_f = _ema(prices, 5)
        ema_s = _ema(prices, 20)
        gap = abs(ema_f - ema_s) / ema_s if ema_s > 0 else 0
        if gap > 0.00005:
            votes.append((1 if ema_f > ema_s else -1, 1.3, "ema_cross"))

    # S6: Price acceleration
    if len(prices) >= 20:
        mid = len(prices) // 2
        v1 = (prices[mid] - prices[0]) / prices[0] if prices[0] > 0 else 0
        v2 = (prices[-1] - prices[mid]) / prices[mid] if prices[mid] > 0 else 0
        acc = v2 - v1
        if abs(acc) > 0.00002:
            votes.append((1 if acc > 0 else -1, 1.0, "acceleration"))

    # S7: Kline candle trend (4/5 green or red)
    if len(klines) >= 5:
        recent = klines[-5:]
        green = sum(1 for c in recent if c["close"] > c["open"])
        if green >= 4:
            votes.append((1, 1.2, "kline_green"))
        elif green <= 1:
            votes.append((-1, 1.2, "kline_red"))

    # S8: Higher highs / lower lows
    if len(klines) >= 6:
        c1 = [c["close"] for c in klines[-6:-3]]
        c2 = [c["close"] for c in klines[-3:]]
        if c1 and c2:
            if max(c2) > max(c1) and min(c2) > min(c1):
                votes.append((1, 1.0, "higher_highs"))
            elif max(c2) < max(c1) and min(c2) < min(c1):
                votes.append((-1, 1.0, "lower_lows"))

    # S9: Tick consistency (6/7 ticks same direction)
    if len(prices) >= 8:
        recent = prices[-8:]
        ups = sum(1 for i in range(1, 8) if recent[i] > recent[i - 1])
        if ups >= 6:
            votes.append((1, 1.1, "tick_up"))
        elif ups <= 1:
            votes.append((-1, 1.1, "tick_down"))

    # S10: Multi-crypto consensus (BTC+ETH+SOL+BNB) — weight 2.5x
    mc = feed.get_multi_momentum()
    if mc["direction"] and mc["confidence"] >= 0.75:
        w = 2.5 * mc["confidence"]
        votes.append((1 if mc["direction"] == "UP" else -1, w, "multi_crypto"))

    if not votes:
        return {"direction": None, "confidence": 0}

    up_w   = sum(w for v, w, _ in votes if v > 0)
    down_w = sum(w for v, w, _ in votes if v < 0)
    total  = up_w + down_w
    if total == 0:
        return {"direction": None, "confidence": 0}

    max_w     = max(up_w, down_w)
    agreement = (max_w - min(up_w, down_w)) / total
    n_agree   = sum(1 for v, w, _ in votes if (v > 0) == (up_w >= down_w))
    depth_bonus = min(n_agree / 8.0, 1.0)
    confidence  = agreement * (0.5 + 0.5 * depth_bonus)
    direction   = "UP" if up_w >= down_w else "DOWN"

    return {
        "direction":   direction,
        "confidence":  round(confidence, 3),
        "votes":       len(votes),
        "up_weight":   round(up_w, 2),
        "down_weight": round(down_w, 2),
    }


class BTC5MinStrategy:
    """
    Strategy 1: BTC 5-Minute momentum.
    Scans Polymarket for active BTC 5-min markets, fires when
    10-signal consensus confidence >= 0.61 + LLM confirms + Bayesian posterior.

    Elite quant stack:
      - Kalman velocity replaces raw momentum as price evidence
      - VPIN detects if informed traders are behind the move
      - Regime confirms we're in a trending (not volatile) market
      - Bayesian updater fuses all signals into posterior probability
    """

    MIN_CONFIDENCE = 0.61
    MIN_VOLUME     = 500.0

    def __init__(
        self,
        feed:   BinanceFeed,
        amf:    AMFBridge,
        poly:   Optional[PolymarketClient] = None,
        kalman  = None,   # KalmanFilter instance
        vpin    = None,   # VPINCalculator instance
        regime  = None,   # RegimeDetector instance
        bayesian = None,  # BayesianUpdater instance
    ):
        self.feed    = feed
        self.amf     = amf
        self.poly    = poly
        self.kalman  = kalman
        self.vpin    = vpin
        self.regime  = regime
        self.bayesian = bayesian
        self.name    = "btc_5min"

    def scan(self, markets: list) -> list[Signal]:
        """Return signals for all tradeable 5-min BTC markets."""
        prices = self.feed.get_prices("btcusdt")
        if len(prices) < 30:
            return []

        klines = self.feed.get_klines("btcusdt", "1m", 10)
        result = compute_momentum_signals(prices, klines, self.feed)

        if not result["direction"] or result["confidence"] < self.MIN_CONFIDENCE:
            return []

        direction  = result["direction"]
        confidence = result["confidence"]
        btc_now    = prices[-1]
        btc_1m     = prices[-30] if len(prices) >= 30 else prices[0]
        btc_change = (btc_now - btc_1m) / btc_1m * 100

        # ── Kalman velocity signal ─────────────────────────────────────────
        kalman_velocity = 0.0
        if self.kalman:
            self.kalman.update("btcusdt", btc_now)
            ks = self.kalman.get_state("btcusdt")
            if ks:
                kalman_velocity = ks.velocity
                kt = self.kalman.get_trend("btcusdt")
                # If Kalman disagrees with momentum direction, reduce confidence
                if kt["direction"] and kt["direction"] != direction:
                    confidence *= 0.80

        # ── VPIN — informed trading filter ────────────────────────────────
        vpin_signal = {}
        if self.vpin:
            self.vpin.update("btcusdt", btc_now)
            vpin_signal = self.vpin.get_signal("btcusdt")
            # If VPIN is informed AND opposes direction → skip trade
            if (vpin_signal.get("informed") and
                    vpin_signal.get("direction") not in ("NEUTRAL", direction) and
                    vpin_signal.get("confidence", 0) > 0.60):
                logger.debug("VPIN blocks %s signal (informed %s detected)",
                             direction, vpin_signal["direction"])
                return []

        # ── Regime filter ─────────────────────────────────────────────────
        regime_state = None
        if self.regime:
            regime_state = self.regime.update("btcusdt", list(prices))
            from ..core.regime import Regime
            # Don't trade momentum in volatile or ranging regimes
            if regime_state.regime == Regime.VOLATILE:
                logger.debug("Regime=VOLATILE → skip btc_5min")
                return []
            if regime_state.regime == Regime.RANGING and confidence < 0.72:
                logger.debug("Regime=RANGING → skip low-conf btc_5min")
                return []

        eth_prices = self.feed.get_prices("ethusdt")
        eth_now    = eth_prices[-1] if eth_prices else 0
        eth_1m     = eth_prices[-30] if len(eth_prices) >= 30 else eth_now
        eth_change = (eth_now - eth_1m) / eth_1m * 100 if eth_1m > 0 else 0

        signals = []
        for m in markets:
            if not self._is_5min_btc(m):
                continue
            if m.volume < self.MIN_VOLUME:
                continue

            token = m.yes_token if direction == "UP" else m.no_token
            price = m.yes_price if direction == "UP" else m.no_price

            if price <= 0 or price >= 0.95:
                continue

            # ── Real order book scan ───────────────────────────────────────
            ob = {}
            if self.poly:
                ob = self.poly.book_signal(m, direction)
                if ob.get("ob_available") and ob["ob_direction"] not in ("NEUTRAL", direction):
                    if ob["ob_strength"] > 0.4:
                        continue

            book_ratio      = ob.get("book_ratio", 1.0)
            spread          = ob.get("spread", abs(m.yes_price - m.no_price))
            depth_imbalance = ob.get("depth_imbalance", 0.0)

            if ob.get("ob_direction") == direction and ob.get("ob_strength", 0) > 0.3:
                confidence = min(0.92, confidence + ob["ob_strength"] * 0.05)

            # ── LLM confirmation (AMF full stack) ─────────────────────────
            llm = self.amf.analyze(
                question=m.question, yes_price=m.yes_price, no_price=m.no_price,
                btc_price=btc_now, btc_change=btc_change,
                eth_price=eth_now, eth_change=eth_change,
                book_ratio=book_ratio, spread=spread,
                depth_imbalance=depth_imbalance,
                momentum=f"{direction} {confidence:.0%} book={book_ratio:.2f}",
                time_elapsed=0, window_duration=300
            )

            if llm.direction and llm.direction != direction:
                continue

            # ── Bayesian posterior fusion ──────────────────────────────────
            if self.bayesian:
                # Build evidence from each signal source
                price_ev = self.bayesian.build_price_evidence(
                    kalman_velocity, direction=direction
                ) if kalman_velocity != 0 else None

                book_ev = self.bayesian.build_book_evidence(
                    book_ratio, depth_imbalance, direction=direction
                ) if ob else None

                vpin_ev = self.bayesian.build_vpin_evidence(
                    vpin_signal, direction=direction
                ) if vpin_signal else None

                regime_ev = self.bayesian.build_regime_evidence(
                    regime_state, direction=direction
                ) if regime_state else None

                # LLM signal as prior
                llm_prior = (llm.confidence if llm.direction == direction
                             else (1.0 - llm.confidence if llm.direction else 0.60))
                prior = max(llm_prior, confidence) * 0.5 + min(llm_prior, confidence) * 0.5

                posterior = self.bayesian.update(
                    m.slug, prior=prior,
                    price_evidence=price_ev,
                    book_evidence=book_ev,
                    vpin_evidence=vpin_ev,
                    regime_evidence=regime_ev,
                )
                combined = posterior.posterior
                if not posterior.is_strong and combined < self.MIN_CONFIDENCE + 0.03:
                    continue
            else:
                combined = confidence * 0.6 + (llm.confidence if llm.direction else 0.5) * 0.4

            if combined < self.MIN_CONFIDENCE:
                continue

            logger.info("BTC5MIN %s conf=%.2f [Kalman=%.4f VPIN=%s regime=%s]",
                        direction, combined, kalman_velocity,
                        vpin_signal.get("vpin", "N/A"),
                        regime_state.regime.value if regime_state else "N/A")

            signals.append(Signal(
                direction=direction, confidence=combined,
                entry_price=price, token_id=token,
                market_slug=m.slug, question=m.question,
                source=self.name
            ))

        return signals

    def _is_5min_btc(self, m) -> bool:
        q    = m.question.lower()
        slug = m.slug.lower()
        return ("btc" in q or "bitcoin" in q) and (
            "5" in slug or "5-min" in q or "5min" in q or "5 min" in q
        )
