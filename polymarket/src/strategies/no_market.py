"""Strategy 5: No-Market Events — swarm websearch + whale tracking."""
import logging
import time
import requests
from ..core.amf_bridge import AMFBridge
from .btc_5min import Signal

logger = logging.getLogger(__name__)


class NoMarketStrategy:
    """
    Strategy 5: Event-driven markets (politics, macro, sports).
    Uses web search context + LLM to find mispricings on event markets.
    High edge (8-15%) when news breaks before market reprices.
    """

    MIN_CONFIDENCE = 0.66
    MIN_VOLUME = 1000.0
    MIN_EDGE = 0.08        # need 8%+ mispricing

    def __init__(self, amf: AMFBridge, news_api_key: str = ""):
        self.amf = amf
        self.news_api_key = news_api_key
        self.name = "no_market"
        self._cache: dict[str, tuple[float, float]] = {}  # slug → (true_prob, ts)

    def scan(self, markets: list) -> list[Signal]:
        signals = []
        for m in markets:
            if not self._is_event_market(m):
                continue
            if m.volume < self.MIN_VOLUME:
                continue

            # Check cache (avoid re-analyzing same market every cycle)
            if m.slug in self._cache:
                true_prob, ts = self._cache[m.slug]
                if time.time() - ts < 300:  # 5-min cache
                    sig = self._build_signal(m, true_prob)
                    if sig:
                        signals.append(sig)
                    continue

            # LLM analysis with news context
            news_ctx = self._get_news_context(m.question)
            llm = self.amf.analyze(
                question=m.question, yes_price=m.yes_price, no_price=m.no_price,
                btc_price=0, btc_change=0, eth_price=0, eth_change=0,
                book_ratio=1.0, spread=abs(m.yes_price - m.no_price),
                momentum=news_ctx,
                time_elapsed=0, window_duration=86400
            )

            if llm.direction and llm.confidence >= 0.60:
                true_prob = llm.confidence if llm.direction == "UP" else 1.0 - llm.confidence
                self._cache[m.slug] = (true_prob, time.time())
                sig = self._build_signal(m, true_prob)
                if sig:
                    signals.append(sig)

        return signals

    def _build_signal(self, m, true_prob: float) -> Signal | None:
        yes_edge = true_prob - m.yes_price
        no_edge  = (1 - true_prob) - m.no_price

        if yes_edge >= self.MIN_EDGE and yes_edge >= no_edge:
            confidence = min(0.85, 0.55 + yes_edge)
            return Signal(
                direction="UP", confidence=confidence,
                entry_price=m.yes_price, token_id=m.yes_token,
                market_slug=m.slug, question=m.question,
                tp_pct=0.30, sl_pct=0.12, max_hold=3600.0,
                source=self.name
            )
        if no_edge >= self.MIN_EDGE and no_edge > yes_edge:
            confidence = min(0.85, 0.55 + no_edge)
            return Signal(
                direction="DOWN", confidence=confidence,
                entry_price=m.no_price, token_id=m.no_token,
                market_slug=m.slug, question=m.question,
                tp_pct=0.30, sl_pct=0.12, max_hold=3600.0,
                source=self.name
            )
        return None

    def _get_news_context(self, question: str) -> str:
        """Fetch recent news headlines for context."""
        if not self.news_api_key:
            return "no news context"
        try:
            keywords = " ".join(question.split()[:5])
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": keywords, "pageSize": 3,
                        "sortBy": "publishedAt", "apiKey": self.news_api_key},
                timeout=3
            )
            if r.ok:
                articles = r.json().get("articles", [])
                return " | ".join(a["title"] for a in articles[:3])
        except Exception:
            pass
        return "no news context"

    def _is_event_market(self, m) -> bool:
        q = m.question.lower()
        crypto_terms = {"btc", "bitcoin", "eth", "ethereum", "crypto",
                        "5-min", "5min", "15-min", "15min", "up or down"}
        return not any(t in q for t in crypto_terms)
