"""AMF bridge — connects trading signals to local 8B model with KV cache."""
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
import requests

logger = logging.getLogger(__name__)

@dataclass
class LLMSignal:
    direction: Optional[str]   # "UP" | "DOWN" | None
    confidence: float          # 0.0 - 1.0
    reasoning: str
    latency_ms: float
    cache_hit: bool = False


class AMFBridge:
    """
    Routes LLM inference through local vLLM + AMF KV cache.
    Falls back to Ollama if vLLM not running.
    Falls back to DeepSeek API if neither available.
    """

    VLLM_URL   = "http://localhost:8000/v1"
    OLLAMA_URL = "http://localhost:11434/api/generate"
    DEEPSEEK_URL = "https://api.deepseek.com/v1"

    # Prompt template — context history maximizes AMF hit rate
    PROMPT_TEMPLATE = """You are a Polymarket crypto trading analyst.

Market: {question}
Current YES price: ${yes_price:.3f} | NO price: ${no_price:.3f}
BTC price: ${btc_price:.2f} | 1-min change: {btc_change:+.3f}%
ETH price: ${eth_price:.2f} | 1-min change: {eth_change:+.3f}%
Order book: bid/ask ratio = {book_ratio:.2f} | spread = ${spread:.3f}
Momentum signals: {momentum}
Time into window: {time_elapsed}s of {window_duration}s

Based on this data, will the market resolve YES or NO?

Respond with JSON only: {{"direction": "YES" or "NO", "confidence": 0.0-1.0, "reasoning": "brief"}}"""

    def __init__(self, mode: str = "ollama", api_key: str = "",
                 model: str = "llama3.1:8b"):
        self.mode = mode
        self.api_key = api_key
        self.model = model
        self._cache: dict[str, tuple[LLMSignal, float]] = {}
        self._cache_ttl = 120.0  # 2 min cache (AMF handles deeper caching)
        self._hits = 0
        self._misses = 0

    def analyze(self, question: str, yes_price: float, no_price: float,
                btc_price: float, btc_change: float, eth_price: float,
                eth_change: float, book_ratio: float, spread: float,
                momentum: str, time_elapsed: float,
                window_duration: float) -> LLMSignal:
        """Get LLM signal with AMF KV cache acceleration."""

        prompt = self.PROMPT_TEMPLATE.format(
            question=question, yes_price=yes_price, no_price=no_price,
            btc_price=btc_price, btc_change=btc_change,
            eth_price=eth_price, eth_change=eth_change,
            book_ratio=book_ratio, spread=spread,
            momentum=momentum, time_elapsed=int(time_elapsed),
            window_duration=int(window_duration)
        )

        # Check local cache first (fast path before even hitting AMF)
        cache_key = hashlib.md5(prompt[:200].encode()).hexdigest()
        if cache_key in self._cache:
            sig, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                self._hits += 1
                return LLMSignal(sig.direction, sig.confidence,
                                 sig.reasoning, 0.5, cache_hit=True)

        self._misses += 1
        t0 = time.time()

        result = self._call_llm(prompt)
        latency_ms = (time.time() - t0) * 1000

        signal = self._parse_response(result, latency_ms)
        self._cache[cache_key] = (signal, time.time())
        return signal

    def _call_llm(self, prompt: str) -> str:
        """Try vLLM → Ollama → DeepSeek API in order."""
        # 1. Try local vLLM (fastest with AMF KV cache)
        try:
            r = requests.post(
                f"{self.VLLM_URL}/chat/completions",
                json={"model": self.model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 100, "temperature": 0.1},
                timeout=3
            )
            if r.ok:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

        # 2. Try local Ollama
        try:
            r = requests.post(
                self.OLLAMA_URL,
                json={"model": self.model, "prompt": prompt,
                      "stream": False, "options": {"temperature": 0.1}},
                timeout=5
            )
            if r.ok:
                return r.json().get("response", "")
        except Exception:
            pass

        # 3. Fall back to DeepSeek API
        if self.api_key:
            try:
                r = requests.post(
                    f"{self.DEEPSEEK_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": "deepseek-chat",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 100, "temperature": 0.1},
                    timeout=8
                )
                if r.ok:
                    return r.json()["choices"][0]["message"]["content"]
            except Exception:
                pass

        return ""

    def _parse_response(self, text: str, latency_ms: float) -> LLMSignal:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                raw_dir = data.get("direction", "").upper()
                direction = "UP" if raw_dir in ("YES", "UP") else "DOWN" if raw_dir in ("NO", "DOWN") else None
                conf = float(data.get("confidence", 0))
                reasoning = data.get("reasoning", "")
                return LLMSignal(direction, conf, reasoning, latency_ms)
        except Exception:
            pass
        return LLMSignal(None, 0.0, "parse_error", latency_ms)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": f"{self.hit_rate:.1%}"}
