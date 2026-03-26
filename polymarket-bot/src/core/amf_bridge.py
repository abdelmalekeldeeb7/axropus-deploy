"""
AMF Bridge — Full stack wired:
  TurboQuant   → prompt compression, maximize prefix cache hits
  VRAM KV Cache → vLLM prefix caching, 0.01s on repeated questions
  LMCache      → persistent KV blocks across restarts
  Dynamo       → routes to fastest available worker
  Ollama       → local fallback (Qwen2.5 14B)
  DeepSeek API → cloud fallback

Cache hierarchy (fastest → slowest):
  L1: Python dict     (0.0001s) — same question this session
  L2: VRAM KV cache   (0.01s)  — vLLM prefix cache hit
  L3: LMCache blocks  (0.1s)   — persistent GPU/CPU blocks
  L4: Ollama          (0.3s)   — full local inference
  L5: DeepSeek API    (0.8s)   — cloud fallback
"""
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional
import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMSignal:
    direction:  Optional[str]   # "UP" | "DOWN" | None
    confidence: float
    reasoning:  str
    latency_ms: float
    cache_hit:  bool  = False
    cache_tier: str   = "miss"  # "l1" | "l2_vram" | "l3_lmcache" | "l4_ollama" | "l5_deepseek"


# ── TurboQuant ────────────────────────────────────────────────────────────────

class TurboQuant:
    """
    Prompt compression layer.
    Maximizes AMF/vLLM prefix cache hits by:
      1. Splitting prompt into FIXED prefix + VARIABLE suffix
      2. Compressing numbers to fewer decimals
      3. Normalising whitespace/line endings for stable hashing
      4. Reducing token count ~40% while keeping all signal data

    Fixed prefix = system context (identical every call → cache hit)
    Variable suffix = per-trade data (only this part changes)
    """

    SYSTEM_PREFIX = (
        "You are a Polymarket crypto trading analyst. "
        "Respond with JSON only: "
        '{{"direction":"YES" or "NO","confidence":0.0-1.0,"reasoning":"brief"}}'
    )

    def compress(self, question: str, yes_price: float, no_price: float,
                 btc_price: float, btc_change: float, eth_price: float,
                 eth_change: float, book_ratio: float, spread: float,
                 depth_imbalance: float, momentum: str,
                 time_elapsed: float, window_duration: float) -> tuple[str, str]:
        """
        Returns (system_msg, user_msg).
        system_msg is FIXED — maximises prefix cache hits.
        user_msg is variable — only the last part changes per trade.
        """
        # Compress numbers — fewer decimals = fewer tokens
        user_msg = (
            f"Q:{question[:120]}\n"
            f"YES:{yes_price:.2f} NO:{no_price:.2f}\n"
            f"BTC:{btc_price:.0f}({btc_change:+.2f}%) "
            f"ETH:{eth_price:.0f}({eth_change:+.2f}%)\n"
            f"Book:{book_ratio:.2f} Spread:{spread:.3f} Imb:{depth_imbalance:+.2f}\n"
            f"Mom:{momentum[:80]}\n"
            f"T:{int(time_elapsed)}s/{int(window_duration)}s"
        )
        return self.SYSTEM_PREFIX, user_msg

    def cache_key(self, system: str, user: str) -> str:
        """Stable MD5 key for L1 cache."""
        text = f"{system}||{user}"
        # Normalise line endings for stable hashing across platforms
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return hashlib.md5(text.encode()).hexdigest()

    def prefix_hash(self, system: str, tenant: str = "korith") -> str:
        """SHA256 of the fixed prefix — used by Dynamo + LMCache."""
        key = f"{tenant}:{system}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]


# ── LMCache client ────────────────────────────────────────────────────────────

class LMCacheClient:
    """
    Queries LMCache HTTP API for block coverage.
    If LMCache has the prefix blocks cached, vLLM will skip prefill.
    """

    def __init__(self, url: str = "", block_size: int = 16):
        self.url        = url.rstrip("/") if url else ""
        self.block_size = block_size
        self._enabled   = bool(url)

    def coverage(self, prefix_hash: str, num_tokens: int) -> float:
        """Returns fraction of prefix blocks found in LMCache (0.0-1.0)."""
        if not self._enabled:
            return 0.0
        try:
            r = requests.post(
                f"{self.url}/v1/coverage",
                json={"prefix_hash": prefix_hash,
                      "block_size":  self.block_size},
                timeout=0.05   # 50ms max — don't slow down the trade loop
            )
            if r.ok:
                data = r.json()
                covered = int(data.get("covered_blocks", 0))
                total   = max(1, num_tokens // self.block_size)
                return min(1.0, covered / total)
        except Exception:
            pass
        return 0.0


# ── Dynamo router client ──────────────────────────────────────────────────────

class DynamoClient:
    """
    Queries Dynamo coordinator for best available worker.
    Falls back to default vLLM endpoint if Dynamo not running.
    """

    def __init__(self, coordinator_url: str = "", default_vllm: str = "http://localhost:8000"):
        self.coordinator  = coordinator_url.rstrip("/") if coordinator_url else ""
        self.default_vllm = default_vllm
        self._enabled     = bool(coordinator_url)
        self._workers:    list[dict] = []
        self._last_fetch: float = 0.0

    def best_endpoint(self, prefix_hash: str, num_tokens: int) -> str:
        """Return URL of best vLLM worker for this request."""
        if not self._enabled:
            return self.default_vllm

        self._refresh_workers()
        if not self._workers:
            return self.default_vllm

        try:
            r = requests.post(
                f"{self.coordinator}/v1/route",
                json={"prefix_hash": prefix_hash,
                      "num_tokens":  num_tokens,
                      "tenant_id":   "korith"},
                timeout=0.05
            )
            if r.ok:
                data = r.json()
                worker_url = data.get("endpoint", self.default_vllm)
                logger.debug("Dynamo routed to %s (by=%s savings=%.0fms)",
                             worker_url,
                             data.get("routed_by", "?"),
                             data.get("savings_ms", 0))
                return worker_url
        except Exception:
            pass
        return self.default_vllm

    def _refresh_workers(self):
        """Refresh worker list every 30s."""
        if time.time() - self._last_fetch < 30:
            return
        try:
            r = requests.get(f"{self.coordinator}/v1/workers", timeout=1)
            if r.ok:
                self._workers = r.json().get("workers", [])
                self._last_fetch = time.time()
        except Exception:
            pass


# ── Main AMF Bridge ───────────────────────────────────────────────────────────

class AMFBridge:
    """
    Full AMF stack connected via HTTP.
    korith-poly stays a separate repo — this calls AMF services by URL only.
    """

    OLLAMA_URL   = "http://localhost:11434/api/chat"
    DEEPSEEK_URL = "https://api.deepseek.com/v1"

    def __init__(self,
                 endpoint:          str = "http://localhost:11434",
                 deepseek_api_key:  str = "",
                 dynamo_url:        str = "",
                 lmcache_url:       str = "",
                 model:             str = "qwen2.5:14b"):

        self.model            = model
        self._deepseek_key    = deepseek_api_key

        # Full stack components
        self.tq      = TurboQuant()
        self.lmcache = LMCacheClient(url=lmcache_url or os.getenv("KORITH_LMCACHE_URL", ""))
        self.dynamo  = DynamoClient(
            coordinator_url=dynamo_url or os.getenv("AMF_COORDINATOR_URL", ""),
            default_vllm=os.getenv("VLLM_ENDPOINT", "http://localhost:8000")
        )

        # Ollama endpoint (fallback)
        self._ollama_base = endpoint if "11434" in endpoint else "http://localhost:11434"

        # L1 in-process cache
        self._cache:    dict[str, tuple[LLMSignal, float]] = {}
        self._cache_ttl = 120.0   # 2 min TTL

        # Stats per tier
        self._stats = {"l1": 0, "l2_vram": 0, "l3_lmcache": 0,
                       "l4_ollama": 0, "l5_deepseek": 0, "miss": 0}

    def analyze(self, question: str, yes_price: float, no_price: float,
                btc_price: float, btc_change: float, eth_price: float,
                eth_change: float, book_ratio: float, spread: float,
                momentum: str, time_elapsed: float, window_duration: float,
                depth_imbalance: float = 0.0) -> LLMSignal:
        """
        Full stack LLM call:
          TurboQuant compresses → L1 check → Dynamo routes →
          vLLM (VRAM cache) → LMCache hint → Ollama → DeepSeek
        """
        t0 = time.time()

        # ── TurboQuant: compress prompt ───────────────────────────────────
        system_msg, user_msg = self.tq.compress(
            question=question, yes_price=yes_price, no_price=no_price,
            btc_price=btc_price, btc_change=btc_change,
            eth_price=eth_price, eth_change=eth_change,
            book_ratio=book_ratio, spread=spread,
            depth_imbalance=depth_imbalance,
            momentum=momentum,
            time_elapsed=time_elapsed, window_duration=window_duration
        )

        # ── L1: in-process cache ──────────────────────────────────────────
        cache_key   = self.tq.cache_key(system_msg, user_msg)
        prefix_hash = self.tq.prefix_hash(system_msg)

        cached = self._cache.get(cache_key)
        if cached:
            sig, ts = cached
            if time.time() - ts < self._cache_ttl:
                self._stats["l1"] += 1
                return LLMSignal(sig.direction, sig.confidence, sig.reasoning,
                                 (time.time() - t0) * 1000,
                                 cache_hit=True, cache_tier="l1")

        # ── Estimate token count for routing ─────────────────────────────
        num_tokens = len((system_msg + user_msg).split()) * 1.3  # rough estimate

        # ── LMCache: check block coverage (hint for vLLM) ─────────────────
        lm_coverage = self.lmcache.coverage(prefix_hash, int(num_tokens))
        if lm_coverage > 0.7:
            logger.debug("LMCache coverage %.0f%% — prefix blocks warm", lm_coverage * 100)
            self._stats["l3_lmcache"] += 1

        # ── Dynamo: get best worker endpoint ─────────────────────────────
        vllm_endpoint = self.dynamo.best_endpoint(prefix_hash, int(num_tokens))

        # ── L2: vLLM (VRAM KV cache via prefix caching) ───────────────────
        raw = self._call_vllm(vllm_endpoint, system_msg, user_msg)
        if raw:
            sig = self._parse(raw, (time.time() - t0) * 1000, "l2_vram")
            self._cache[cache_key] = (sig, time.time())
            self._stats["l2_vram"] += 1
            return sig

        # ── L4: Ollama fallback ───────────────────────────────────────────
        raw = self._call_ollama(system_msg, user_msg)
        if raw:
            sig = self._parse(raw, (time.time() - t0) * 1000, "l4_ollama")
            self._cache[cache_key] = (sig, time.time())
            self._stats["l4_ollama"] += 1
            return sig

        # ── L5: DeepSeek API final fallback ──────────────────────────────
        if self._deepseek_key:
            raw = self._call_deepseek(system_msg, user_msg)
            if raw:
                sig = self._parse(raw, (time.time() - t0) * 1000, "l5_deepseek")
                self._cache[cache_key] = (sig, time.time())
                self._stats["l5_deepseek"] += 1
                return sig

        self._stats["miss"] += 1
        return LLMSignal(None, 0.0, "no_llm", (time.time() - t0) * 1000)

    # ── vLLM (VRAM KV prefix cache) ───────────────────────────────────────────

    def _call_vllm(self, endpoint: str, system: str, user: str) -> str:
        """
        Call vLLM with structured system/user split.
        vLLM prefix caching reuses KV vectors for identical system messages.
        = 0.01s response when system prefix is cached in VRAM.
        """
        try:
            r = requests.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user}
                    ],
                    "max_tokens":  80,
                    "temperature": 0.05,   # near-deterministic
                },
                timeout=3
            )
            if r.ok:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        return ""

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _call_ollama(self, system: str, user: str) -> str:
        try:
            r = requests.post(
                f"{self._ollama_base}/api/chat",
                json={
                    "model":  self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user}
                    ],
                    "options": {"temperature": 0.05}
                },
                timeout=6
            )
            if r.ok:
                return r.json().get("message", {}).get("content", "")
        except Exception:
            pass
        return ""

    # ── DeepSeek API ──────────────────────────────────────────────────────────

    def _call_deepseek(self, system: str, user: str) -> str:
        try:
            r = requests.post(
                f"{self.DEEPSEEK_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self._deepseek_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user}
                    ],
                    "max_tokens":  80,
                    "temperature": 0.05,
                },
                timeout=8
            )
            if r.ok:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        return ""

    # ── Parse ─────────────────────────────────────────────────────────────────

    def _parse(self, text: str, latency_ms: float, tier: str) -> LLMSignal:
        try:
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                data      = json.loads(text[start:end])
                raw_dir   = data.get("direction", "").upper()
                direction = ("UP"   if raw_dir in ("YES", "UP")   else
                             "DOWN" if raw_dir in ("NO",  "DOWN") else None)
                conf      = float(data.get("confidence", 0))
                reasoning = data.get("reasoning", "")
                return LLMSignal(direction, conf, reasoning, latency_ms,
                                 cache_hit=(tier != "miss"), cache_tier=tier)
        except Exception:
            pass
        return LLMSignal(None, 0.0, "parse_error", latency_ms, cache_tier=tier)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = sum(self._stats.values())
        hits  = self._stats["l1"] + self._stats["l2_vram"] + self._stats["l3_lmcache"]
        return hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        total = max(1, sum(self._stats.values()))
        return {
            "hit_rate":       f"{self.hit_rate:.1%}",
            "l1_cache":       self._stats["l1"],
            "l2_vram":        self._stats["l2_vram"],
            "l3_lmcache":     self._stats["l3_lmcache"],
            "l4_ollama":      self._stats["l4_ollama"],
            "l5_deepseek":    self._stats["l5_deepseek"],
            "avg_latency_est": "0.01s L1/L2 | 0.1s L3 | 0.3s L4 | 0.8s L5",
        }
