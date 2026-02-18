from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import HTTPException, status


class InMemoryRateLimiter:
    """Simple per-key sliding-window limiter."""

    def __init__(self) -> None:
        self._events: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            q = self._events[key]
            while q and q[0] < cutoff:
                q.popleft()

            if len(q) >= limit:
                return False

            q.append(now)
            return True


rate_limiter = InMemoryRateLimiter()


def enforce_rate_limit(key: str, *, limit: int, window_seconds: int) -> None:
    if not rate_limiter.allow(key=key, limit=limit, window_seconds=window_seconds):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
