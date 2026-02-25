from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class RateLimitDecision:
    allowed: bool
    reason: str
    org_request_rate: float
    org_token_rate: float
    rpm_cap: int
    tpm_cap: int
    window_epoch: int


class SqliteRateLimiter:
    def __init__(self, db_path: Path, burst_factor: float = 2.0) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.burst_factor = max(1.0, float(burst_factor))
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limit_windows (
                  org_id TEXT NOT NULL,
                  window_epoch INTEGER NOT NULL,
                  request_count INTEGER NOT NULL,
                  token_count INTEGER NOT NULL,
                  PRIMARY KEY(org_id, window_epoch)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limit_org ON rate_limit_windows(org_id)")

    def check_and_consume(
        self,
        org_id: str,
        request_inc: int,
        token_inc: int,
        rpm_limit: int,
        tpm_limit: int,
    ) -> RateLimitDecision:
        now = int(time.time())
        window = now - (now % 60)
        rpm_cap = max(1, int(rpm_limit * self.burst_factor))
        tpm_cap = max(1, int(tpm_limit * self.burst_factor))

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM rate_limit_windows WHERE window_epoch < ?", (window - 120,))
                cur = conn.execute(
                    "SELECT request_count, token_count FROM rate_limit_windows WHERE org_id=? AND window_epoch=?",
                    (org_id, window),
                )
                row = cur.fetchone()
                req = int(row[0]) if row else 0
                tok = int(row[1]) if row else 0

                next_req = req + request_inc
                next_tok = tok + token_inc
                if next_req > rpm_cap:
                    return RateLimitDecision(False, "rpm_exceeded", float(req), float(tok), rpm_cap, tpm_cap, window)
                if next_tok > tpm_cap:
                    return RateLimitDecision(False, "tpm_exceeded", float(req), float(tok), rpm_cap, tpm_cap, window)

                conn.execute(
                    """
                    INSERT INTO rate_limit_windows(org_id, window_epoch, request_count, token_count)
                    VALUES(?,?,?,?)
                    ON CONFLICT(org_id, window_epoch) DO UPDATE SET
                        request_count=excluded.request_count,
                        token_count=excluded.token_count
                    """,
                    (org_id, window, next_req, next_tok),
                )

        return RateLimitDecision(True, "ok", float(next_req), float(next_tok), rpm_cap, tpm_cap, window)


class RedisRateLimiter:
    def __init__(self, url: str, burst_factor: float = 2.0) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:
            raise RuntimeError("redis client unavailable") from exc
        self.client = redis.from_url(url)
        self.burst_factor = max(1.0, float(burst_factor))

    def check_and_consume(
        self,
        org_id: str,
        request_inc: int,
        token_inc: int,
        rpm_limit: int,
        tpm_limit: int,
    ) -> RateLimitDecision:
        now = int(time.time())
        minute = now - (now % 60)
        req_key = f"korith:rl:req:{org_id}:{minute}"
        tok_key = f"korith:rl:tok:{org_id}:{minute}"
        pipe = self.client.pipeline()
        pipe.incrby(req_key, request_inc)
        pipe.expire(req_key, 180)
        pipe.incrby(tok_key, token_inc)
        pipe.expire(tok_key, 180)
        req_val, _, tok_val, _ = pipe.execute()

        rpm_cap = max(1, int(rpm_limit * self.burst_factor))
        tpm_cap = max(1, int(tpm_limit * self.burst_factor))
        if int(req_val) > rpm_cap:
            return RateLimitDecision(False, "rpm_exceeded", float(req_val), float(tok_val), rpm_cap, tpm_cap, minute)
        if int(tok_val) > tpm_cap:
            return RateLimitDecision(False, "tpm_exceeded", float(req_val), float(tok_val), rpm_cap, tpm_cap, minute)
        return RateLimitDecision(True, "ok", float(req_val), float(tok_val), rpm_cap, tpm_cap, minute)


def build_rate_limiter(
    backend: str,
    sqlite_path: Path,
    redis_url: Optional[str],
    burst_factor: float,
):
    if backend == "redis" and redis_url:
        try:
            return RedisRateLimiter(redis_url, burst_factor=burst_factor)
        except Exception:
            pass
    return SqliteRateLimiter(sqlite_path, burst_factor=burst_factor)
