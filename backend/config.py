from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    database_url: str
    jwt_secret: str
    jwt_algorithm: str
    jwt_expire_minutes: int
    trial_key_days: int
    max_keys_per_customer: int
    cors_allow_origins: list[str]
    login_rate_limit_per_minute: int
    signup_rate_limit_per_hour: int


def _csv_env(name: str, default: str) -> list[str]:
    raw = str(os.getenv(name, default)).strip()
    if not raw:
        return ["*"]
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        database_url=str(os.getenv("AXROPUS_DATABASE_URL", "sqlite:////data/axropus.db")),
        jwt_secret=str(os.getenv("AXROPUS_JWT_SECRET", "change-me-in-production")),
        jwt_algorithm=str(os.getenv("AXROPUS_JWT_ALGORITHM", "HS256")),
        jwt_expire_minutes=int(str(os.getenv("AXROPUS_JWT_EXPIRE_MINUTES", "1440"))),
        trial_key_days=int(str(os.getenv("AXROPUS_TRIAL_KEY_DAYS", "7"))),
        max_keys_per_customer=int(str(os.getenv("AXROPUS_MAX_KEYS_PER_CUSTOMER", "5"))),
        cors_allow_origins=_csv_env("AXROPUS_CORS_ALLOW_ORIGINS", "*"),
        login_rate_limit_per_minute=int(str(os.getenv("AXROPUS_LOGIN_RATE_LIMIT_PER_MINUTE", "20"))),
        signup_rate_limit_per_hour=int(str(os.getenv("AXROPUS_SIGNUP_RATE_LIMIT_PER_HOUR", "10"))),
    )
