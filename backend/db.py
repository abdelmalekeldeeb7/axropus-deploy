from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

_engine = None
_SessionLocal = None


def _get_url() -> str:
    return os.getenv("AXROPUS_DATABASE_URL", "sqlite:////data/axropus.db")


def get_engine():
    global _engine
    if _engine is None:
        url = _get_url()
        sqlite_mode = url.startswith("sqlite")
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False} if sqlite_mode else {},
            future=True,
        )
        if sqlite_mode:
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
    return _engine


def SessionLocal():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False, future=True
        )
    return _SessionLocal()


def init_db() -> None:
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=get_engine())


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
