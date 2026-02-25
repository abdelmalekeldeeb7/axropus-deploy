from __future__ import annotations

from pathlib import Path

from .store import SQLiteLedgerStore
from .postgres_store import PostgresLedgerStore


def migrate_sqlite(db_path: Path) -> None:
    store = SQLiteLedgerStore(db_path=db_path)
    store.init()


def migrate_postgres(dsn: str) -> None:
    store = PostgresLedgerStore(dsn=dsn)
    store.init()
