from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


class RestoreStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS restore_requests (
                  fingerprint_hash TEXT PRIMARY KEY,
                  snapshot_path TEXT NOT NULL,
                  created_at REAL NOT NULL
                )
                """
            )

    def set(self, fingerprint_hash: str, snapshot_path: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO restore_requests(fingerprint_hash, snapshot_path, created_at) VALUES(?,?,strftime('%s','now'))",
                (fingerprint_hash, snapshot_path),
            )

    def get(self, fingerprint_hash: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT snapshot_path FROM restore_requests WHERE fingerprint_hash=?",
                (fingerprint_hash,),
            )
            row = cur.fetchone()
            return row[0] if row else None
