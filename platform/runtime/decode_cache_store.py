from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional


class DecodeCacheStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decode_cache (
                  org_id TEXT NOT NULL,
                  backend_id TEXT NOT NULL,
                  fingerprint_hash TEXT NOT NULL,
                  prompt_hash TEXT NOT NULL,
                  sampling_hash TEXT NOT NULL,
                  output_text TEXT NOT NULL,
                  tokens_out INTEGER NOT NULL,
                  decode_ms REAL NOT NULL DEFAULT 0.0,
                  total_ms REAL NOT NULL DEFAULT 0.0,
                  updated_at REAL NOT NULL,
                  PRIMARY KEY (org_id, backend_id, fingerprint_hash, prompt_hash, sampling_hash)
                )
                """
            )

    def get(
        self,
        *,
        org_id: str,
        backend_id: str,
        fingerprint_hash: str,
        prompt_hash: str,
        sampling_hash: str,
    ) -> Optional[Dict[str, object]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT output_text, tokens_out, decode_ms, total_ms, updated_at
                FROM decode_cache
                WHERE org_id=? AND backend_id=? AND fingerprint_hash=? AND prompt_hash=? AND sampling_hash=?
                """,
                (org_id, backend_id, fingerprint_hash, prompt_hash, sampling_hash),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "output_text": str(row[0] or ""),
                "tokens_out": int(row[1] or 0),
                "decode_ms": float(row[2] or 0.0),
                "total_ms": float(row[3] or 0.0),
                "updated_at": float(row[4] or 0.0),
            }

    def set(
        self,
        *,
        org_id: str,
        backend_id: str,
        fingerprint_hash: str,
        prompt_hash: str,
        sampling_hash: str,
        output_text: str,
        tokens_out: int,
        decode_ms: float,
        total_ms: float,
        updated_at: float,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decode_cache(
                  org_id, backend_id, fingerprint_hash, prompt_hash, sampling_hash,
                  output_text, tokens_out, decode_ms, total_ms, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    org_id,
                    backend_id,
                    fingerprint_hash,
                    prompt_hash,
                    sampling_hash,
                    output_text,
                    int(tokens_out),
                    float(decode_ms),
                    float(total_ms),
                    float(updated_at),
                ),
            )
