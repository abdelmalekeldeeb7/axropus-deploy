from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ledger.store import LedgerStore


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SnapshotIndex:
    def __init__(
        self,
        ledger: LedgerStore,
        redis_url: str = "",
        key_prefix: str = "korith:snapshot_index",
    ) -> None:
        self.ledger = ledger
        self._redis = None
        self._key_prefix = key_prefix
        if redis_url:
            try:
                import redis  # type: ignore

                self._redis = redis.Redis.from_url(redis_url)
                self._redis.ping()
            except Exception:
                self._redis = None

    def _tier_roots(self) -> Dict[str, Path]:
        roots: Dict[str, Path] = {}
        for tier, env_name in (
            ("vram", "KORITH_SNAPSHOT_VRAM_DIR"),
            ("ram", "KORITH_SNAPSHOT_RAM_DIR"),
            ("nvme", "KORITH_SNAPSHOT_NVME_DIR"),
        ):
            raw = str(os.environ.get(env_name, "") or "").strip()
            if not raw:
                continue
            try:
                roots[tier] = Path(raw).expanduser().resolve()
            except Exception:
                continue
        return roots

    def _detect_storage_tier(self, snapshot_path: str) -> str:
        text = str(snapshot_path or "").strip()
        if not text:
            return "unknown"
        try:
            resolved = Path(text).expanduser().resolve()
        except Exception:
            resolved = Path(text)
        roots = self._tier_roots()
        for tier in ("vram", "ram", "nvme"):
            root = roots.get(tier)
            if root is None:
                continue
            try:
                if resolved.is_relative_to(root):
                    return tier
            except Exception:
                pass
        lowered = text.lower()
        if "/vram/" in lowered:
            return "vram"
        if "/ram/" in lowered:
            return "ram"
        if "/nvme/" in lowered:
            return "nvme"
        return "unknown"

    def _tier_rank(self, tier: str) -> int:
        if tier == "vram":
            return 0
        if tier == "ram":
            return 1
        if tier == "nvme":
            return 2
        return 3

    def _sort_locations(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for row in rows:
            out = dict(row)
            tier = self._detect_storage_tier(str(out.get("snapshot_path", "")))
            out["storage_tier"] = tier
            enriched.append(out)

        def _used_ts(row: Dict[str, Any]) -> float:
            raw = str(row.get("last_used_at", "") or "")
            if not raw:
                return 0.0
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0

        def _sort_key(row: Dict[str, Any]) -> tuple[int, float]:
            tier = str(row.get("storage_tier", "unknown"))
            return (self._tier_rank(tier), -_used_ts(row))

        return sorted(enriched, key=_sort_key)

    def _redis_key(self, org_id: str, fingerprint_hash: str) -> str:
        return f"{self._key_prefix}:{org_id}:{fingerprint_hash}"

    def upsert_location(
        self,
        fingerprint_hash: str,
        snapshot_id: str,
        org_id: str,
        node_id: str,
        worker_id: Optional[str],
        snapshot_path: str,
        size_bytes: int,
        created_at: Optional[str] = None,
        last_used_at: Optional[str] = None,
    ) -> None:
        created_at = created_at or utc_now()
        last_used_at = last_used_at or created_at
        self.ledger.upsert_snapshot_location(
            fingerprint_hash=fingerprint_hash,
            snapshot_id=snapshot_id,
            org_id=org_id,
            node_id=node_id,
            worker_id=worker_id,
            snapshot_path=snapshot_path,
            size_bytes=int(size_bytes),
            created_at=created_at,
            last_used_at=last_used_at,
        )
        if self._redis is not None:
            payload = {
                "fingerprint_hash": fingerprint_hash,
                "snapshot_id": snapshot_id,
                "org_id": org_id,
                "node_id": node_id,
                "worker_id": worker_id,
                "snapshot_path": snapshot_path,
                "size_bytes": int(size_bytes),
                "created_at": created_at,
                "last_used_at": last_used_at,
            }
            score = float(time.time())
            self._redis.zadd(
                self._redis_key(org_id, fingerprint_hash),
                {json.dumps(payload, ensure_ascii=False): score},
            )

    def get_locations(self, fingerprint_hash: str, org_id: str) -> List[Dict[str, Any]]:
        if self._redis is not None:
            key = self._redis_key(org_id, fingerprint_hash)
            try:
                members = self._redis.zrevrange(key, 0, -1)
                rows: List[Dict[str, Any]] = []
                for raw in members:
                    payload = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw))
                    rows.append(payload)
                if rows:
                    return self._sort_locations(rows)
            except Exception:
                pass
        rows = self.ledger.list_snapshot_locations(fingerprint_hash=fingerprint_hash, org_id=org_id)
        return self._sort_locations(rows)

    def mark_used(self, fingerprint_hash: str, org_id: str, node_id: str, snapshot_id: str) -> None:
        now_str = utc_now()
        self.ledger.mark_snapshot_location_used(
            fingerprint_hash=fingerprint_hash,
            org_id=org_id,
            node_id=node_id,
            snapshot_id=snapshot_id,
            last_used_at=now_str,
        )
        if self._redis is not None:
            # Rebuild one key from ledger view for correctness.
            rows = self.ledger.list_snapshot_locations(fingerprint_hash=fingerprint_hash, org_id=org_id)
            key = self._redis_key(org_id, fingerprint_hash)
            self._redis.delete(key)
            now = time.time()
            for idx, row in enumerate(rows):
                self._redis.zadd(
                    key,
                    {json.dumps(row, ensure_ascii=False): now - idx},
                )

    def prune_old(self, max_entries_per_fingerprint: int = 8) -> int:
        return self.ledger.prune_snapshot_locations(max_entries_per_fingerprint=max_entries_per_fingerprint)
