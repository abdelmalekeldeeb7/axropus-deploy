import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.artifacts.store import ArtifactStore
from platform.gateway.auth import authenticate, hash_api_key, issue_api_key
from platform.gateway.rate_limit import SqliteRateLimiter
from platform.ledger.store import SQLiteLedgerStore


class PlatformControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.db = self.base / "ledger.sqlite"
        self.store = SQLiteLedgerStore(self.db)
        self.store.init()
        self._prev_salt = os.environ.get("KORITH_API_KEY_SALT")
        os.environ["KORITH_API_KEY_SALT"] = "test_salt_for_platform_controls"

    def tearDown(self):
        if self._prev_salt is None:
            os.environ.pop("KORITH_API_KEY_SALT", None)
        else:
            os.environ["KORITH_API_KEY_SALT"] = self._prev_salt
        self.tmp.cleanup()

    def test_api_key_auth_and_revoke(self):
        key_id = "key-a"
        raw = issue_api_key(key_id)
        self.store.create_api_key(
            key_id=key_id,
            key_hash=hash_api_key(raw),
            org_id="pilot_a",
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            rate_limit_tpm=100,
            rate_limit_rpm=10,
            permissions_json=json.dumps({"read": True}),
        )

        ctx = authenticate({"Authorization": f"Bearer {raw}"}, self.store)
        self.assertEqual(ctx.org_id, "pilot_a")

        self.store.revoke_api_key(key_id, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        with self.assertRaises(Exception):
            authenticate({"Authorization": f"Bearer {raw}"}, self.store)

    def test_sqlite_rate_limit_blocks(self):
        rl = SqliteRateLimiter(self.base / "limits.sqlite", burst_factor=1.0)
        allowed = rl.check_and_consume("org-a", 1, 50, rpm_limit=2, tpm_limit=100)
        self.assertTrue(allowed.allowed)
        allowed = rl.check_and_consume("org-a", 1, 40, rpm_limit=2, tpm_limit=100)
        self.assertTrue(allowed.allowed)
        blocked = rl.check_and_consume("org-a", 1, 20, rpm_limit=2, tpm_limit=100)
        self.assertFalse(blocked.allowed)

    def test_artifact_org_partition(self):
        artifacts = ArtifactStore(self.base / "artifacts")
        paths = artifacts.init_job("job-1", org_id="pilot_a")
        self.assertIn("pilot_a", str(paths["job_dir"]))

    def test_replay_governance_persistence(self):
        fp = "fp-123"
        self.store.upsert_replay_governance(
            fingerprint_hash=fp,
            replay_disabled=1,
            disabled_reason="corruption_detected",
            disabled_at="2026-01-01T00:00:00Z",
            cooldown_until=0.0,
            negative_roi_streak=0,
            corruption_detected=1,
            restore_guard_disabled=0,
            updated_at="2026-01-01T00:00:00Z",
        )
        row = self.store.get_replay_governance(fp)
        self.assertIsNotNone(row)
        self.assertEqual(row["disabled_reason"], "corruption_detected")


if __name__ == "__main__":
    unittest.main()
