from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.runtime.cluster_worker import ClusterWorker
from platform.runtime.decode_cache_store import DecodeCacheStore


class DecodeCacheStoreTests(unittest.TestCase):
    def test_decode_cache_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            store = DecodeCacheStore(Path(td) / "decode_cache.sqlite")
            key = {
                "org_id": "default",
                "backend_id": "vllm",
                "fingerprint_hash": "fp",
                "prompt_hash": "ph",
                "sampling_hash": "sh",
            }
            store.set(
                **key,
                output_text="hello world",
                tokens_out=2,
                decode_ms=12.5,
                total_ms=15.0,
                updated_at=1.0,
            )
            row = store.get(**key)
            self.assertIsNotNone(row)
            self.assertEqual(str(row["output_text"]), "hello world")
            self.assertEqual(int(row["tokens_out"]), 2)
            self.assertEqual(float(row["decode_ms"]), 12.5)

    def test_decode_cache_eligibility_requires_deterministic_defaults(self):
        worker = ClusterWorker.__new__(ClusterWorker)
        worker._decode_cache_enabled = True
        worker._decode_cache_require_deterministic = True
        ok_job = {"deterministic_cfg": {"seed": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 0}}
        bad_job = {"deterministic_cfg": {"seed": 1, "temperature": 0.7, "top_p": 1.0, "top_k": 0}}
        self.assertTrue(ClusterWorker._decode_cache_eligible(worker, ok_job))
        self.assertFalse(ClusterWorker._decode_cache_eligible(worker, bad_job))


if __name__ == "__main__":
    unittest.main()
