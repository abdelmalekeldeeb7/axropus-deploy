from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import os
import sqlite3
import time

from platform.queue.sqlite_queue import SQLiteQueue


class SQLiteQueueTest(unittest.TestCase):
    def test_target_worker_dequeue(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "queue.sqlite"
            q = SQLiteQueue(db_path=db_path)
            q.enqueue("job1", {"target_worker_id": "worker-a", "lane": "HIT"})
            q.enqueue("job2", {"target_worker_id": "worker-b", "lane": "MISS"})

            item_a = q.dequeue("worker-a", timeout_s=0.1)
            self.assertIsNotNone(item_a)
            self.assertEqual(item_a.job_id, "job1")

            item_b = q.dequeue("worker-b", timeout_s=0.1)
            self.assertIsNotNone(item_b)
            self.assertEqual(item_b.job_id, "job2")

    def test_reclaim_stale_inflight_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "queue.sqlite"
            q = SQLiteQueue(db_path=db_path)
            q.enqueue("job1", {"target_worker_id": "worker-a", "lane": "HIT"})
            item = q.dequeue("worker-a", timeout_s=0.1)
            self.assertIsNotNone(item)

            # Mark inflight as stale so a new queue instance can reclaim it.
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE queue SET available_at=? WHERE job_id=?",
                    (time.time() - 3600.0, "job1"),
                )
            os.environ["KORITH_QUEUE_RECLAIM_INFLIGHT"] = "1"
            os.environ["KORITH_QUEUE_INFLIGHT_STALE_S"] = "10"
            try:
                q2 = SQLiteQueue(db_path=db_path)
                reclaimed = q2.dequeue("worker-a", timeout_s=0.1)
                self.assertIsNotNone(reclaimed)
                self.assertEqual(reclaimed.job_id, "job1")
            finally:
                os.environ.pop("KORITH_QUEUE_RECLAIM_INFLIGHT", None)
                os.environ.pop("KORITH_QUEUE_INFLIGHT_STALE_S", None)
