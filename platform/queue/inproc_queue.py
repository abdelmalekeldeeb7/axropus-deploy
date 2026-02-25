from __future__ import annotations

import queue
import time
from typing import Any, Dict, Optional

from .base import QueueBase, QueueItem


class InProcQueue(QueueBase):
    def __init__(self) -> None:
        self._q_any: queue.Queue = queue.Queue()
        self._by_worker: Dict[str, queue.Queue] = {}
        self._lane_counts: Dict[str, int] = {}

    def enqueue(self, job_id: str, payload: Dict[str, Any]) -> None:
        target = payload.get("target_worker_id")
        lane = str(payload.get("lane", "MISS")).upper()
        self._lane_counts[lane] = self._lane_counts.get(lane, 0) + 1
        if target:
            if target not in self._by_worker:
                self._by_worker[target] = queue.Queue()
            self._by_worker[target].put((job_id, payload, time.time(), 0))
        else:
            self._q_any.put((job_id, payload, time.time(), 0))

    def dequeue(self, worker_id: str, timeout_s: float = 1.0) -> Optional[QueueItem]:
        payload = None
        try:
            if worker_id in self._by_worker and not self._by_worker[worker_id].empty():
                job_id, payload, enqueued_at, attempts = self._by_worker[worker_id].get(timeout=timeout_s)
            else:
                job_id, payload, enqueued_at, attempts = self._q_any.get(timeout=timeout_s)
        except queue.Empty:
            return None
        lane = str((payload or {}).get("lane", "MISS")).upper()
        if lane in self._lane_counts and self._lane_counts[lane] > 0:
            self._lane_counts[lane] -= 1
        return QueueItem(job_id=job_id, payload=payload, enqueued_at=enqueued_at, attempts=attempts)

    def ack(self, job_id: str) -> None:
        return

    def retry(self, job_id: str, delay_s: float) -> None:
        # In-proc queue: re-enqueue after delay is best-effort.
        time.sleep(delay_s)

    def stats(self) -> Dict[str, Any]:
        depth_hit = int(self._lane_counts.get("HIT", 0)) + int(self._lane_counts.get("SPEC_HIT", 0))
        depth_miss = int(self._lane_counts.get("MISS", 0)) + int(self._lane_counts.get("SPEC_MISS", 0))
        return {
            "QUEUED": depth_hit + depth_miss,
            "queue_depth_hit": depth_hit,
            "queue_depth_miss": depth_miss,
        }
