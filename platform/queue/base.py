from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class QueueItem:
    job_id: str
    payload: Dict[str, Any]
    enqueued_at: float
    attempts: int = 0


class QueueBase:
    def enqueue(self, job_id: str, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    def dequeue(self, worker_id: str, timeout_s: float = 1.0) -> Optional[QueueItem]:
        raise NotImplementedError

    def ack(self, job_id: str) -> None:
        raise NotImplementedError

    def retry(self, job_id: str, delay_s: float) -> None:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        return {}
