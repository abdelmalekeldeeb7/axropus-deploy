from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

from .base import QueueBase, QueueItem


class NatsQueue(QueueBase):
    def __init__(self, url: str, subject: str = "korith.queue") -> None:
        try:
            import nats  # type: ignore
        except Exception as exc:
            raise RuntimeError("nats-py library not installed") from exc
        self._nats = nats
        self._url = url
        self._subject = subject
        self._nc = None

    async def _connect(self):
        if self._nc is None:
            self._nc = await self._nats.connect(self._url)
        return self._nc

    def enqueue(self, job_id: str, payload: Dict[str, Any]) -> None:
        envelope = {
            "job_id": job_id,
            "payload": payload,
            "enqueued_at": time.time(),
            "attempts": 0,
        }
        async def _pub():
            nc = await self._connect()
            await nc.publish(self._subject, json.dumps(envelope).encode("utf-8"))
        asyncio.get_event_loop().run_until_complete(_pub())

    def dequeue(self, worker_id: str, timeout_s: float = 1.0) -> Optional[QueueItem]:
        async def _sub():
            nc = await self._connect()
            sub = await nc.subscribe(self._subject)
            try:
                msg = await sub.next_msg(timeout=timeout_s)
            except Exception:
                return None
            envelope = json.loads(msg.data.decode("utf-8"))
            return QueueItem(
                job_id=envelope["job_id"],
                payload=envelope["payload"],
                enqueued_at=envelope["enqueued_at"],
                attempts=envelope.get("attempts", 0),
            )
        return asyncio.get_event_loop().run_until_complete(_sub())

    def ack(self, job_id: str) -> None:
        return None

    def retry(self, job_id: str, delay_s: float) -> None:
        return None

    def stats(self) -> Dict[str, Any]:
        return {"QUEUED": 0, "queue_depth_hit": 0, "queue_depth_miss": 0}
