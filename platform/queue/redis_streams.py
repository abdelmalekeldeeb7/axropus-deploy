from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from .base import QueueBase, QueueItem


class RedisStreamsQueue(QueueBase):
    def __init__(
        self,
        url: str,
        stream_prefix: str = "korith:jobs",
        consumer_group: str = "korith",
        node_id: str = "",
    ) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:
            raise RuntimeError("redis library not installed") from exc
        self._redis = redis.Redis.from_url(url)
        self._prefix = stream_prefix
        self._group = consumer_group
        self._node_id = node_id.strip()
        self._receipts: Dict[str, tuple[str, str]] = {}
        self._payloads: Dict[str, Dict[str, Any]] = {}

    def _stream(self, node_id: str) -> str:
        node = node_id.strip() or "default"
        return f"{self._prefix}:{node}"

    def _ensure_group(self, stream: str) -> None:
        try:
            self._redis.xgroup_create(stream, self._group, id="$", mkstream=True)
        except Exception:
            pass

    def enqueue(self, job_id: str, payload: Dict[str, Any]) -> None:
        target_node_id = str(payload.get("target_node_id", "") or "")
        stream = self._stream(target_node_id)
        self._ensure_group(stream)
        envelope = {
            "job_id": job_id,
            "payload": json.dumps(payload, ensure_ascii=False),
            "enqueued_at": str(payload.get("enqueued_at") or time.time()),
            "attempts": str(payload.get("attempts") or 0),
        }
        self._redis.xadd(stream, envelope)

    def _decode_message(self, stream: str, message: tuple[Any, Dict[Any, Any]]) -> QueueItem:
        msg_id, data = message
        job_id = data.get(b"job_id", b"").decode("utf-8")
        payload = json.loads(data.get(b"payload", b"{}").decode("utf-8"))
        enqueued_at = float(data.get(b"enqueued_at", b"0").decode("utf-8"))
        attempts = int(data.get(b"attempts", b"0").decode("utf-8"))
        msg_id_s = msg_id.decode("utf-8") if isinstance(msg_id, (bytes, bytearray)) else str(msg_id)
        self._receipts[job_id] = (stream, msg_id_s)
        self._payloads[job_id] = payload
        return QueueItem(job_id=job_id, payload=payload, enqueued_at=enqueued_at, attempts=attempts)

    def dequeue(self, worker_id: str, timeout_s: float = 1.0) -> Optional[QueueItem]:
        streams = []
        if self._node_id:
            streams.append(self._stream(self._node_id))
        streams.append(self._stream("default"))
        for stream in streams:
            self._ensure_group(stream)

        for stream in streams:
            resp = self._redis.xreadgroup(
                groupname=self._group,
                consumername=worker_id,
                streams={stream: ">"},
                count=1,
                block=int(timeout_s * 1000),
            )
            if not resp:
                continue
            _, messages = resp[0]
            if not messages:
                continue
            return self._decode_message(stream, messages[0])
        return None

    def ack(self, job_id: str) -> None:
        receipt = self._receipts.pop(job_id, None)
        if receipt:
            stream, msg_id = receipt
            self._redis.xack(stream, self._group, msg_id)
        self._payloads.pop(job_id, None)

    def retry(self, job_id: str, delay_s: float) -> None:
        payload = self._payloads.pop(job_id, {})
        receipt = self._receipts.pop(job_id, None)
        if receipt:
            stream, msg_id = receipt
            self._redis.xack(stream, self._group, msg_id)
        payload = dict(payload)
        payload["attempts"] = int(payload.get("attempts", 0) or 0) + 1
        payload["enqueued_at"] = time.time() + max(0.0, float(delay_s))
        self.enqueue(job_id, payload)

    def stats(self) -> Dict[str, Any]:
        streams = []
        if self._node_id:
            streams.append(self._stream(self._node_id))
        streams.append(self._stream("default"))
        queued = 0
        for stream in streams:
            try:
                queued += int(self._redis.xlen(stream))
            except Exception:
                continue
        return {"QUEUED": queued, "queue_depth_hit": 0, "queue_depth_miss": queued}
