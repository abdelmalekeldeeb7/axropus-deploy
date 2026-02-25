from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from .base import QueueBase, QueueItem


class RedisQueue(QueueBase):
    def __init__(self, url: str, namespace: str = "korith:queue") -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:
            raise RuntimeError("redis library not installed") from exc
        self._redis = redis.Redis.from_url(url)
        self._ns = namespace
        self._group = f"{namespace}:group"
        self._receipts: Dict[str, tuple[str, str]] = {}
        self._payloads: Dict[str, Dict[str, Any]] = {}

    def _stream(self, target: str) -> str:
        return f"{self._ns}:stream:{target}"

    def _ensure_group(self, stream: str) -> None:
        try:
            self._redis.xgroup_create(stream, self._group, id="$", mkstream=True)
        except Exception:
            pass

    def enqueue(self, job_id: str, payload: Dict[str, Any]) -> None:
        target = payload.get("target_worker_id", "any")
        stream = self._stream(target)
        self._ensure_group(stream)
        envelope = {
            "job_id": job_id,
            "payload": json.dumps(payload),
            "enqueued_at": str(time.time()),
            "attempts": "0",
        }
        self._redis.xadd(stream, envelope)

    def dequeue(self, worker_id: str, timeout_s: float = 1.0) -> Optional[QueueItem]:
        streams = [self._stream(worker_id), self._stream("any")]
        for stream in streams:
            self._ensure_group(stream)
        resp = self._redis.xreadgroup(
            groupname=self._group,
            consumername=worker_id,
            streams={streams[0]: ">"},
            count=1,
            block=int(timeout_s * 1000),
        )
        if not resp:
            resp = self._redis.xreadgroup(
                groupname=self._group,
                consumername=worker_id,
                streams={streams[1]: ">"},
                count=1,
                block=int(timeout_s * 1000),
            )
        if not resp:
            return None
        _, messages = resp[0]
        if not messages:
            return None
        msg_id, data = messages[0]
        job_id = data.get(b"job_id", b"").decode("utf-8")
        payload = json.loads(data.get(b"payload", b"{}").decode("utf-8"))
        enqueued_at = float(data.get(b"enqueued_at", b"0").decode("utf-8"))
        attempts = int(data.get(b"attempts", b"0").decode("utf-8"))
        # Track which stream the message came from.
        stream_name = resp[0][0].decode("utf-8") if isinstance(resp[0][0], (bytes, bytearray)) else resp[0][0]
        self._receipts[job_id] = (stream_name, msg_id.decode("utf-8"))
        self._payloads[job_id] = payload
        return QueueItem(job_id=job_id, payload=payload, enqueued_at=enqueued_at, attempts=attempts)

    def ack(self, job_id: str) -> None:
        receipt = self._receipts.pop(job_id, None)
        if receipt:
            stream, msg_id = receipt
            self._redis.xack(stream, self._group, msg_id)
        self._payloads.pop(job_id, None)

    def retry(self, job_id: str, delay_s: float) -> None:
        time.sleep(delay_s)
        # Requeue by creating a new stream entry; ack existing if present.
        receipt = self._receipts.pop(job_id, None)
        if receipt:
            stream, msg_id = receipt
            self._redis.xack(stream, self._group, msg_id)
        payload = self._payloads.pop(job_id, None) or {}
        self.enqueue(job_id, payload)

    def stats(self) -> Dict[str, Any]:
        any_len = 0
        hit_len = 0
        try:
            any_len = int(self._redis.xinfo_stream(self._stream("any")).get("length", 0))
        except Exception:
            any_len = 0
        # Worker-targeted entries are typically HIT lane.
        for key in self._redis.keys(f"{self._ns}:stream:*"):
            name = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
            if name.endswith(":any"):
                continue
            try:
                hit_len += int(self._redis.xinfo_stream(name).get("length", 0))
            except Exception:
                continue
        return {
            "QUEUED": any_len + hit_len,
            "queue_depth_hit": hit_len,
            "queue_depth_miss": any_len,
        }
