from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from .base import QueueBase, QueueItem


class SqsQueue(QueueBase):
    def __init__(self, queue_url: str, region: str | None = None) -> None:
        try:
            import boto3  # type: ignore
        except Exception as exc:
            raise RuntimeError("boto3 not installed") from exc
        self._client = boto3.client("sqs", region_name=region)
        self._queue_url = queue_url
        self._receipts: Dict[str, str] = {}

    def enqueue(self, job_id: str, payload: Dict[str, Any]) -> None:
        envelope = {
            "job_id": job_id,
            "payload": payload,
            "enqueued_at": time.time(),
            "attempts": 0,
        }
        self._client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=json.dumps(envelope),
        )

    def dequeue(self, worker_id: str, timeout_s: float = 1.0) -> Optional[QueueItem]:
        resp = self._client.receive_message(
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=int(timeout_s),
            VisibilityTimeout=30,
        )
        messages = resp.get("Messages", [])
        if not messages:
            return None
        msg = messages[0]
        envelope = json.loads(msg["Body"])
        self._receipts[envelope["job_id"]] = msg["ReceiptHandle"]
        return QueueItem(
            job_id=envelope["job_id"],
            payload=envelope["payload"],
            enqueued_at=envelope["enqueued_at"],
            attempts=envelope.get("attempts", 0),
        )

    def ack(self, job_id: str) -> None:
        receipt = self._receipts.pop(job_id, None)
        if receipt:
            self._client.delete_message(QueueUrl=self._queue_url, ReceiptHandle=receipt)

    def retry(self, job_id: str, delay_s: float) -> None:
        receipt = self._receipts.get(job_id)
        if receipt:
            self._client.change_message_visibility(
                QueueUrl=self._queue_url,
                ReceiptHandle=receipt,
                VisibilityTimeout=int(delay_s),
            )

    def stats(self) -> Dict[str, Any]:
        return {"QUEUED": 0, "queue_depth_hit": 0, "queue_depth_miss": 0}
