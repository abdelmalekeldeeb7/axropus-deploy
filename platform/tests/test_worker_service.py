from __future__ import annotations

import json
import threading
import unittest
from urllib.request import urlopen

from platform.runtime.worker_service import WorkerHandler, ThreadingHTTPServer, _amf_health_payload


class _FakeThread:
    def is_alive(self) -> bool:
        return True


class _FakeWorker:
    def __init__(self) -> None:
        self._thread = _FakeThread()

    def amf_health(self) -> dict:
        return {
            "cache_entries": 12,
            "hit_rate": 0.75,
            "warm_ratio": 0.9,
            "ready": True,
        }


class WorkerServiceTests(unittest.TestCase):
    def test_amf_health_payload_defaults_without_worker(self):
        payload = _amf_health_payload(None)
        self.assertEqual(payload["cache_entries"], 0)
        self.assertEqual(payload["hit_rate"], 0.0)
        self.assertEqual(payload["warm_ratio"], 0.0)
        self.assertFalse(payload["ready"])

    def test_health_amf_endpoint_returns_worker_payload(self):
        WorkerHandler.worker = _FakeWorker()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), WorkerHandler)
        host, port = httpd.server_address
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://{host}:{port}/health/amf", timeout=2.0) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(payload["cache_entries"], 12)
            self.assertAlmostEqual(float(payload["hit_rate"]), 0.75, places=6)
            self.assertAlmostEqual(float(payload["warm_ratio"]), 0.9, places=6)
            self.assertTrue(bool(payload["ready"]))
        finally:
            httpd.shutdown()
            httpd.server_close()

