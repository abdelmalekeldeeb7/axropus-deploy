from __future__ import annotations

import concurrent.futures
import json
import threading
import unittest
from urllib.request import Request, urlopen

from platform.runtime.amf_coordinator import CoordinatorHandler, ThreadingHTTPServer


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=3.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


class AmfCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reset global index for deterministic test behavior.
        CoordinatorHandler.index = CoordinatorHandler.index.__class__()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), CoordinatorHandler)
        self.host, self.port = self.httpd.server_address
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def test_register_lookup_and_evict(self):
        register = _request_json(
            "POST",
            self._url("/register"),
            {
                "hash": "abcd1234",
                "tenant_id": "tenant-a",
                "node_id": "node-1",
                "worker_id": "worker-1",
                "metadata": {"hit_rate": 0.9, "cache_entries": 5},
            },
        )
        self.assertTrue(bool(register.get("ok")))

        lookup = _request_json("GET", self._url("/lookup?hash=abcd1234&tenant_id=tenant-a"))
        nodes = lookup.get("nodes", [])
        self.assertEqual(len(nodes), 1)
        self.assertEqual(str(nodes[0].get("node_id")), "node-1")

        evict = _request_json("DELETE", self._url("/evict?hash=abcd1234&tenant_id=tenant-a"))
        self.assertTrue(bool(evict.get("ok")))
        self.assertEqual(int(evict.get("deleted", 0)), 1)

        lookup_after = _request_json("GET", self._url("/lookup?hash=abcd1234&tenant_id=tenant-a"))
        self.assertEqual(len(lookup_after.get("nodes", [])), 0)

    def test_heartbeat_rebuilds_node_index(self):
        hb = _request_json(
            "POST",
            self._url("/heartbeat"),
            {
                "node_id": "node-2",
                "entries": [
                    {"tenant_id": "tenant-a", "hash": "h1", "worker_id": "w2"},
                    {"tenant_id": "tenant-a", "hash": "h2", "worker_id": "w2"},
                ],
            },
        )
        self.assertTrue(bool(hb.get("ok")))
        lookup = _request_json("GET", self._url("/lookup?hash=h1&tenant_id=tenant-a"))
        self.assertEqual(len(lookup.get("nodes", [])), 1)

        hb2 = _request_json(
            "POST",
            self._url("/heartbeat"),
            {
                "node_id": "node-2",
                "entries": [
                    {"tenant_id": "tenant-a", "hash": "h2", "worker_id": "w2"},
                ],
            },
        )
        self.assertTrue(bool(hb2.get("ok")))
        lookup_stale = _request_json("GET", self._url("/lookup?hash=h1&tenant_id=tenant-a"))
        self.assertEqual(len(lookup_stale.get("nodes", [])), 0)

    def test_concurrent_lookups(self):
        _request_json(
            "POST",
            self._url("/register"),
            {
                "hash": "stress1",
                "tenant_id": "tenant-stress",
                "node_id": "node-s",
                "worker_id": "worker-s",
                "metadata": {},
            },
        )

        def _do_lookup(_: int) -> int:
            out = _request_json("GET", self._url("/lookup?hash=stress1&tenant_id=tenant-stress"))
            return len(out.get("nodes", []))

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            rows = list(pool.map(_do_lookup, range(100)))
        self.assertTrue(all(v == 1 for v in rows))


if __name__ == "__main__":
    unittest.main()
