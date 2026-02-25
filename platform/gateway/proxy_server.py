from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request


ROUTER_URL = os.environ.get("KORITH_GATEWAY_ROUTER_URL", "http://127.0.0.1:8000").rstrip("/")


class ProxyHandler(BaseHTTPRequestHandler):
    def _proxy(self, method: str) -> None:
        url = f"{ROUTER_URL}{self.path}"
        body = None
        if method in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b""
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        request_id = self.headers.get("X-Request-Id")
        if request_id:
            headers["X-Request-Id"] = request_id
        req = request.Request(url=url, data=body, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=30) as resp:
                payload = resp.read()
                status = resp.getcode()
                content_type = resp.headers.get("Content-Type", "application/json")
        except error.HTTPError as exc:
            payload = exc.read() or b"{}"
            status = exc.code
            content_type = exc.headers.get("Content-Type", "application/json")
        except Exception as exc:
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            status = 502
            content_type = "application/json"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._proxy("GET")

    def do_POST(self) -> None:
        self._proxy("POST")

    def do_PUT(self) -> None:
        self._proxy("PUT")

    def do_PATCH(self) -> None:
        self._proxy("PATCH")

    def do_DELETE(self) -> None:
        self._proxy("DELETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
