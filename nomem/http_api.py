"""A small JSON HTTP API for NoMem, built on the standard library only.

Run it with ``nomem serve`` (see ``nomem serve --help``). Endpoints, all JSON:

======  ======================  ==============================================
Method  Path                    What it does
======  ======================  ==============================================
GET     /health                 ``{"status": "ok", "version": ...}``
POST    /v1/messages            Store one message, or ``{"messages": [...]}``
POST    /v1/context             Build a token-budgeted context
POST    /v1/search              Ranked search with scores and reasons
GET     /v1/facts               Current facts (``?history=true`` for all)
POST    /v1/facts               Pin a fact: ``{"key", "value"}``
GET     /v1/conflicts           Memories that a newer one may contradict
GET     /v1/stats               Message count for a user and session
======  ======================  ==============================================

Every request may carry ``user_id`` and ``session_id`` (in the JSON body, or
the query string for GET). Set ``NOMEM_API_KEY`` (or ``--api-key``) to require
``Authorization: Bearer <key>``. There is no LLM behind any endpoint.
"""

from __future__ import annotations

import hmac
import sys
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .service import MemoryService

MAX_BODY = 5 * 1024 * 1024  # 5 MB


class _Error(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def make_handler(service: MemoryService, api_key: Optional[str] = None, version: str = ""):
    """Build a request handler class bound to ``service``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"nomem/{version}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter default logging
            if getattr(self.server, "verbose", False):
                super().log_message(fmt, *args)

        # -- plumbing ----------------------------------------------------

        def _send(self, status: int, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _auth(self) -> None:
            if not api_key:
                return
            got = self.headers.get("Authorization", "")
            if not hmac.compare_digest(got, f"Bearer {api_key}"):
                raise _Error(401, "missing or wrong bearer token")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise _Error(413, "request body too large")
            if n == 0:
                return {}
            try:
                data = json.loads(self.rfile.read(n))
            except (ValueError, UnicodeDecodeError):
                raise _Error(400, "body must be JSON")
            if not isinstance(data, dict):
                raise _Error(400, "body must be a JSON object")
            return data

        def _dispatch(self, method: str) -> None:
            try:
                url = urlparse(self.path)
                if url.path == "/health" and method == "GET":
                    return self._send(200, {"status": "ok", "version": version})
                self._auth()
                q = {k: v[-1] for k, v in parse_qs(url.query).items()}
                data = self._body() if method == "POST" else q
                ids = {"user_id": str(data.get("user_id") or q.get("user_id") or "default")}
                sess = str(data.get("session_id") or q.get("session_id") or "default")
                route = (method, url.path.rstrip("/"))
                if route == ("POST", "/v1/messages"):
                    if "messages" in data:
                        if not isinstance(data["messages"], list):
                            raise _Error(400, "messages must be a list")
                        out = service.add_many(data["messages"], session_id=sess, **ids)
                        return self._send(201, {"messages": out})
                    out = service.add(data.get("content", ""), data.get("role", "user"),
                                      session_id=sess, timestamp=data.get("timestamp"),
                                      metadata=data.get("metadata"), **ids)
                    return self._send(201, out)
                if route == ("POST", "/v1/context"):
                    return self._send(200, service.context(
                        data.get("query"), session_id=sess, max_tokens=data.get("max_tokens"),
                        explain=bool(data.get("explain")), **ids))
                if route in (("POST", "/v1/search"), ("GET", "/v1/search")):
                    query = data.get("query") or data.get("q") or ""
                    hits = service.search(query, session_id=sess, k=int(data.get("k") or 5), **ids)
                    return self._send(200, {"hits": hits})
                if route == ("GET", "/v1/facts"):
                    hist = str(q.get("history", "")).lower() in ("1", "true", "yes")
                    return self._send(200, service.facts(history=hist, **ids))
                if route == ("POST", "/v1/facts"):
                    return self._send(201, service.remember(str(data.get("key") or ""),
                                                            str(data.get("value") or ""), **ids))
                if route == ("GET", "/v1/conflicts"):
                    return self._send(200, {"conflicts": service.conflicts(**ids)})
                if route == ("GET", "/v1/stats"):
                    return self._send(200, service.stats(session_id=sess, **ids))
                raise _Error(404, f"no route for {method} {url.path}")
            except _Error as e:
                self._send(e.status, {"error": e.message})
            except (ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # pragma: no cover - last resort
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def serve(service: MemoryService, host: str = "127.0.0.1", port: int = 8420,
          api_key: Optional[str] = None, verbose: bool = False) -> None:
    """Serve ``service`` until interrupted."""
    from . import __version__

    httpd = ThreadingHTTPServer((host, port), make_handler(service, api_key, __version__))
    httpd.verbose = verbose
    print(f"NoMem HTTP API on http://{host}:{port}  (Ctrl+C to stop)", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        service.close()
