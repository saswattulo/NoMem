"""Command line: ``nomem serve`` (HTTP API) and ``nomem mcp`` (MCP server)."""

from __future__ import annotations

import argparse
import os
import sys


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=os.environ.get("NOMEM_DB", "nomem.db"),
                   help="SQLite file (default: nomem.db, or $NOMEM_DB)")
    p.add_argument("--max-tokens", type=int, default=3000, help="default context budget")
    p.add_argument("--core", default="", help="persona/rules placed first in every context")
    p.add_argument("--embedder", default=os.environ.get("NOMEM_EMBEDDER", "none"),
                   choices=["none", "model2vec", "hashing"],
                   help="optional vector ranker (model2vec needs nomem[embed])")


def main(argv=None) -> int:
    from . import __version__

    ap = argparse.ArgumentParser(prog="nomem", description="NoMem: LLM-free memory for agents.")
    ap.add_argument("--version", action="version", version=f"nomem {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the JSON HTTP API")
    _common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8420)
    s.add_argument("--api-key", default=os.environ.get("NOMEM_API_KEY"),
                   help="require 'Authorization: Bearer <key>' (or set $NOMEM_API_KEY)")
    s.add_argument("--verbose", action="store_true", help="log every request")

    m = sub.add_parser("mcp", help="run the MCP server (stdio by default)")
    _common(m)
    m.add_argument("--user-id", default=os.environ.get("NOMEM_USER_ID", "default"))
    m.add_argument("--session-id", default=os.environ.get("NOMEM_SESSION_ID", "default"))
    m.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    m.add_argument("--host", default=None)
    m.add_argument("--port", type=int, default=None)

    args = ap.parse_args(argv)

    from .service import MemoryService

    service = MemoryService(args.db, max_tokens=args.max_tokens, core=args.core,
                            embedder=args.embedder)
    if args.cmd == "serve":
        from .http_api import serve

        if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
            print("warning: listening beyond localhost with no --api-key", file=sys.stderr)
        serve(service, args.host, args.port, args.api_key, args.verbose)
    else:
        from .mcp_server import run

        run(service, args.user_id, args.session_id, args.transport, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
