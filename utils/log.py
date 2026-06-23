"""Stderr-only logging.

CRITICAL for stdio MCP: stdout carries the JSON-RPC protocol. A single stray write
to stdout corrupts the connection. Every diagnostic in this project goes through
``log`` which writes to stderr and flushes immediately.
"""

from __future__ import annotations

import sys
from datetime import datetime


def log(*parts: object) -> None:
    msg = " ".join(str(p) for p in parts)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}",
          file=sys.stderr, flush=True)
