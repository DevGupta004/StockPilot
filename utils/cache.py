"""Tiny per-day JSON cache shared by the data and news layers.

Keyed by (namespace, key, YYYY-MM-DD) so price/news for a symbol is fetched at most
once per calendar day, which keeps us well under the free-tier rate limits. Failures
are swallowed — caching is best-effort and must never crash a tool call.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from typing import Any

from utils.log import log


def _today() -> str:
    return date.today().isoformat()


class DayCache:
    def __init__(self, root: str) -> None:
        self.root = root
        try:
            os.makedirs(root, exist_ok=True)
        except OSError as exc:  # pragma: no cover - disk issues
            log(f"cache: cannot create {root}: {exc}")

    def _path(self, namespace: str, key: str) -> str:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        return os.path.join(self.root, f"{namespace}__{safe}__{_today()}.json")

    def get(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log(f"cache: read miss {path}: {exc}")
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"_ts": time.time(), "data": value}, fh)
        except (OSError, TypeError) as exc:
            log(f"cache: write fail {path}: {exc}")

    def get_data(self, namespace: str, key: str) -> Any | None:
        """Return only the payload portion of a cached entry."""
        wrapped = self.get(namespace, key)
        if isinstance(wrapped, dict) and "data" in wrapped:
            return wrapped["data"]
        return wrapped
