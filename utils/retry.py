"""Retry with exponential backoff + jitter.

Used to ride out transient rate-limits / network blips from the free data sources
(yfinance, nsepython, NSE archives). A failed attempt sleeps ``base * 2**n`` seconds
plus a little random jitter, then retries; after the last attempt the exception is
re-raised (or, with ``default``, swallowed and the default returned).
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from utils.log import log

T = TypeVar("T")

_SENTINEL = object()


def retry_call(fn: Callable[[], T], *, attempts: int = 3, base: float = 1.0,
               max_sleep: float = 20.0, label: str = "",
               default: object = _SENTINEL) -> T:
    """Call ``fn`` up to ``attempts`` times with exponential backoff + jitter.

    On total failure: re-raise the last exception, unless ``default`` is given (then
    log and return it). ``label`` is used only in log lines.
    """
    last: Exception | None = None
    for n in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - providers raise anything
            last = exc
            if n == attempts - 1:
                break
            sleep = min(base * (2 ** n) + random.uniform(0, base), max_sleep)
            log(f"retry[{label or fn.__name__}]: attempt {n + 1}/{attempts} failed "
                f"({exc}); backing off {sleep:.1f}s")
            time.sleep(sleep)
    if default is not _SENTINEL:
        log(f"retry[{label}]: all {attempts} attempts failed ({last}); using default")
        return default  # type: ignore[return-value]
    raise last  # type: ignore[misc]
