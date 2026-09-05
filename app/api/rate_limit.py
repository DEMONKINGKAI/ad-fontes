"""In-memory per-IP rate limiting for the one endpoint that costs money.

Why in-process and not Redis: the brief forbids managed services, the Space is a
single process, and the goal is only to stop casual abuse of a personal project
(same rationale as fons-iuris). A sliding window of request timestamps per client
key is enough. Memory is bounded by pruning empty buckets on each call.

Privacy: the client key is a salted, truncated hash of the IP, never the raw
address — the brief forbids logging full IPs, and this keeps them out of memory
too.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from threading import Lock

_SALT = os.environ.get("AD_FONTES_RL_SALT", "ad-fontes-local-salt").encode("utf-8")


def client_key(ip: str | None) -> str:
    """Salted, truncated hash of an IP. Stable within a process run, not reversible."""
    raw = (ip or "unknown").encode("utf-8")
    return hashlib.blake2b(raw, salt=_SALT[:16], digest_size=8).hexdigest()


class RateLimiter:
    """Sliding-window limiter: ``max_requests`` per ``window_s`` per client key."""

    def __init__(self, max_requests: int, window_s: float) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int, float]:
        """Return ``(allowed, remaining, retry_after_s)`` and record the hit if allowed."""
        now = time.monotonic() if now is None else now
        cutoff = now - self.window_s
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = bucket[0] + self.window_s - now
                return False, 0, max(retry_after, 0.0)
            bucket.append(now)
            self._prune(cutoff)
            return True, self.max_requests - len(bucket), 0.0

    def _prune(self, cutoff: float) -> None:
        """Drop client keys whose windows have fully expired. Caller holds the lock."""
        empty = [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]
        for k in empty:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
