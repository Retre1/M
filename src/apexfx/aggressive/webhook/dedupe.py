"""Idempotency cache for webhook signals.

Why
---
TradingView can fire the same alert twice in scenarios:

  * User clicks "Trigger now" on the alert page (debug/testing)
  * TV's internal retry on a network blip
  * Strategy emits an alert at bar close that matches a previously-fired
    bar (rare but possible with ``calc_on_every_tick``)

Without dedup, each duplicate becomes a real OKX order — doubling the
position from a single market signal.  That's exactly the kind of bug
that turns a $1k account into $0 overnight.

How
---
We keep an in-memory ring buffer of recent ``SignalId`` keys.  On every
new alert, we compute the key and check if it's been seen within the
TTL window.  Seen ⇒ drop the alert with ``status: duplicate``.

Why in-memory and not Redis
---------------------------
Single-VPS retail bot.  No horizontal scaling.  In-memory is the right
amount of complexity.  If you ever shard across multiple receivers,
swap this implementation for a Redis ``SETEX`` — the interface is
identical.

Thread safety
-------------
Flask's default dev server is single-threaded but production setups use
gunicorn with workers.  We use ``threading.Lock`` to make this safe
across workers in a single process.  For multi-process gunicorn, set
``--workers 1`` (recommended for retail anyway) or migrate to Redis.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict


class DedupeCache:
    """Bounded LRU cache with per-entry TTL.

    Invariants:
      * At most ``max_size`` entries (oldest evicted on overflow).
      * Entries older than ``ttl_seconds`` are filtered on lookup.
      * All operations are O(1) amortized except occasional ttl sweeps.

    Not a perfect implementation but good enough for ~hundreds of alerts
    per day in our use case.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def add_if_new(self, key: str) -> bool:
        """Insert ``key`` if not seen recently.

        Returns ``True`` if the key was new (added now), ``False`` if it
        was already in the cache and still within TTL.  This is the
        "check-and-set" primitive the webhook uses to dedupe.
        """
        now = time.monotonic()
        with self._lock:
            # Evict expired entries opportunistically
            self._sweep_expired(now)

            if key in self._entries:
                # Already seen, NOT a new signal
                return False

            # Insert and enforce size cap
            self._entries[key] = now
            if len(self._entries) > self._max_size:
                # OrderedDict.popitem(last=False) → FIFO eviction
                self._entries.popitem(last=False)
            return True

    def contains(self, key: str) -> bool:
        """Read-only check (for tests / metrics).  Does NOT add the key."""
        now = time.monotonic()
        with self._lock:
            self._sweep_expired(now)
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Wipe all entries.  Used in tests; would also be a debug
        endpoint in a real ops setup."""
        with self._lock:
            self._entries.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sweep_expired(self, now: float) -> None:
        """Drop all entries whose insertion time is older than TTL.

        OrderedDict iterates in insertion order, so we can stop at the
        first non-expired entry — keeps the sweep O(k) where k is the
        number of expired entries this round.

        Caller MUST hold the lock.
        """
        cutoff = now - self._ttl
        keys_to_drop: list[str] = []
        for key, ts in self._entries.items():
            if ts < cutoff:
                keys_to_drop.append(key)
            else:
                break  # Remainder is fresher (insertion order)
        for key in keys_to_drop:
            self._entries.pop(key, None)
