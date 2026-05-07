"""Tests for the dedup cache — idempotency primitive."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from apexfx.aggressive.webhook.dedupe import DedupeCache


class TestConstruction:
    def test_invalid_max_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            DedupeCache(max_size=0)
        with pytest.raises(ValueError):
            DedupeCache(max_size=-1)

    def test_invalid_ttl_rejected(self) -> None:
        with pytest.raises(ValueError):
            DedupeCache(ttl_seconds=0)
        with pytest.raises(ValueError):
            DedupeCache(ttl_seconds=-1)


class TestAddIfNew:
    def test_first_insert_returns_true(self) -> None:
        cache = DedupeCache(max_size=10, ttl_seconds=60)
        assert cache.add_if_new("key1") is True

    def test_duplicate_returns_false(self) -> None:
        cache = DedupeCache(max_size=10, ttl_seconds=60)
        cache.add_if_new("key1")
        assert cache.add_if_new("key1") is False

    def test_different_keys_independent(self) -> None:
        cache = DedupeCache(max_size=10, ttl_seconds=60)
        assert cache.add_if_new("key1") is True
        assert cache.add_if_new("key2") is True
        assert cache.add_if_new("key1") is False

    def test_eviction_when_full(self) -> None:
        cache = DedupeCache(max_size=3, ttl_seconds=60)
        cache.add_if_new("a")
        cache.add_if_new("b")
        cache.add_if_new("c")
        cache.add_if_new("d")  # 'a' should be evicted (FIFO)
        assert cache.add_if_new("a") is True  # New again post-eviction
        assert cache.add_if_new("d") is False
        assert len(cache) == 3

    def test_ttl_expiry(self) -> None:
        cache = DedupeCache(max_size=10, ttl_seconds=0.5)
        cache.add_if_new("key1")
        # First call should still see it
        assert cache.add_if_new("key1") is False
        # Wait past TTL
        time.sleep(0.6)
        # Now expired — treated as new
        assert cache.add_if_new("key1") is True


class TestContains:
    def test_returns_true_for_added(self) -> None:
        cache = DedupeCache()
        cache.add_if_new("k")
        assert cache.contains("k") is True

    def test_returns_false_for_missing(self) -> None:
        cache = DedupeCache()
        assert cache.contains("k") is False

    def test_does_not_add(self) -> None:
        cache = DedupeCache()
        cache.contains("k")  # readonly check
        assert len(cache) == 0


class TestClear:
    def test_clear_removes_all(self) -> None:
        cache = DedupeCache()
        for i in range(10):
            cache.add_if_new(f"k{i}")
        assert len(cache) == 10
        cache.clear()
        assert len(cache) == 0
        # Previously-seen key is now new again
        assert cache.add_if_new("k0") is True


class TestThreadSafety:
    def test_concurrent_add_no_double_count(self) -> None:
        """Sanity check — two threads adding the same key shouldn't both
        return True (would let two webhook requests both place orders).
        Not exhaustive but catches obvious lock bugs."""
        import threading

        cache = DedupeCache(max_size=100, ttl_seconds=60)
        results: list[bool] = []

        def add() -> None:
            results.append(cache.add_if_new("shared-key"))

        threads = [threading.Thread(target=add) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one True, rest False
        assert results.count(True) == 1
        assert results.count(False) == 19


class TestSweep:
    def test_sweep_does_not_remove_fresh(self) -> None:
        cache = DedupeCache(max_size=10, ttl_seconds=10)
        cache.add_if_new("fresh")
        # Manual sweep via internal method
        cache._sweep_expired(time.monotonic())
        assert cache.contains("fresh")

    def test_sweep_removes_old_entries(self) -> None:
        cache = DedupeCache(max_size=10, ttl_seconds=1)
        with patch("time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            cache.add_if_new("old")
            mock_time.return_value = 1010.0
            # Now lookups treat as expired
            assert cache.add_if_new("old") is True  # New after expiry
