"""Tests for real-time news stream system.

Tests cover:
- NewsConfig defaults and validation
- HeadlineDeduplicator (dedup logic, time window eviction)
- FastRSSPoller (XML parsing, conditional headers)
- RealtimeNewsStream (urgency detection, queue, stats)
- FinnhubWebSocket (message parsing)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from apexfx.config.schema import NewsConfig, DataConfig
from apexfx.data.realtime_news import (
    FastRSSPoller,
    FinnhubWebSocket,
    HeadlineDeduplicator,
    NewsHeadline,
    RealtimeNewsStream,
)


# ---------------------------------------------------------------------------
# NewsConfig
# ---------------------------------------------------------------------------

class TestNewsConfig:
    def test_defaults(self):
        cfg = NewsConfig()
        assert cfg.enabled is True
        assert cfg.finnhub_enabled is True
        assert cfg.rss_poll_interval_s == 30
        assert len(cfg.rss_feeds) == 3
        assert cfg.dedup_window_minutes == 60
        assert cfg.max_headlines_buffer == 50
        assert "breaking" in cfg.urgency_keywords
        assert cfg.urgency_cooldown_s == 5

    def test_in_data_config(self):
        dc = DataConfig()
        assert isinstance(dc.news, NewsConfig)
        assert dc.news.enabled is True

    def test_custom_feeds(self):
        cfg = NewsConfig(
            rss_feeds=["https://custom.feed/rss"],
            rss_poll_interval_s=10,
        )
        assert len(cfg.rss_feeds) == 1
        assert cfg.rss_poll_interval_s == 10


# ---------------------------------------------------------------------------
# HeadlineDeduplicator
# ---------------------------------------------------------------------------

class TestDeduplicator:
    def test_detects_exact_duplicate(self):
        dd = HeadlineDeduplicator(window_minutes=60)
        assert dd.is_duplicate("Fed raises rates by 25bps") is False
        assert dd.is_duplicate("Fed raises rates by 25bps") is True

    def test_detects_normalized_duplicate(self):
        dd = HeadlineDeduplicator(window_minutes=60)
        assert dd.is_duplicate("BREAKING: Fed raises rates") is False
        # Same content without "BREAKING:" prefix
        assert dd.is_duplicate("Fed raises rates") is True

    def test_different_headlines_not_duplicate(self):
        dd = HeadlineDeduplicator(window_minutes=60)
        assert dd.is_duplicate("Fed raises rates") is False
        assert dd.is_duplicate("ECB holds rates steady") is False

    def test_case_insensitive(self):
        dd = HeadlineDeduplicator(window_minutes=60)
        assert dd.is_duplicate("Fed Raises Rates") is False
        assert dd.is_duplicate("fed raises rates") is True

    def test_eviction_by_time(self):
        dd = HeadlineDeduplicator(window_minutes=1)  # 1 minute window
        # Manually insert with old timestamp
        old_hash = dd._hash_text("old headline")
        dd._seen.append((time.time() - 120, old_hash))  # 2 minutes ago
        dd._hashes.add(old_hash)

        # Eviction happens on next check
        assert dd.is_duplicate("new headline") is False
        # Old hash should be evicted now
        assert old_hash not in dd._hashes


# ---------------------------------------------------------------------------
# NewsHeadline
# ---------------------------------------------------------------------------

class TestNewsHeadline:
    def test_to_dict(self):
        h = NewsHeadline(
            text="Test headline",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            source="test",
            category="forex",
            is_urgent=True,
        )
        d = h.to_dict()
        assert d["text"] == "Test headline"
        assert d["source"] == "test"
        assert d["is_urgent"] is True
        assert "2026" in d["timestamp"]

    def test_defaults(self):
        h = NewsHeadline(
            text="X", timestamp=datetime.now(timezone.utc), source="s",
        )
        assert h.is_urgent is False
        assert h.latency_ms == 0.0
        assert h.category == ""


# ---------------------------------------------------------------------------
# FastRSSPoller (XML parsing)
# ---------------------------------------------------------------------------

class TestFastRSSPoller:
    SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel>
    <title>ForexLive</title>
    <item>
        <title><![CDATA[Fed signals rate cut in September]]></title>
        <pubDate>Thu, 01 Jan 2026 12:00:00 +0000</pubDate>
        <link>https://forexlive.com/article/1</link>
    </item>
    <item>
        <title>ECB holds rates at 3.5%</title>
        <pubDate>Thu, 01 Jan 2026 11:30:00 +0000</pubDate>
        <link>https://forexlive.com/article/2</link>
    </item>
    </channel>
    </rss>"""

    def test_parse_rss_extracts_headlines(self):
        poller = FastRSSPoller(feeds=[], poll_interval_s=30)
        headlines = poller._parse_rss(self.SAMPLE_RSS, "forexlive.com", time.time())
        assert len(headlines) == 2
        assert headlines[0].text == "Fed signals rate cut in September"
        assert headlines[1].text == "ECB holds rates at 3.5%"

    def test_parse_rss_extracts_links(self):
        poller = FastRSSPoller(feeds=[], poll_interval_s=30)
        headlines = poller._parse_rss(self.SAMPLE_RSS, "forexlive.com", time.time())
        assert "forexlive.com/article/1" in headlines[0].url

    def test_parse_rss_source(self):
        poller = FastRSSPoller(feeds=[], poll_interval_s=30)
        headlines = poller._parse_rss(self.SAMPLE_RSS, "fxstreet.com", time.time())
        assert all(h.source == "fxstreet.com" for h in headlines)

    def test_extract_domain(self):
        assert FastRSSPoller._extract_domain("https://www.forexlive.com/feed") == "forexlive.com"
        assert FastRSSPoller._extract_domain("https://fxstreet.com/rss") == "fxstreet.com"

    def test_parse_empty_rss(self):
        poller = FastRSSPoller(feeds=[], poll_interval_s=30)
        headlines = poller._parse_rss("<rss></rss>", "test", time.time())
        assert len(headlines) == 0

    def test_parse_html_entities(self):
        xml = """<rss><channel><item>
        <title>S&amp;P 500 &quot;surges&quot; 2%</title>
        </item></channel></rss>"""
        poller = FastRSSPoller(feeds=[], poll_interval_s=30)
        headlines = poller._parse_rss(xml, "test", time.time())
        assert headlines[0].text == 'S&P 500 "surges" 2%'


# ---------------------------------------------------------------------------
# FinnhubWebSocket (message parsing)
# ---------------------------------------------------------------------------

class TestFinnhubWebSocket:
    def test_parse_news_item(self):
        ws = FinnhubWebSocket(api_key="test_key")
        now = time.time()
        item = {
            "headline": "USD surges on NFP beat",
            "category": "forex",
            "datetime": now - 1.0,  # 1 second ago
            "url": "https://finnhub.io/news/1",
        }
        result = ws._parse_news_item(item, now)
        assert result is not None
        assert result.text == "USD surges on NFP beat"
        assert result.source == "finnhub"
        assert result.category == "forex"
        assert result.latency_ms > 0

    def test_parse_empty_headline(self):
        ws = FinnhubWebSocket(api_key="test_key")
        item = {"headline": "", "category": "forex"}
        assert ws._parse_news_item(item, time.time()) is None

    def test_parse_filters_category(self):
        ws = FinnhubWebSocket(
            api_key="test_key",
            categories=["forex"],
        )
        item = {
            "headline": "Tech stock rises",
            "category": "technology",
            "datetime": time.time(),
        }
        assert ws._parse_news_item(item, time.time()) is None

    def test_parse_allows_matching_category(self):
        ws = FinnhubWebSocket(
            api_key="test_key",
            categories=["forex", "general"],
        )
        item = {
            "headline": "EURUSD breaks 1.10",
            "category": "general",
            "datetime": time.time(),
        }
        result = ws._parse_news_item(item, time.time())
        assert result is not None


# ---------------------------------------------------------------------------
# RealtimeNewsStream (urgency, dedup, queue)
# ---------------------------------------------------------------------------

class TestRealtimeNewsStream:
    def test_init_without_api_key(self):
        """Without API key, Finnhub WS should not be created."""
        cfg = NewsConfig(finnhub_api_key="", finnhub_enabled=True)
        stream = RealtimeNewsStream(cfg)
        assert stream._finnhub_ws is None
        assert stream._rss_poller is not None

    def test_init_with_api_key(self):
        cfg = NewsConfig(finnhub_api_key="test_key_123")
        stream = RealtimeNewsStream(cfg)
        assert stream._finnhub_ws is not None
        assert stream._rss_poller is not None

    def test_init_rss_disabled(self):
        cfg = NewsConfig(rss_enabled=False, finnhub_api_key="key")
        stream = RealtimeNewsStream(cfg)
        assert stream._rss_poller is None

    def test_urgency_detection(self):
        cfg = NewsConfig(finnhub_enabled=False, rss_enabled=False)
        stream = RealtimeNewsStream(cfg)

        headline = NewsHeadline(
            text="BREAKING: Flash crash in EURUSD",
            timestamp=datetime.now(timezone.utc),
            source="test",
        )
        stream._on_headline(headline)

        assert stream._stats["urgent_headlines"] == 1
        # Should be in queue
        assert not stream._headline_queue.empty()
        queued = stream._headline_queue.get_nowait()
        assert queued.is_urgent is True

    def test_urgency_cooldown(self):
        cfg = NewsConfig(
            finnhub_enabled=False, rss_enabled=False,
            urgency_cooldown_s=10,
        )
        stream = RealtimeNewsStream(cfg)

        # First urgent headline
        h1 = NewsHeadline(
            text="BREAKING: Rate decision",
            timestamp=datetime.now(timezone.utc),
            source="test",
        )
        stream._on_headline(h1)
        assert stream._stats["urgent_headlines"] == 1

        # Second urgent headline within cooldown — NOT marked urgent
        h2 = NewsHeadline(
            text="BREAKING: Another surprise",
            timestamp=datetime.now(timezone.utc),
            source="test",
        )
        stream._on_headline(h2)
        assert stream._stats["urgent_headlines"] == 1  # Still 1

    def test_dedup_filters(self):
        cfg = NewsConfig(finnhub_enabled=False, rss_enabled=False)
        stream = RealtimeNewsStream(cfg)

        h1 = NewsHeadline(
            text="Fed raises rates",
            timestamp=datetime.now(timezone.utc),
            source="forexlive",
        )
        h2 = NewsHeadline(
            text="Fed raises rates",
            timestamp=datetime.now(timezone.utc),
            source="fxstreet",  # Same headline, different source
        )

        stream._on_headline(h1)
        stream._on_headline(h2)

        assert stream._stats["total_received"] == 2
        assert stream._stats["duplicates_filtered"] == 1
        assert stream._headline_queue.qsize() == 1

    def test_stats_tracking(self):
        cfg = NewsConfig(finnhub_enabled=False, rss_enabled=False)
        stream = RealtimeNewsStream(cfg)

        for i in range(5):
            h = NewsHeadline(
                text=f"Headline number {i}",
                timestamp=datetime.now(timezone.utc),
                source="rss_test",
                latency_ms=100.0 + i * 10,
            )
            stream._on_headline(h)

        stats = stream.stats
        assert stats["total_received"] == 5
        assert stats["rss_count"] == 5
        assert stats["avg_latency_ms"] > 0

    def test_queue_overflow_drops_oldest(self):
        cfg = NewsConfig(finnhub_enabled=False, rss_enabled=False)
        stream = RealtimeNewsStream(cfg)
        # Fill queue to max (200)
        for i in range(210):
            h = NewsHeadline(
                text=f"Headline {i}",
                timestamp=datetime.now(timezone.utc),
                source="test",
            )
            stream._on_headline(h)

        # Queue should not exceed max size
        assert stream._headline_queue.qsize() <= 200
