"""Tests for src/apexfx/data/news_fetcher.py."""

from __future__ import annotations

import pytest

from apexfx.data.news_fetcher import NewsFetcher, _MAX_RSS_BYTES


class TestIsAllowedFeed:
    def test_allowed_domain_accepted(self):
        assert NewsFetcher._is_allowed_feed("https://www.forexlive.com/feed/news") is True
        assert NewsFetcher._is_allowed_feed("https://fxstreet.com/rss") is True
        assert NewsFetcher._is_allowed_feed("https://www.reuters.com/rss") is True

    def test_subdomain_accepted(self):
        assert NewsFetcher._is_allowed_feed("https://news.reuters.com/feed") is True

    def test_unknown_domain_rejected(self):
        assert NewsFetcher._is_allowed_feed("https://evil.com/rss") is False
        assert NewsFetcher._is_allowed_feed("https://malicious-site.org/feed") is False

    def test_empty_url_rejected(self):
        assert NewsFetcher._is_allowed_feed("") is False

    def test_invalid_url_rejected(self):
        assert NewsFetcher._is_allowed_feed("not-a-url") is False


class TestParseRss:
    def test_valid_rss_xml(self):
        xml = """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>EUR/USD rises on dovish Fed</title>
              <pubDate>Mon, 24 Mar 2026 10:00:00 GMT</pubDate>
            </item>
            <item>
              <title>Gold hits record high</title>
            </item>
          </channel>
        </rss>"""
        fetcher = NewsFetcher(feeds=["https://forexlive.com/feed"])
        headlines = fetcher._parse_rss(xml, "https://forexlive.com/feed")
        assert len(headlines) == 2
        assert headlines[0]["text"] == "EUR/USD rises on dovish Fed"
        assert headlines[0]["source"] == "forexlive.com"
        assert "timestamp" in headlines[0]

    def test_malformed_xml_falls_back_to_regex(self):
        malformed_xml = """<rss>
          <channel>
            <item><title>Headline from broken feed</title></item>
            <!-- missing closing tags -->
        """
        fetcher = NewsFetcher(feeds=["https://forexlive.com/feed"])
        headlines = fetcher._parse_rss(malformed_xml, "https://forexlive.com/feed")
        assert len(headlines) == 1
        assert headlines[0]["text"] == "Headline from broken feed"

    def test_atom_feed_title_without_namespace(self):
        """Atom entries are found but title lookup uses unqualified 'title',
        so Atom entries with namespaced <title> are skipped. This test
        documents that current behavior."""
        atom_xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>BOJ holds rates steady</title>
          </entry>
        </feed>"""
        fetcher = NewsFetcher(feeds=["https://reuters.com/feed"])
        headlines = fetcher._parse_rss(atom_xml, "https://reuters.com/feed")
        # Atom entries have namespaced <title>, so find("title") returns None
        assert len(headlines) == 0

    def test_empty_title_skipped(self):
        xml = """<?xml version="1.0"?>
        <rss><channel>
          <item><title></title></item>
          <item><title>Valid headline</title></item>
        </channel></rss>"""
        fetcher = NewsFetcher(feeds=["https://forexlive.com/feed"])
        headlines = fetcher._parse_rss(xml, "https://forexlive.com/feed")
        assert len(headlines) == 1
        assert headlines[0]["text"] == "Valid headline"


class TestExtractDomain:
    def test_extracts_domain(self):
        assert NewsFetcher._extract_domain("https://www.forexlive.com/feed/news") == "forexlive.com"
        assert NewsFetcher._extract_domain("https://fxstreet.com/rss") == "fxstreet.com"

    def test_handles_subdomain(self):
        assert NewsFetcher._extract_domain("https://news.reuters.com/feed") == "news.reuters.com"

    def test_invalid_url_returns_input(self):
        assert NewsFetcher._extract_domain("") == ""


class TestHeadlineLengthCap:
    def test_headline_capped_at_500_chars(self):
        long_title = "A" * 600
        xml = f"""<?xml version="1.0"?>
        <rss><channel>
          <item><title>{long_title}</title></item>
        </channel></rss>"""
        fetcher = NewsFetcher(feeds=["https://forexlive.com/feed"])
        headlines = fetcher._parse_rss(xml, "https://forexlive.com/feed")
        assert len(headlines) == 1
        assert len(headlines[0]["text"]) == 500


class TestRssSizeGuard:
    def test_max_rss_bytes_constant(self):
        assert _MAX_RSS_BYTES == 1_048_576  # 1 MB


class TestNewsFetcherInit:
    def test_blocked_feeds_filtered(self):
        feeds = [
            "https://forexlive.com/feed",
            "https://evil-site.com/rss",
        ]
        fetcher = NewsFetcher(feeds=feeds)
        assert len(fetcher._feeds) == 1
        assert fetcher._feeds[0] == "https://forexlive.com/feed"

    def test_default_feeds_all_allowed(self):
        fetcher = NewsFetcher()
        assert len(fetcher._feeds) == len(NewsFetcher.DEFAULT_FEEDS)
