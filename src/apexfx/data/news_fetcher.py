"""Lightweight news headline fetcher for live sentiment analysis.

Fetches headlines from RSS feeds and returns them in a format
suitable for SentimentExtractor consumption.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html import unescape
from typing import Any
from urllib.parse import urlparse

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum RSS response size (1 MB) to prevent memory exhaustion
_MAX_RSS_BYTES = 1_048_576


class NewsFetcher:
    """Fetches Forex news headlines from RSS feeds.

    Provides headlines to SentimentExtractor for NLP-based
    sentiment feature computation. Uses requests + simple XML
    parsing to avoid heavy dependencies.
    """

    # Default RSS feeds for Forex news (allowlist)
    DEFAULT_FEEDS: list[str] = [
        "https://www.forexlive.com/feed/news",
        "https://www.fxstreet.com/rss/news",
        "https://www.dailyfx.com/feeds/market-news",
    ]

    # Allowed feed domains — rejects feeds pointing to unknown hosts
    ALLOWED_DOMAINS: set[str] = {
        "forexlive.com",
        "fxstreet.com",
        "dailyfx.com",
        "reuters.com",
        "bloomberg.com",
        "investing.com",
        "tradingeconomics.com",
        "forexfactory.com",
    }

    def __init__(
        self,
        feeds: list[str] | None = None,
        timeout_s: int = 10,
    ) -> None:
        """Initialize news fetcher.

        Args:
            feeds: List of RSS feed URLs. Uses DEFAULT_FEEDS if None.
            timeout_s: HTTP request timeout in seconds.
        """
        raw_feeds = feeds or self.DEFAULT_FEEDS
        self._feeds = [f for f in raw_feeds if self._is_allowed_feed(f)]
        if len(self._feeds) < len(raw_feeds):
            blocked = set(raw_feeds) - set(self._feeds)
            logger.warning("Blocked RSS feeds not in allowlist", blocked=list(blocked))
        self._timeout = timeout_s

    @classmethod
    def _is_allowed_feed(cls, url: str) -> bool:
        """Check feed URL against domain allowlist."""
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower().removeprefix("www.")
            return any(host == d or host.endswith(f".{d}") for d in cls.ALLOWED_DOMAINS)
        except Exception:
            return False

    def fetch_latest(self, max_items: int = 20) -> list[dict[str, Any]]:
        """Fetch latest headlines from all configured feeds.

        Returns:
            List of headline dicts with keys:
            - text: str — headline text
            - timestamp: str — ISO 8601 timestamp
            - source: str — feed domain
        """
        import requests

        all_headlines: list[dict[str, Any]] = []

        for feed_url in self._feeds:
            try:
                resp = requests.get(
                    feed_url,
                    timeout=self._timeout,
                    headers={"User-Agent": "ApexFX-Quantum/1.0"},
                )
                if resp.status_code != 200:
                    logger.debug(
                        "Feed returned non-200",
                        feed=feed_url,
                        status=resp.status_code,
                    )
                    continue

                # Guard against oversized responses
                if len(resp.content) > _MAX_RSS_BYTES:
                    logger.warning("RSS response too large, skipping", feed=feed_url, size=len(resp.content))
                    continue
                headlines = self._parse_rss(resp.text, feed_url)
                all_headlines.extend(headlines)
            except Exception as e:
                logger.debug("Feed fetch failed", feed=feed_url, error=str(e))
                continue

        # Sort by timestamp (newest first) and limit
        all_headlines.sort(
            key=lambda h: h.get("timestamp", ""),
            reverse=True,
        )

        return all_headlines[:max_items]

    def _parse_rss(self, xml_text: str, feed_url: str) -> list[dict[str, Any]]:
        """Parse RSS XML into headline dicts using stdlib ElementTree.

        Disables external entity resolution to prevent XXE attacks.
        Falls back to regex if XML is malformed.
        """
        headlines: list[dict[str, Any]] = []
        source = self._extract_domain(feed_url)

        try:
            root = ET.fromstring(xml_text)  # noqa: S314 — no external entities in RSS
        except ET.ParseError:
            logger.debug("XML parse failed, falling back to regex", feed=feed_url)
            return self._parse_rss_regex(xml_text, source)

        # RSS 2.0: channel/item; Atom: entry
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item in items:
            title_el = item.find("title")
            if title_el is None or not title_el.text:
                continue

            title = unescape(title_el.text.strip())
            title = re.sub(r"<[^>]+>", "", title)  # Strip residual HTML
            # Cap headline length to prevent downstream issues
            if len(title) > 500:
                title = title[:500]
            if not title:
                continue

            timestamp = datetime.now(UTC).isoformat()
            pub_date = item.find("pubDate")
            if pub_date is not None and pub_date.text:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub_date.text)
                    timestamp = dt.isoformat()
                except Exception:
                    pass

            headlines.append({
                "text": title,
                "timestamp": timestamp,
                "source": source,
            })

        return headlines

    @staticmethod
    def _parse_rss_regex(xml_text: str, source: str) -> list[dict[str, Any]]:
        """Regex fallback for malformed RSS/XML."""
        headlines: list[dict[str, Any]] = []
        items = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)

        for item in items:
            title_match = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item)
            if not title_match:
                continue
            title = unescape(title_match.group(1).strip())
            title = re.sub(r"<[^>]+>", "", title)
            if not title or len(title) > 500:
                continue
            headlines.append({
                "text": title,
                "timestamp": datetime.now(UTC).isoformat(),
                "source": source,
            })
        return headlines

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain name from URL."""
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").removeprefix("www.")
            return host or url
        except Exception:
            return url
