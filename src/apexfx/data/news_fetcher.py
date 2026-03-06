"""Lightweight news headline fetcher for live sentiment analysis.

Fetches headlines from RSS feeds and returns them in a format
suitable for SentimentExtractor consumption.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class NewsFetcher:
    """Fetches Forex news headlines from RSS feeds.

    Provides headlines to SentimentExtractor for NLP-based
    sentiment feature computation. Uses requests + simple XML
    parsing to avoid heavy dependencies.
    """

    # Default RSS feeds for Forex news
    DEFAULT_FEEDS: list[str] = [
        "https://www.forexlive.com/feed/news",
        "https://www.fxstreet.com/rss/news",
        "https://www.dailyfx.com/feeds/market-news",
    ]

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
        self._feeds = feeds or self.DEFAULT_FEEDS
        self._timeout = timeout_s

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
        """Parse RSS XML into headline dicts.

        Uses regex-based parsing to avoid lxml/defusedxml dependency.
        """
        headlines: list[dict[str, Any]] = []
        source = self._extract_domain(feed_url)

        # Extract <item> blocks
        items = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)

        for item in items:
            title_match = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item)
            date_match = re.search(r"<pubDate>(.*?)</pubDate>", item)

            if not title_match:
                continue

            title = title_match.group(1).strip()
            # Clean HTML entities
            title = re.sub(r"&amp;", "&", title)
            title = re.sub(r"&lt;", "<", title)
            title = re.sub(r"&gt;", ">", title)
            title = re.sub(r"&quot;", '"', title)
            title = re.sub(r"<[^>]+>", "", title)  # Strip any HTML tags

            if not title:
                continue

            timestamp = datetime.now(timezone.utc).isoformat()
            if date_match:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(date_match.group(1))
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
    def _extract_domain(url: str) -> str:
        """Extract domain name from URL."""
        match = re.search(r"https?://(?:www\.)?([\w.-]+)", url)
        return match.group(1) if match else url
