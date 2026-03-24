"""Real-time news stream for institutional-speed reaction.

Priority chain for lowest latency:
1. Finnhub WebSocket (~1s latency) — push-based, no polling
2. Fast RSS polling (~30s) — aggressive polling with deduplication
3. Standard RSS fallback (15min) — original behavior

Architecture:
    RealtimeNewsStream
    ├── FinnhubWebSocket (primary, async)
    ├── FastRSSPoller (fallback, async)
    └── HeadlineDeduplicator (prevents duplicates across sources)

All sources feed into a unified asyncio.Queue consumed by the trading loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from apexfx.config.schema import NewsConfig
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NewsHeadline:
    """Unified headline from any source."""

    text: str
    timestamp: datetime
    source: str
    category: str = ""
    url: str = ""
    is_urgent: bool = False
    latency_ms: float = 0.0  # Time from publish to receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "category": self.category,
            "url": self.url,
            "is_urgent": self.is_urgent,
        }


class HeadlineDeduplicator:
    """Prevents duplicate headlines across multiple sources.

    Uses content hash (normalized lowercase text) within a time window.
    """

    def __init__(self, window_minutes: int = 60) -> None:
        self._window_s = window_minutes * 60
        self._seen: deque[tuple[float, str]] = deque()  # (timestamp, hash)
        self._hashes: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Check if headline is a duplicate. Returns True if seen before."""
        now = time.time()
        self._evict_old(now)

        h = self._hash_text(text)
        if h in self._hashes:
            return True

        self._seen.append((now, h))
        self._hashes.add(h)
        return False

    def _evict_old(self, now: float) -> None:
        while self._seen and (now - self._seen[0][0]) > self._window_s:
            _, old_hash = self._seen.popleft()
            self._hashes.discard(old_hash)

    @staticmethod
    def _hash_text(text: str) -> str:
        """Normalize and hash headline for dedup."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        # Remove common prefixes like "BREAKING:" or "UPDATE:"
        normalized = re.sub(r"^(breaking|update|flash|alert)[:\s-]*", "", normalized)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]


class FinnhubWebSocket:
    """Finnhub WebSocket client for real-time news (~1s latency).

    Finnhub pushes news as they arrive — no polling needed.
    Free tier: 60 API calls/min, WebSocket with 50 symbols.
    News WebSocket endpoint: wss://ws.finnhub.io?token=API_KEY

    Protocol:
    - Send: {"type":"subscribe-news","symbol":"FOREX:EURUSD"}
    - Receive: {"data":[{"category":"forex","headline":"...","datetime":...}],"type":"news"}
    """

    def __init__(
        self,
        api_key: str,
        categories: list[str] | None = None,
        on_headline: Callable[[NewsHeadline], Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._categories = set(categories or ["forex", "general"])
        self._on_headline = on_headline
        self._ws = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    async def connect(self) -> None:
        """Start WebSocket connection with auto-reconnect."""
        self._running = True

        while self._running:
            try:
                await self._run_connection()
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    "Finnhub WS disconnected, reconnecting...",
                    error=str(e),
                    delay=self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay,
                )

    async def _run_connection(self) -> None:
        """Single WebSocket session."""
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets package not installed. "
                "Install with: pip install websockets"
            )
            self._running = False
            return

        url = f"wss://ws.finnhub.io?token={self._api_key}"
        logger.info("Connecting to Finnhub WebSocket...")

        async with websockets.connect(url, ping_interval=30) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0  # Reset on successful connect
            logger.info("Finnhub WebSocket connected")

            # Subscribe to news for forex symbols
            for symbol in ["FOREX:EURUSD", "FOREX:GBPUSD", "FOREX:USDJPY"]:
                subscribe_msg = json.dumps({
                    "type": "subscribe-news",
                    "symbol": symbol,
                })
                await ws.send(subscribe_msg)

            # Also subscribe to general news
            await ws.send(json.dumps({
                "type": "subscribe-news",
                "symbol": "",
            }))

            async for raw_msg in ws:
                if not self._running:
                    break

                receive_time = time.time()
                try:
                    msg = json.loads(raw_msg)
                    await self._handle_message(msg, receive_time)
                except json.JSONDecodeError:
                    continue

    async def _handle_message(self, msg: dict, receive_time: float) -> None:
        """Process incoming WebSocket message."""
        msg_type = msg.get("type", "")

        if msg_type == "news":
            for item in msg.get("data", []):
                headline = self._parse_news_item(item, receive_time)
                if headline and self._on_headline:
                    self._on_headline(headline)

        elif msg_type == "ping":
            # Heartbeat
            pass

    def _parse_news_item(
        self, item: dict, receive_time: float
    ) -> NewsHeadline | None:
        """Parse Finnhub news item into NewsHeadline."""
        headline_text = item.get("headline", "").strip()
        if not headline_text:
            return None

        category = item.get("category", "")
        if category and category not in self._categories:
            return None

        # Compute latency
        publish_ts = item.get("datetime", 0)
        if publish_ts:
            latency_ms = (receive_time - publish_ts) * 1000
        else:
            latency_ms = 0.0
            publish_ts = receive_time

        return NewsHeadline(
            text=headline_text,
            timestamp=datetime.fromtimestamp(publish_ts, tz=UTC),
            source="finnhub",
            category=category,
            url=item.get("url", ""),
            latency_ms=max(latency_ms, 0),
        )

    def stop(self) -> None:
        """Stop WebSocket connection."""
        self._running = False


class FastRSSPoller:
    """Aggressive RSS polling with 30-second interval.

    Much faster than the original 15-minute polling. Uses conditional
    HTTP requests (If-Modified-Since / ETag) to minimize bandwidth.
    Parses XML with regex (no lxml dependency).
    """

    def __init__(
        self,
        feeds: list[str],
        poll_interval_s: int = 30,
        on_headline: Callable[[NewsHeadline], Any] | None = None,
    ) -> None:
        self._feeds = feeds
        self._poll_interval = poll_interval_s
        self._on_headline = on_headline
        self._running = False
        # Per-feed last-modified / ETag for conditional requests
        self._feed_etags: dict[str, str] = {}
        self._feed_last_modified: dict[str, str] = {}

    async def run(self) -> None:
        """Main polling loop."""
        self._running = True
        logger.info(
            "Fast RSS poller started",
            interval_s=self._poll_interval,
            n_feeds=len(self._feeds),
        )

        while self._running:
            fetch_start = time.time()

            # Poll all feeds concurrently
            tasks = [self._poll_feed(url) for url in self._feeds]
            await asyncio.gather(*tasks, return_exceptions=True)

            elapsed = time.time() - fetch_start
            sleep_time = max(0, self._poll_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _poll_feed(self, url: str) -> None:
        """Poll a single RSS feed."""
        try:
            import requests
        except ImportError:
            return

        try:
            headers: dict[str, str] = {"User-Agent": "ApexFX-Quantum/2.0"}

            # Conditional request headers (save bandwidth)
            if url in self._feed_etags:
                headers["If-None-Match"] = self._feed_etags[url]
            if url in self._feed_last_modified:
                headers["If-Modified-Since"] = self._feed_last_modified[url]

            # Run blocking HTTP in thread pool
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: requests.get(url, timeout=8, headers=headers),
            )

            # 304 Not Modified — no new content
            if resp.status_code == 304:
                return

            if resp.status_code != 200:
                return

            # Cache headers for next request
            if "ETag" in resp.headers:
                self._feed_etags[url] = resp.headers["ETag"]
            if "Last-Modified" in resp.headers:
                self._feed_last_modified[url] = resp.headers["Last-Modified"]

            receive_time = time.time()
            source = self._extract_domain(url)
            headlines = self._parse_rss(resp.text, source, receive_time)

            for h in headlines:
                if self._on_headline:
                    self._on_headline(h)

        except Exception as e:
            logger.debug("RSS poll failed", feed=url, error=str(e))

    def _parse_rss(
        self, xml_text: str, source: str, receive_time: float
    ) -> list[NewsHeadline]:
        """Parse RSS XML into headlines."""
        headlines: list[NewsHeadline] = []
        items = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)

        for item in items[:20]:  # Max 20 per feed
            title_match = re.search(
                r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item
            )
            date_match = re.search(r"<pubDate>(.*?)</pubDate>", item)
            link_match = re.search(r"<link>(.*?)</link>", item)

            if not title_match:
                continue

            title = title_match.group(1).strip()
            title = re.sub(r"&amp;", "&", title)
            title = re.sub(r"&lt;", "<", title)
            title = re.sub(r"&gt;", ">", title)
            title = re.sub(r"&quot;", '"', title)
            title = re.sub(r"<[^>]+>", "", title)

            if not title:
                continue

            timestamp = datetime.now(UTC)
            latency_ms = 0.0
            if date_match:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(date_match.group(1))
                    timestamp = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
                    latency_ms = (receive_time - timestamp.timestamp()) * 1000
                except Exception:
                    pass

            url = ""
            if link_match:
                url = link_match.group(1).strip()

            headlines.append(NewsHeadline(
                text=title,
                timestamp=timestamp,
                source=source,
                url=url,
                latency_ms=max(latency_ms, 0),
            ))

        return headlines

    @staticmethod
    def _extract_domain(url: str) -> str:
        match = re.search(r"https?://(?:www\.)?([\w.-]+)", url)
        return match.group(1) if match else url

    def stop(self) -> None:
        self._running = False


class RealtimeNewsStream:
    """Unified real-time news stream with multi-source failover.

    Combines WebSocket (Finnhub) and fast RSS polling into a single
    async queue. The trading loop consumes from this queue.

    Usage:
        stream = RealtimeNewsStream(config.data.news)
        queue = stream.headline_queue

        # In trading loop
        asyncio.create_task(stream.start())

        # Consume
        while True:
            headline = await queue.get()
            sentiment_extractor.update_headlines([headline.to_dict()])
    """

    def __init__(self, config: NewsConfig) -> None:
        self._config = config
        self._headline_queue: asyncio.Queue[NewsHeadline] = asyncio.Queue(maxsize=200)
        self._dedup = HeadlineDeduplicator(
            window_minutes=config.dedup_window_minutes,
        )
        self._urgency_keywords = set(
            kw.lower() for kw in config.urgency_keywords
        )
        self._urgency_cooldown_s = config.urgency_cooldown_s
        self._last_urgency_time: float = 0.0
        self._running = False

        # Stats
        self._stats = {
            "total_received": 0,
            "duplicates_filtered": 0,
            "urgent_headlines": 0,
            "finnhub_count": 0,
            "rss_count": 0,
            "avg_latency_ms": 0.0,
        }
        self._latencies: deque[float] = deque(maxlen=100)

        # API key from config or env
        self._finnhub_key = config.finnhub_api_key or os.environ.get(
            "FINNHUB_API_KEY", ""
        )

        # Components
        self._finnhub_ws: FinnhubWebSocket | None = None
        self._rss_poller: FastRSSPoller | None = None

        if config.finnhub_enabled and self._finnhub_key:
            self._finnhub_ws = FinnhubWebSocket(
                api_key=self._finnhub_key,
                categories=config.finnhub_categories,
                on_headline=self._on_headline,
            )

        if config.rss_enabled:
            self._rss_poller = FastRSSPoller(
                feeds=config.rss_feeds,
                poll_interval_s=config.rss_poll_interval_s,
                on_headline=self._on_headline,
            )

    @property
    def headline_queue(self) -> asyncio.Queue[NewsHeadline]:
        """Queue for consuming headlines in the trading loop."""
        return self._headline_queue

    @property
    def stats(self) -> dict[str, Any]:
        """Current stream statistics."""
        return dict(self._stats)

    def _on_headline(self, headline: NewsHeadline) -> None:
        """Callback from any source. Dedup + urgency check + enqueue."""
        self._stats["total_received"] += 1

        # Deduplication
        if self._dedup.is_duplicate(headline.text):
            self._stats["duplicates_filtered"] += 1
            return

        # Urgency detection
        text_lower = headline.text.lower()
        if any(kw in text_lower for kw in self._urgency_keywords):
            now = time.time()
            if (now - self._last_urgency_time) > self._urgency_cooldown_s:
                headline.is_urgent = True
                self._last_urgency_time = now
                self._stats["urgent_headlines"] += 1
                logger.warning(
                    "URGENT NEWS DETECTED",
                    headline=headline.text[:100],
                    source=headline.source,
                    latency_ms=round(headline.latency_ms, 1),
                )

        # Track stats
        if headline.source == "finnhub":
            self._stats["finnhub_count"] += 1
        else:
            self._stats["rss_count"] += 1

        if headline.latency_ms > 0:
            self._latencies.append(headline.latency_ms)
            self._stats["avg_latency_ms"] = round(
                sum(self._latencies) / len(self._latencies), 1
            )

        # Enqueue (non-blocking, drop oldest if full)
        try:
            self._headline_queue.put_nowait(headline)
        except asyncio.QueueFull:
            try:
                dropped = self._headline_queue.get_nowait()  # Drop oldest
                self._headline_queue.put_nowait(headline)
                logger.warning(
                    "News queue full, dropped oldest headline",
                    dropped=dropped.text[:60],
                    queue_size=self._headline_queue.qsize(),
                )
            except asyncio.QueueEmpty:
                logger.error(
                    "Failed to enqueue headline after drop attempt",
                    headline=headline.text[:60],
                )

        logger.debug(
            "News headline received",
            source=headline.source,
            urgent=headline.is_urgent,
            latency_ms=round(headline.latency_ms, 1),
            text=headline.text[:80],
        )

    async def start(self) -> None:
        """Start all news sources concurrently."""
        self._running = True
        tasks: list[asyncio.Task] = []

        if self._finnhub_ws:
            tasks.append(asyncio.create_task(
                self._finnhub_ws.connect(),
                name="finnhub-ws",
            ))
            logger.info("Finnhub WebSocket news stream enabled (~1s latency)")

        if self._rss_poller:
            tasks.append(asyncio.create_task(
                self._rss_poller.run(),
                name="rss-poller",
            ))
            logger.info(
                "Fast RSS poller enabled",
                interval_s=self._config.rss_poll_interval_s,
            )

        if not tasks:
            logger.warning("No news sources configured!")
            return

        logger.info(
            "RealtimeNewsStream started",
            sources=len(tasks),
            finnhub=self._finnhub_ws is not None,
            rss=self._rss_poller is not None,
        )

        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        """Stop all news sources."""
        self._running = False
        if self._finnhub_ws:
            self._finnhub_ws.stop()
        if self._rss_poller:
            self._rss_poller.stop()
        logger.info("RealtimeNewsStream stopped", stats=self._stats)
