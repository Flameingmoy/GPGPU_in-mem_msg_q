"""
Track A Producer: Sends messages to Redis Stream.

Uses Redis Streams (XADD) for reliable message delivery with automatic IDs.
"""

import time
from dataclasses import dataclass

import redis


@dataclass
class ProducerConfig:
    """Configuration for Track A producer."""

    stream_name: str
    max_len: int = 10000  # Max stream length (approximate, uses ~)
    host: str = "localhost"
    port: int = 6379
    rate_limit: float | None = None  # Messages per second, None = unlimited


class Producer:
    """
    Redis Stream producer for Track A validation.

    Sends messages to a Redis Stream using XADD with automatic ID generation.
    Supports rate limiting for controlled testing.
    """

    def __init__(
        self,
        stream_name: str,
        host: str = "localhost",
        port: int = 6379,
        max_len: int = 10000,
        rate_limit: float | None = None,
    ):
        """
        Initialize producer.

        Args:
            stream_name: Redis Stream name
            host: Redis host
            port: Redis port
            max_len: Maximum stream length (approximate trimming)
            rate_limit: Max messages per second (None = unlimited)
        """
        self.config = ProducerConfig(
            stream_name=stream_name,
            host=host,
            port=port,
            max_len=max_len,
            rate_limit=rate_limit,
        )

        self._client = redis.Redis(
            host=host,
            port=port,
            decode_responses=False,  # Keep bytes
        )

        self._last_send_time: float = 0
        self._send_count: int = 0

    def send(self, payload: bytes, msg_id: str | None = None) -> str:
        """
        Send a message to the stream.

        Args:
            payload: Message payload (bytes)
            msg_id: Optional message ID (default: auto-generated "*")

        Returns:
            Stream entry ID (e.g., "1234567890123-0")
        """
        # Rate limiting
        if self.config.rate_limit:
            min_interval = 1.0 / self.config.rate_limit
            elapsed = time.perf_counter() - self._last_send_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        # XADD with automatic trimming
        entry_id = self._client.xadd(
            self.config.stream_name,
            {"payload": payload},
            id=msg_id or "*",
            maxlen=self.config.max_len,
            approximate=True,
        )

        self._last_send_time = time.perf_counter()
        self._send_count += 1

        return entry_id.decode() if isinstance(entry_id, bytes) else entry_id

    def send_batch(self, payloads: list[bytes]) -> list[str]:
        """
        Send multiple messages using pipeline for efficiency.

        Args:
            payloads: List of message payloads

        Returns:
            List of stream entry IDs
        """
        pipe = self._client.pipeline()

        for payload in payloads:
            pipe.xadd(
                self.config.stream_name,
                {"payload": payload},
                id="*",
                maxlen=self.config.max_len,
                approximate=True,
            )

        results = pipe.execute()
        self._send_count += len(payloads)

        return [r.decode() if isinstance(r, bytes) else r for r in results]

    @property
    def send_count(self) -> int:
        """Total messages sent."""
        return self._send_count

    def stream_info(self) -> dict:
        """Get stream information."""
        try:
            info = self._client.xinfo_stream(self.config.stream_name)
            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0),
            }
        except redis.ResponseError:
            return {"length": 0, "error": "Stream does not exist"}

    def close(self):
        """Close Redis connection."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
