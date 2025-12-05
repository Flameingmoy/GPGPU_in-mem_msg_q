"""
Track A Consumer: Fetches messages from Redis Stream using consumer groups.

Uses XREADGROUP for at-least-once delivery with XACK for acknowledgment.
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import uuid
import redis


@dataclass
class Message:
    """A message fetched from Redis Stream."""
    stream_id: str  # Redis stream entry ID (e.g., "1234567890123-0")
    payload: bytes  # Message payload
    timestamp: float = field(default_factory=time.time)
    
    def __repr__(self):
        return f"Message(id={self.stream_id}, len={len(self.payload)})"


@dataclass
class Batch:
    """A batch of messages for GPU processing."""
    messages: list[Message]
    stream_name: str
    consumer_group: str
    
    def __len__(self):
        return len(self.messages)
    
    def __iter__(self):
        return iter(self.messages)
    
    @property
    def ids(self) -> list[str]:
        """Get all stream IDs in this batch."""
        return [m.stream_id for m in self.messages]
    
    @property
    def payloads(self) -> list[bytes]:
        """Get all payloads in this batch."""
        return [m.payload for m in self.messages]
    
    @property
    def total_bytes(self) -> int:
        """Total payload bytes in batch."""
        return sum(len(m.payload) for m in self.messages)


@dataclass
class ConsumerConfig:
    """Configuration for Track A consumer."""
    stream_name: str
    group_name: str
    consumer_id: str
    host: str = "localhost"
    port: int = 6379
    batch_size: int = 64
    block_ms: int = 1000  # Block timeout for XREADGROUP


class Consumer:
    """
    Redis Stream consumer with consumer group support.
    
    Uses XREADGROUP for reliable message delivery:
    - Messages are assigned to this consumer
    - Must ACK after processing to remove from pending
    - Unacked messages can be claimed by other consumers
    """
    
    def __init__(
        self,
        stream_name: str,
        group: str,
        consumer_id: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        batch_size: int = 64,
        block_ms: int = 1000,
    ):
        """
        Initialize consumer.
        
        Args:
            stream_name: Redis Stream name
            group: Consumer group name
            consumer_id: Unique consumer ID (auto-generated if None)
            host: Redis host
            port: Redis port
            batch_size: Default batch size for fetch_batch
            block_ms: Block timeout in milliseconds
        """
        self.config = ConsumerConfig(
            stream_name=stream_name,
            group_name=group,
            consumer_id=consumer_id or f"consumer-{uuid.uuid4().hex[:8]}",
            host=host,
            port=port,
            batch_size=batch_size,
            block_ms=block_ms,
        )
        
        self._client = redis.Redis(
            host=host,
            port=port,
            decode_responses=False,
        )
        
        self._ack_count: int = 0
        self._fetch_count: int = 0
        
        # Ensure consumer group exists
        self._ensure_group()
    
    def _ensure_group(self):
        """Create consumer group if it doesn't exist."""
        try:
            self._client.xgroup_create(
                self.config.stream_name,
                self.config.group_name,
                id="0",  # Start from beginning
                mkstream=True,  # Create stream if doesn't exist
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            # Group already exists, that's fine
    
    def fetch_batch(
        self,
        max_messages: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ) -> Optional[Batch]:
        """
        Fetch a batch of messages from the stream.
        
        Args:
            max_messages: Maximum messages to fetch (default: config.batch_size)
            timeout_ms: Block timeout in ms (default: config.block_ms)
            
        Returns:
            Batch of messages, or None if timeout with no messages
        """
        count = max_messages or self.config.batch_size
        block = timeout_ms if timeout_ms is not None else self.config.block_ms
        
        # XREADGROUP: fetch new messages assigned to this consumer
        result = self._client.xreadgroup(
            groupname=self.config.group_name,
            consumername=self.config.consumer_id,
            streams={self.config.stream_name: ">"},  # ">" = only new messages
            count=count,
            block=block,
        )
        
        if not result:
            return None
        
        # Parse result: [(stream_name, [(id, {fields}), ...])]
        messages = []
        for stream_name, entries in result:
            for entry_id, fields in entries:
                stream_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                payload = fields.get(b"payload", b"")
                messages.append(Message(stream_id=stream_id, payload=payload))
        
        self._fetch_count += len(messages)
        
        if not messages:
            return None
        
        return Batch(
            messages=messages,
            stream_name=self.config.stream_name,
            consumer_group=self.config.group_name,
        )
    
    def ack(self, batch: Batch) -> int:
        """
        Acknowledge processed messages.
        
        Args:
            batch: Batch to acknowledge
            
        Returns:
            Number of messages acknowledged
        """
        if not batch.messages:
            return 0
        
        count = self._client.xack(
            batch.stream_name,
            batch.consumer_group,
            *batch.ids,
        )
        
        self._ack_count += count
        return count
    
    def ack_one(self, stream_id: str) -> bool:
        """Acknowledge a single message by ID."""
        count = self._client.xack(
            self.config.stream_name,
            self.config.group_name,
            stream_id,
        )
        if count:
            self._ack_count += 1
        return count > 0
    
    def pending_count(self) -> int:
        """Get count of pending (unacked) messages for this consumer."""
        try:
            info = self._client.xpending(
                self.config.stream_name,
                self.config.group_name,
            )
            return info.get("pending", 0) if info else 0
        except redis.ResponseError:
            return 0
    
    def claim_pending(
        self,
        min_idle_ms: int = 60000,
        max_messages: int = 10,
    ) -> Optional[Batch]:
        """
        Claim pending messages from other consumers that have timed out.
        
        Args:
            min_idle_ms: Minimum idle time before claiming
            max_messages: Maximum messages to claim
            
        Returns:
            Batch of claimed messages, or None
        """
        # XAUTOCLAIM: automatically claim idle pending messages
        try:
            result = self._client.xautoclaim(
                self.config.stream_name,
                self.config.group_name,
                self.config.consumer_id,
                min_idle_time=min_idle_ms,
                count=max_messages,
            )
            
            if not result or not result[1]:
                return None
            
            # result = (next_start_id, [(id, fields), ...], deleted_ids)
            messages = []
            for entry_id, fields in result[1]:
                stream_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                payload = fields.get(b"payload", b"")
                messages.append(Message(stream_id=stream_id, payload=payload))
            
            if not messages:
                return None
            
            return Batch(
                messages=messages,
                stream_name=self.config.stream_name,
                consumer_group=self.config.group_name,
            )
        except redis.ResponseError:
            return None
    
    @property
    def stats(self) -> dict:
        """Get consumer statistics."""
        return {
            "consumer_id": self.config.consumer_id,
            "group": self.config.group_name,
            "stream": self.config.stream_name,
            "fetched": self._fetch_count,
            "acked": self._ack_count,
            "pending": self.pending_count(),
        }
    
    def close(self):
        """Close Redis connection."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
