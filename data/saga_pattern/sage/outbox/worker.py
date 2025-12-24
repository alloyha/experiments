"""
Outbox Worker - Background processor for outbox events.

Polls the outbox for pending events and publishes them to the message broker.
Handles retries, dead-letter queue, and graceful shutdown.

Usage:
    >>> from sage.outbox import OutboxWorker, InMemoryOutboxStorage, InMemoryBroker
    >>> 
    >>> storage = InMemoryOutboxStorage()
    >>> broker = InMemoryBroker()
    >>> await broker.connect()
    >>> 
    >>> worker = OutboxWorker(storage, broker)
    >>> await worker.start()  # Runs until stopped
    >>> # or
    >>> await worker.process_batch()  # Process one batch
"""

import asyncio
import logging
import os
import signal
import sys
import uuid
from collections.abc import Awaitable, Callable

from sage.outbox.brokers.base import BrokerError, MessageBroker
from sage.outbox.state_machine import OutboxStateMachine
from sage.outbox.storage.base import OutboxStorage
from sage.outbox.types import OutboxConfig, OutboxEvent, OutboxStatus

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OutboxWorker:
    """
    Background worker that processes outbox events.
    
    Features:
        - Batch processing for efficiency
        - Parallel publish within batches
        - Exponential backoff on failures
        - Graceful shutdown on SIGTERM/SIGINT
        - Stuck event recovery
        - Dead letter queue handling
    
    Usage:
        >>> worker = OutboxWorker(storage, broker, config)
        >>> 
        >>> # Run continuously
        >>> await worker.start()
        >>> 
        >>> # Or process manually
        >>> processed = await worker.process_batch()
        >>> print(f"Processed {processed} events")
    
    Lifecycle:
        1. Claim batch of PENDING events (with SKIP LOCKED)
        2. Publish each event to broker in parallel
        3. Mark successful events as SENT
        4. Mark failed events as FAILED (retry later)
        5. Move exceeded-retry events to DEAD_LETTER
        6. Sleep and repeat
    """

    def __init__(
        self,
        storage: OutboxStorage,
        broker: MessageBroker,
        config: OutboxConfig | None = None,
        worker_id: str | None = None,
        on_event_published: Callable[[OutboxEvent], Awaitable[None]] | None = None,
        on_event_failed: Callable[[OutboxEvent, Exception], Awaitable[None]] | None = None,
    ):
        """
        Initialize the outbox worker.
        
        Args:
            storage: Outbox storage implementation
            broker: Message broker implementation
            config: Worker configuration
            worker_id: Unique ID for this worker (auto-generated if not provided)
            on_event_published: Callback when event is published successfully
            on_event_failed: Callback when event fails to publish
        """
        self.storage = storage
        self.broker = broker
        self.config = config or OutboxConfig()
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

        self._state_machine = OutboxStateMachine(max_retries=self.config.max_retries)
        self._running = False
        self._shutdown_event = asyncio.Event()

        self._on_event_published = on_event_published
        self._on_event_failed = on_event_failed

        # Metrics
        self._events_processed = 0
        self._events_failed = 0
        self._events_dead_lettered = 0

    async def start(self) -> None:
        """
        Start the worker loop.
        
        Runs continuously until stop() is called or shutdown signal received.
        """
        self._running = True
        self._shutdown_event.clear()

        logger.info(f"Outbox worker {self.worker_id} starting")

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:  # pragma: no cover
                # Windows doesn't support add_signal_handler
                pass

        try:
            while self._running:
                try:
                    # Process a batch
                    processed = await self.process_batch()

                    if processed == 0:
                        # No events to process, wait before polling again
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self.config.poll_interval_seconds
                        )
                except TimeoutError:
                    # Normal timeout from wait_for, continue loop
                    pass
                except asyncio.CancelledError:  # pragma: no cover
                    logger.info(f"Worker {self.worker_id} cancelled")
                    break
                except Exception as e:  # pragma: no cover
                    logger.error(f"Worker {self.worker_id} error: {e}")
                    await asyncio.sleep(self.config.poll_interval_seconds)
        finally:
            self._running = False
            logger.info(f"Outbox worker {self.worker_id} stopped")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self.worker_id}")
        self._running = False
        self._shutdown_event.set()

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info(f"Shutdown signal received for worker {self.worker_id}")
        asyncio.create_task(self.stop())

    async def process_batch(self) -> int:
        """
        Process a single batch of events.
        
        Returns:
            Number of events processed
        """
        # Claim batch of events
        events = await self.storage.claim_batch(
            worker_id=self.worker_id,
            batch_size=self.config.batch_size,
        )

        if not events:  # pragma: no cover
            return 0

        logger.debug(f"Worker {self.worker_id} claimed {len(events)} events")

        # Process events in parallel
        tasks = [self._process_event(event) for event in events]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        processed = 0
        for event, result in zip(events, results):
            if isinstance(result, Exception):  # pragma: no cover
                logger.error(f"Failed to process event {event.event_id}: {result}")
            else:
                processed += 1

        return processed

    async def _process_event(self, event: OutboxEvent) -> None:
        """
        Process a single event.
        
        Args:
            event: The event to process
        """
        try:
            # Publish to broker
            await self.broker.publish_event(event)

            # Mark as sent
            await self.storage.update_status(
                event.event_id,
                OutboxStatus.SENT,
            )

            self._events_processed += 1

            if self._on_event_published:
                await self._on_event_published(event)

            logger.debug(f"Event {event.event_id} published successfully")

        except BrokerError as e:  # pragma: no cover
            await self._handle_publish_failure(event, e)
        except Exception as e:  # pragma: no cover
            await self._handle_publish_failure(event, e)

    async def _handle_publish_failure(
        self,
        event: OutboxEvent,
        error: Exception,
    ) -> None:
        """
        Handle a publish failure.
        
        Args:
            event: The event that failed
            error: The exception that occurred
        """
        error_message = str(error)

        logger.warning(
            f"Event {event.event_id} failed to publish: {error_message} "
            f"(attempt {event.retry_count + 1}/{self.config.max_retries})"
        )

        # Update to failed status
        event = await self.storage.update_status(
            event.event_id,
            OutboxStatus.FAILED,
            error_message=error_message,
        )

        self._events_failed += 1

        if self._on_event_failed:  # pragma: no cover
            await self._on_event_failed(event, error)

        # Check if should move to dead letter
        if event.retry_count >= self.config.max_retries:
            await self._move_to_dead_letter(event)
        else:
            # Reset to pending for retry
            await self.storage.update_status(
                event.event_id,
                OutboxStatus.PENDING,
            )

    async def _move_to_dead_letter(self, event: OutboxEvent) -> None:
        """
        Move an event to the dead letter queue.
        
        Args:
            event: The event to move
        """
        await self.storage.update_status(
            event.event_id,
            OutboxStatus.DEAD_LETTER,
        )

        self._events_dead_lettered += 1

        logger.error(
            f"Event {event.event_id} moved to dead letter queue "
            f"after {event.retry_count} attempts. "
            f"Last error: {event.last_error}"
        )

    async def recover_stuck_events(self) -> int:
        """
        Recover events that appear stuck.
        
        This should be called periodically to handle events
        claimed by crashed workers.
        
        Returns:
            Number of events recovered
        """
        count = await self.storage.release_stuck_events(
            claimed_older_than_seconds=self.config.claim_timeout_seconds
        )

        if count > 0:
            logger.info(f"Recovered {count} stuck events")

        return count

    def get_stats(self) -> dict:
        """
        Get worker statistics.
        
        Returns:
            Dictionary of stats
        """
        return {
            "worker_id": self.worker_id,
            "running": self._running,
            "events_processed": self._events_processed,
            "events_failed": self._events_failed,
            "events_dead_lettered": self._events_dead_lettered,
        }



def get_storage():
    """Create storage backend from environment."""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    from sage.outbox.storage.postgresql import PostgreSQLOutboxStorage
    return PostgreSQLOutboxStorage(connection_string=database_url)


def get_broker():
    """Create broker from environment."""
    broker_type = os.getenv("BROKER_TYPE", "kafka").lower()

    # Check for broker-specific URL first, then generic BROKER_URL
    if broker_type == "rabbitmq":
        broker_url = os.getenv("RABBITMQ_URL") or os.getenv("BROKER_URL")
    else:
        broker_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS") or os.getenv("BROKER_URL")

    if not broker_url:
        logger.error(f"No broker URL configured. Set BROKER_URL or {broker_type.upper()}_URL")
        sys.exit(1)

    if broker_type == "kafka":
        from sage.outbox.brokers.kafka import KafkaBroker, KafkaBrokerConfig
        config = KafkaBrokerConfig(
            bootstrap_servers=broker_url,
            topic=os.getenv("KAFKA_TOPIC", "saga-events"),
        )
        return KafkaBroker(config=config)
    if broker_type == "rabbitmq":
        from sage.outbox.brokers.rabbitmq import RabbitMQBroker, RabbitMQBrokerConfig
        config = RabbitMQBrokerConfig(
            url=broker_url,
            exchange_name=os.getenv("RABBITMQ_EXCHANGE", "saga-events"),
        )
        return RabbitMQBroker(config=config)
    logger.error(f"Unknown broker type: {broker_type}")
    sys.exit(1)


async def main():
    """Main entry point."""
    logger.info("Starting Sage Outbox Worker...")

    # Create storage and broker
    storage = get_storage()
    broker = get_broker()

    # Create worker config
    from sage.outbox.types import OutboxConfig
    config = OutboxConfig(
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        poll_interval_seconds=float(os.getenv("POLL_INTERVAL", "1.0")),
        max_retries=int(os.getenv("MAX_RETRIES", "5")),
    )

    # Create worker
    from sage.outbox.worker import OutboxWorker
    worker = OutboxWorker(
        storage=storage,
        broker=broker,
        config=config,
        worker_id=os.getenv("WORKER_ID"),
    )

    try:
        # Initialize connections
        logger.info("Initializing storage...")
        await storage.initialize()

        logger.info("Connecting to broker...")
        await broker.connect()

        logger.info(f"Worker {worker.worker_id} starting...")
        await worker.start()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    finally:
        logger.info("Shutting down...")
        await worker.stop()
        await broker.close()
        await storage.close()
        logger.info("Worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())
