"""
Broker Factory - Easy creation of message broker instances.

Provides a unified API for creating brokers without importing specific classes.

Usage:
    >>> from sage.outbox.brokers import create_broker, get_available_brokers
    >>> 
    >>> # Check available brokers
    >>> print(get_available_brokers())
    ['memory', 'kafka', 'rabbitmq']
    >>> 
    >>> # Create a broker
    >>> broker = create_broker("kafka", bootstrap_servers="localhost:9092")
    >>> await broker.connect()
"""

from typing import Optional, Dict, List, Any

from sage.outbox.brokers.base import MessageBroker, BrokerError
from sage.outbox.brokers.memory import InMemoryBroker
from sage.exceptions import MissingDependencyError


def get_available_brokers() -> List[str]:
    """
    Get list of available broker backends.
    
    Returns:
        List of broker names that can be used
    """
    available = ["memory"]  # Always available
    
    try:
        from sage.outbox.brokers.kafka import KAFKA_AVAILABLE
        if KAFKA_AVAILABLE:
            available.append("kafka")
    except ImportError:
        pass  # pragma: no cover
    
    try:
        from sage.outbox.brokers.rabbitmq import RABBITMQ_AVAILABLE
        if RABBITMQ_AVAILABLE:
            available.append("rabbitmq")
    except ImportError:
        pass  # pragma: no cover
    
    try:
        from sage.outbox.brokers.redis import REDIS_AVAILABLE
        if REDIS_AVAILABLE:
            available.append("redis")
    except ImportError:
        pass  # pragma: no cover
    
    return available


def print_available_brokers() -> None:
    """Print available brokers with installation instructions."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                    Available Brokers                         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    # Memory (always available)
    print("║  ✅ memory     - In-memory (for testing)                     ║")
    
    # Kafka
    try:
        from sage.outbox.brokers.kafka import KAFKA_AVAILABLE
        if KAFKA_AVAILABLE:
            print("║  ✅ kafka      - Apache Kafka                               ║")
        else:
            print("║  ❌ kafka      - pip install aiokafka                       ║")  # pragma: no cover
    except ImportError:
        print("║  ❌ kafka      - pip install aiokafka                       ║")  # pragma: no cover
    
    # RabbitMQ
    try:
        from sage.outbox.brokers.rabbitmq import RABBITMQ_AVAILABLE
        if RABBITMQ_AVAILABLE:
            print("║  ✅ rabbitmq   - RabbitMQ/AMQP                               ║")
        else:
            print("║  ❌ rabbitmq   - pip install aio-pika                        ║")  # pragma: no cover
    except ImportError:
        print("║  ❌ rabbitmq   - pip install aio-pika                        ║")  # pragma: no cover
    
    # Redis
    try:
        from sage.outbox.brokers.redis import REDIS_AVAILABLE
        if REDIS_AVAILABLE:
            print("║  ✅ redis      - Redis Streams                                ║")
        else:
            print("║  ❌ redis      - pip install redis                            ║")  # pragma: no cover
    except ImportError:
        print("║  ❌ redis      - pip install redis                            ║")  # pragma: no cover
    
    print("╚══════════════════════════════════════════════════════════════╝\n")


def _create_kafka_broker(kwargs: dict):
    """Create Kafka broker instance."""
    from sage.outbox.brokers.kafka import KafkaBroker, KafkaBrokerConfig, KAFKA_AVAILABLE
    if not KAFKA_AVAILABLE:
        raise MissingDependencyError("aiokafka", "Kafka message broker")
    config = KafkaBrokerConfig(**kwargs) if kwargs else None
    return KafkaBroker(config)


def _create_rabbitmq_broker(kwargs: dict):
    """Create RabbitMQ broker instance."""
    from sage.outbox.brokers.rabbitmq import RabbitMQBroker, RabbitMQBrokerConfig, RABBITMQ_AVAILABLE
    if not RABBITMQ_AVAILABLE:
        raise MissingDependencyError("aio-pika", "RabbitMQ message broker")
    config = RabbitMQBrokerConfig(**kwargs) if kwargs else None
    return RabbitMQBroker(config)


def _create_redis_broker(kwargs: dict):
    """Create Redis broker instance."""
    from sage.outbox.brokers.redis import RedisBroker, RedisBrokerConfig, REDIS_AVAILABLE
    if not REDIS_AVAILABLE:
        raise MissingDependencyError("redis", "Redis message broker")
    config = RedisBrokerConfig(**kwargs) if kwargs else None
    return RedisBroker(config)


# Broker registry: type -> (factory_function, dependency_name)
_BROKER_REGISTRY = {
    "memory": (lambda _: InMemoryBroker(), None),
    "kafka": (_create_kafka_broker, "aiokafka"),
    "rabbitmq": (_create_rabbitmq_broker, "aio-pika"),
    "rabbit": (_create_rabbitmq_broker, "aio-pika"),
    "amqp": (_create_rabbitmq_broker, "aio-pika"),
    "redis": (_create_redis_broker, "redis"),
}


def create_broker(
    broker_type: str,
    **kwargs: Any,
) -> MessageBroker:
    """
    Create a message broker instance.
    
    Args:
        broker_type: Type of broker ('memory', 'kafka', 'rabbitmq', 'redis')
        **kwargs: Broker-specific configuration
    
    Returns:
        Configured broker instance
    
    Raises:
        MissingDependencyError: If required package is not installed
        ValueError: If broker type is unknown
    
    Examples:
        >>> # In-memory broker for testing
        >>> broker = create_broker("memory")
        >>> 
        >>> # Kafka broker
        >>> broker = create_broker("kafka", bootstrap_servers="localhost:9092")
        >>> 
        >>> # RabbitMQ broker
        >>> broker = create_broker("rabbitmq", url="amqp://guest:guest@localhost/")
    """
    broker_type = broker_type.lower().strip()
    
    if broker_type not in _BROKER_REGISTRY:
        available = get_available_brokers()
        raise ValueError(
            f"Unknown broker type: '{broker_type}'\n"
            f"Available brokers: {', '.join(available)}"
        )
    
    factory, dependency = _BROKER_REGISTRY[broker_type]
    
    try:
        return factory(kwargs)
    except ImportError:
        if dependency:
            raise MissingDependencyError(dependency, f"{broker_type} message broker")
        raise  # pragma: no cover


def create_broker_from_env() -> MessageBroker:
    """
    Create a message broker from environment variables.
    
    Reads BROKER_TYPE environment variable to determine which
    broker to create, then uses broker-specific env vars.
    
    Environment Variables:
        BROKER_TYPE: Broker type (kafka, rabbitmq, memory)
        
        For Kafka:
            KAFKA_BOOTSTRAP_SERVERS
            KAFKA_CLIENT_ID
            KAFKA_SASL_USERNAME
            KAFKA_SASL_PASSWORD
        
        For RabbitMQ:
            RABBITMQ_URL
            RABBITMQ_EXCHANGE
    
    Returns:
        Configured broker instance
    """
    import os
    
    broker_type = os.getenv("BROKER_TYPE", "memory").lower()
    
    if broker_type == "memory":
        return InMemoryBroker()
    
    elif broker_type == "kafka":
        from sage.outbox.brokers.kafka import KafkaBroker
        return KafkaBroker.from_env()
    
    elif broker_type in ("rabbitmq", "rabbit", "amqp"):
        from sage.outbox.brokers.rabbitmq import RabbitMQBroker
        return RabbitMQBroker.from_env()
    
    elif broker_type == "redis":
        from sage.outbox.brokers.redis import RedisBroker
        return RedisBroker.from_env()
    
    else:
        raise ValueError(f"Unknown BROKER_TYPE: {broker_type}")
